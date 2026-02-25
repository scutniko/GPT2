"""
批量将历史 checkpoint 迁移为 v2 schema（纯数据格式）。

该脚本内置旧版兼容逻辑，用于一次性迁移历史文件。
主训练/评估代码仍保持仅支持 v2。
"""

import argparse
import glob
import importlib
import os
import shutil
import sys
import types
from dataclasses import field, fields, make_dataclass

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import GPTConfig

SCHEMA_V2 = 2


def _install_legacy_aliases():
    # 兼容旧根包名 GPT2.*
    if "GPT2" not in sys.modules:
        sys.modules["GPT2"] = types.ModuleType("GPT2")
    try:
        sys.modules["GPT2.core"] = importlib.import_module("core")
        sys.modules["GPT2.core.config"] = importlib.import_module("core.config")
        sys.modules["GPT2.core.data_loader"] = importlib.import_module("core.data_loader")
        sys.modules["GPT2.models"] = importlib.import_module("models")
        sys.modules["GPT2.models.gpt"] = importlib.import_module("models.gpt")
        sys.modules["GPT2.modules"] = importlib.import_module("modules")
        sys.modules["GPT2.modules.position_encodings"] = importlib.import_module(
            "modules.position_encodings"
        )
    except Exception:
        pass

    # 兼容已删除的 experiments.* dataclass 配置
    if "experiments" not in sys.modules:
        pkg = types.ModuleType("experiments")
        pkg.__path__ = []
        sys.modules["experiments"] = pkg

    def _ensure_legacy_config_module(module_name, class_name, extra_defaults):
        if module_name in sys.modules:
            return
        extra_fields = []
        for key, value in extra_defaults.items():
            extra_fields.append((key, type(value), field(default=value)))
        cls = make_dataclass(class_name, extra_fields, bases=(GPTConfig,))
        cls.__module__ = module_name
        module = types.ModuleType(module_name)
        setattr(module, class_name, cls)
        sys.modules[module_name] = module

    _ensure_legacy_config_module("experiments.gqa", "GQAConfig", {"n_kv_head": 4})
    _ensure_legacy_config_module(
        "experiments.mla",
        "MLAConfig",
        {"kv_lora_rank": 192, "q_lora_rank": 384},
    )
    _ensure_legacy_config_module(
        "experiments.moe",
        "MoEConfig",
        {
            "n_experts": 4,
            "moe_top_k": 1,
            "moe_capacity_factor": 1.0,
            "moe_router_noise": 0.0,
            "moe_expert_type": "mlp",
        },
    )


def _torch_load_with_fallback(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location=map_location, weights_only=False)


def load_checkpoint_legacy_compatible(path, map_location="cpu"):
    try:
        return _torch_load_with_fallback(path, map_location=map_location)
    except (ModuleNotFoundError, ImportError, AttributeError):
        _install_legacy_aliases()
        return _torch_load_with_fallback(path, map_location=map_location)


def get_schema_version(ckpt):
    ver = ckpt.get("schema_version", None)
    if isinstance(ver, int) and ver >= 1:
        return ver
    if "config_dict" in ckpt:
        return 2
    return 1


def serialize_config(config):
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    raise TypeError(f"不支持的配置类型: {type(config)}")


def deserialize_config(config_dict):
    if not isinstance(config_dict, dict):
        raise TypeError(f"config_dict 必须是 dict，实际是: {type(config_dict)}")
    core_field_names = {f.name for f in fields(GPTConfig)}
    core_kwargs = {}
    extra_kwargs = {}
    for key, value in config_dict.items():
        if key in core_field_names:
            core_kwargs[key] = value
        else:
            extra_kwargs[key] = value
    cfg = GPTConfig(**core_kwargs)
    for key, value in extra_kwargs.items():
        setattr(cfg, key, value)
    return cfg


def extract_config_any_version(ckpt):
    ver = get_schema_version(ckpt)
    if ver >= 2 and "config_dict" in ckpt:
        return deserialize_config(ckpt["config_dict"])
    if "config" in ckpt:
        return deserialize_config(serialize_config(ckpt["config"]))
    if "config_dict" in ckpt:
        return deserialize_config(ckpt["config_dict"])
    raise KeyError("checkpoint 中既没有 config_dict 也没有 config")


def _infer_experiment_name_from_path(path):
    if not path:
        return None
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)
    for i, p in enumerate(parts):
        if p == "log_train" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def convert_to_v2(ckpt, source_path=None):
    if "model" not in ckpt:
        raise KeyError("checkpoint 缺少必填字段: model")
    src_ver = get_schema_version(ckpt)
    cfg = extract_config_any_version(ckpt)
    out = {
        "schema_version": SCHEMA_V2,
        "model": ckpt["model"],
        "config_dict": serialize_config(cfg),
    }
    if "step" in ckpt:
        out["step"] = ckpt["step"]
    if "val_loss" in ckpt:
        out["val_loss"] = ckpt["val_loss"]
    if "optimizer" in ckpt:
        out["optimizer"] = ckpt["optimizer"]
    if "train_loader_state" in ckpt:
        out["train_loader_state"] = ckpt["train_loader_state"]
    if "rng_state" in ckpt:
        out["rng_state"] = ckpt["rng_state"]
    if "cuda_rng_state_all" in ckpt:
        out["cuda_rng_state_all"] = ckpt["cuda_rng_state_all"]
    if "experiment_name" in ckpt:
        out["experiment_name"] = ckpt["experiment_name"]
    else:
        inferred = _infer_experiment_name_from_path(source_path)
        if inferred is not None:
            out["experiment_name"] = inferred
    if "config_ref" in ckpt:
        out["config_ref"] = ckpt["config_ref"]
    out["source_schema_version"] = src_ver
    return out


def _collect_targets(input_path, pattern, recursive):
    if os.path.isfile(input_path):
        return [os.path.abspath(input_path)]
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    if recursive:
        p = os.path.join(input_path, "**", pattern)
        files = glob.glob(p, recursive=True)
    else:
        p = os.path.join(input_path, pattern)
        files = glob.glob(p, recursive=False)
    files = [os.path.abspath(x) for x in files if os.path.isfile(x)]
    return sorted(files)


def _resolve_output_path(src_path, input_path, output_root):
    if output_root is None:
        return src_path
    output_root = os.path.abspath(output_root)
    if os.path.isfile(input_path):
        return os.path.join(output_root, os.path.basename(src_path))
    rel = os.path.relpath(src_path, start=os.path.abspath(input_path))
    return os.path.join(output_root, rel)


def _atomic_save(obj, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = dst_path + ".tmp_v2"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, dst_path)


def main():
    parser = argparse.ArgumentParser(description="批量迁移 checkpoint 到 v2 schema")
    parser.add_argument("--input", required=True, type=str, help="输入 checkpoint 文件或目录")
    parser.add_argument("--pattern", type=str, default="model_*.pt", help="目录扫描时的文件名匹配模式")
    parser.add_argument("--no_recursive", action="store_true", help="关闭递归扫描（默认递归）")
    parser.add_argument("--output_root", type=str, default=None, help="输出根目录（不填则就地覆盖）")
    parser.add_argument("--dry_run", action="store_true", help="仅打印将执行的操作，不写文件")
    parser.add_argument("--force", action="store_true", help="即使已是 v2 也重新规范化并写出")
    parser.add_argument("--backup", action="store_true", help="就地覆盖时先备份原文件")
    parser.add_argument("--backup_suffix", type=str, default=".v1.bak", help="备份后缀（仅 --backup 生效）")
    parser.add_argument("--overwrite_backup", action="store_true", help="允许覆盖已存在的备份文件")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    recursive = not args.no_recursive
    targets = _collect_targets(input_path, args.pattern, recursive=recursive)
    if not targets:
        print("未找到任何匹配的 checkpoint 文件")
        return

    print(f"发现 {len(targets)} 个候选 checkpoint")
    if args.output_root:
        print(f"输出目录: {os.path.abspath(args.output_root)}")
    else:
        print("输出模式: 就地覆盖")

    converted = 0
    skipped = 0
    failed = 0
    for src in targets:
        dst = _resolve_output_path(src, input_path, args.output_root)
        try:
            ckpt = load_checkpoint_legacy_compatible(src, map_location="cpu")
            src_ver = get_schema_version(ckpt)
            if src_ver >= 2 and not args.force:
                print(f"[跳过] 已是 v2: {src}")
                skipped += 1
                continue

            v2_payload = convert_to_v2(ckpt, source_path=src)
            if args.dry_run:
                print(f"[预览] v{src_ver} -> v2: {src} -> {dst}")
                converted += 1
                continue

            if args.output_root is None and args.backup:
                backup_path = src + args.backup_suffix
                if os.path.exists(backup_path) and not args.overwrite_backup:
                    raise FileExistsError(
                        f"备份已存在（可加 --overwrite_backup）: {backup_path}"
                    )
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                shutil.copy2(src, backup_path)

            _atomic_save(v2_payload, dst)
            print(f"[完成] v{src_ver} -> v2: {src} -> {dst}")
            converted += 1
        except Exception as e:
            print(f"[失败] {src} | {type(e).__name__}: {e}")
            failed += 1

    print("-" * 60)
    print(f"总数: {len(targets)} | 已转换: {converted} | 跳过: {skipped} | 失败: {failed}")
    if args.dry_run:
        print("当前为 dry-run，未写入任何文件")


if __name__ == "__main__":
    main()

