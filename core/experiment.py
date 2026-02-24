"""
实验配置加载（YAML + 继承合并 + registry）。
"""

import copy
import glob
import os
from dataclasses import dataclass, fields

import yaml

from core.config import GPTConfig
from core.registry import resolve_component

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EXPERIMENT_CONFIG_DIR = os.path.join(REPO_ROOT, "configs", "experiments")


@dataclass
class ExperimentSpec:
    experiment_name: str
    model_config: object
    attention_class: type
    position_encoding_class: type
    train_config: dict
    mlp_class: type | None = None
    norm_class: type | None = None


def _list_available_experiment_configs():
    pattern = os.path.join(DEFAULT_EXPERIMENT_CONFIG_DIR, "*.yaml")
    names = []
    for path in sorted(glob.glob(pattern)):
        names.append(os.path.splitext(os.path.basename(path))[0])
    return names


def get_available_experiments_text():
    """返回可用 YAML 实验配置列表。"""
    names = _list_available_experiment_configs()
    if not names:
        return "当前未发现可用实验配置（configs/experiments/*.yaml）"
    return "可用实验配置: " + ", ".join(names)


def _expand_ref_candidates(config_ref, roots):
    candidates = []
    ref = str(config_ref).strip()
    if not ref:
        return candidates
    variants = [ref]
    if not os.path.splitext(ref)[1]:
        variants.append(f"{ref}.yaml")
        variants.append(f"{ref}.yml")
    for item in variants:
        candidates.append(item)
        for root in roots:
            candidates.append(os.path.join(root, item))
    # 去重并保持顺序
    seen = set()
    deduped = []
    for cand in candidates:
        abs_cand = os.path.abspath(cand)
        if abs_cand in seen:
            continue
        seen.add(abs_cand)
        deduped.append(abs_cand)
    return deduped


def resolve_experiment_config_path(config_ref):
    """解析实验配置路径，支持绝对路径/相对路径/短名。"""
    roots = [os.getcwd(), DEFAULT_EXPERIMENT_CONFIG_DIR, REPO_ROOT]
    for cand in _expand_ref_candidates(config_ref, roots):
        if os.path.isfile(cand):
            return cand
    raise ValueError(
        f"错误: 找不到配置 '{config_ref}'\n"
        f"{get_available_experiments_text()}"
    )


def _resolve_base_config_path(base_ref, current_config_dir):
    roots = [current_config_dir, DEFAULT_EXPERIMENT_CONFIG_DIR, REPO_ROOT, os.getcwd()]
    for cand in _expand_ref_candidates(base_ref, roots):
        if os.path.isfile(cand):
            return cand
    raise ValueError(f"错误: 找不到 base 配置 '{base_ref}'")


def _deep_merge_dict(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是字典结构: {path}")
    return data


def _load_config_with_bases(path, loading_stack):
    abs_path = os.path.abspath(path)
    if abs_path in loading_stack:
        chain = " -> ".join(loading_stack + [abs_path])
        raise ValueError(f"检测到循环继承: {chain}")

    data = _load_yaml_dict(abs_path)
    bases = data.pop("base", [])
    if isinstance(bases, str):
        bases = [bases]
    if bases is None:
        bases = []
    if not isinstance(bases, list):
        raise ValueError(f"base 字段必须是字符串或列表: {abs_path}")

    merged = {}
    current_dir = os.path.dirname(abs_path)
    for base_ref in bases:
        base_path = _resolve_base_config_path(base_ref, current_dir)
        base_cfg = _load_config_with_bases(base_path, loading_stack + [abs_path])
        merged = _deep_merge_dict(merged, base_cfg)
    merged = _deep_merge_dict(merged, data)
    return merged


def _build_model_config(model_dict):
    if not isinstance(model_dict, dict):
        raise ValueError("model 配置必须是字典")

    gpt_fields = {f.name for f in fields(GPTConfig)}
    core_kwargs = {}
    extra_kwargs = {}
    for key, value in model_dict.items():
        if key in gpt_fields:
            core_kwargs[key] = value
        else:
            extra_kwargs[key] = value

    cfg = GPTConfig(**core_kwargs)
    for key, value in extra_kwargs.items():
        setattr(cfg, key, value)
    return cfg


def load_experiment_spec(config_ref):
    """
    从 YAML 加载实验配置，并转成统一 ExperimentSpec。
    """
    config_path = resolve_experiment_config_path(config_ref)
    merged = _load_config_with_bases(config_path, loading_stack=[])

    for key in ("experiment_name", "model", "components", "train"):
        if key not in merged:
            raise ValueError(f"配置缺少必填字段 '{key}': {config_path}")

    components = merged["components"]
    if not isinstance(components, dict):
        raise ValueError(f"components 必须是字典: {config_path}")

    attention_key = components.get("attention")
    position_key = components.get("position_encoding")
    if attention_key is None:
        raise ValueError(f"components.attention 缺失: {config_path}")
    if position_key is None:
        raise ValueError(f"components.position_encoding 缺失: {config_path}")

    mlp_key = components.get("mlp", "default")
    norm_key = components.get("norm", "default")

    attention_class = resolve_component("attention", attention_key)
    position_encoding_class = resolve_component("position_encoding", position_key)

    mlp_class = None
    if str(mlp_key).strip().lower() not in {"default", "mlp"}:
        mlp_class = resolve_component("mlp", mlp_key)

    norm_class = None
    if str(norm_key).strip().lower() not in {"default", "layernorm"}:
        norm_class = resolve_component("norm", norm_key)

    train_config = merged["train"]
    if not isinstance(train_config, dict):
        raise ValueError(f"train 配置必须是字典: {config_path}")
    required_train_keys = (
        "max_lr",
        "min_lr",
        "warmup_steps",
        "max_steps",
        "weight_decay",
        "total_batch_size",
        "micro_batch_size",
        "sequence_length",
    )
    missing_train_keys = [k for k in required_train_keys if k not in train_config]
    if missing_train_keys:
        missing = ", ".join(missing_train_keys)
        raise ValueError(f"train 配置缺少字段: {missing} ({config_path})")

    return ExperimentSpec(
        experiment_name=str(merged["experiment_name"]),
        model_config=_build_model_config(merged["model"]),
        attention_class=attention_class,
        position_encoding_class=position_encoding_class,
        train_config=copy.deepcopy(train_config),
        mlp_class=mlp_class,
        norm_class=norm_class,
    )
