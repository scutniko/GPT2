"""
实验配置加载器
支持两种配置来源：
1) YAML配置（推荐）
2) 兼容旧版 experiments.<name> Python 配置
"""

from dataclasses import dataclass, fields
import copy
import importlib
import os

from core.config import GPTConfig
from core.registry import resolve_attention, resolve_position, resolve_mlp, resolve_norm

try:
    import yaml
except ImportError:  # pragma: no cover - 依赖缺失时抛出清晰错误
    yaml = None


@dataclass
class ExperimentSpec:
    experiment_name: str
    model_config: GPTConfig
    attention_class: type
    position_encoding_class: type
    train_config: dict
    mlp_class: type | None
    norm_class: type | None


def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml_with_base(path, stack=None):
    if yaml is None:
        raise RuntimeError("未安装 PyYAML，请先执行: python -m pip install pyyaml")
    abs_path = os.path.abspath(path)
    stack = stack or []
    if abs_path in stack:
        chain = " -> ".join(stack + [abs_path])
        raise ValueError(f"检测到循环 base 引用: {chain}")
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"配置文件不存在: {abs_path}")

    with open(abs_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    base_ref = data.pop("base", None)
    if not base_ref:
        return data

    base_path = base_ref
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.path.dirname(abs_path), base_ref)
    base_data = _load_yaml_with_base(base_path, stack=stack + [abs_path])
    return _deep_merge(base_data, data)


def _build_model_config(model_dict):
    model_dict = model_dict or {}
    gpt_fields = {f.name for f in fields(GPTConfig)}
    base_kwargs = {k: v for k, v in model_dict.items() if k in gpt_fields}
    extra_kwargs = {k: v for k, v in model_dict.items() if k not in gpt_fields}

    config = GPTConfig(**base_kwargs)
    for key, value in extra_kwargs.items():
        setattr(config, key, value)
    return config


def _load_from_yaml(config_path):
    raw = _load_yaml_with_base(config_path)
    components = raw.get("components", {}) or {}

    attention_name = components.get("attention", "base")
    position_name = components.get("position_encoding", "learned")
    mlp_name = components.get("mlp")
    norm_name = components.get("norm")

    experiment_name = raw.get("experiment_name")
    if not experiment_name:
        experiment_name = os.path.splitext(os.path.basename(config_path))[0]

    return ExperimentSpec(
        experiment_name=experiment_name,
        model_config=_build_model_config(raw.get("model", {})),
        attention_class=resolve_attention(attention_name),
        position_encoding_class=resolve_position(position_name),
        train_config=copy.deepcopy(raw.get("training", {})),
        mlp_class=resolve_mlp(mlp_name) if mlp_name else None,
        norm_class=resolve_norm(norm_name) if norm_name else None,
    )


def _load_from_python_module(experiment):
    exp_module = importlib.import_module(f"experiments.{experiment}")
    return ExperimentSpec(
        experiment_name=exp_module.EXPERIMENT_NAME,
        model_config=exp_module.MODEL_CONFIG,
        attention_class=exp_module.ATTENTION_CLASS,
        position_encoding_class=exp_module.POSITION_ENCODING_CLASS,
        train_config=copy.deepcopy(exp_module.TRAINING_CONFIG),
        mlp_class=getattr(exp_module, "MLP_CLASS", None),
        norm_class=getattr(exp_module, "NORM_CLASS", None),
    )


def load_experiment_spec(experiment=None, config_path=None):
    """加载实验配置（config_path 优先）。"""
    if config_path:
        return _load_from_yaml(config_path)
    if experiment:
        return _load_from_python_module(experiment)
    raise ValueError("必须提供 --experiment 或 --config")
