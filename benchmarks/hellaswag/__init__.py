"""
HellaSwag 基准任务模块。
"""

from benchmarks.hellaswag.data import DATA_CACHE_DIR, download, iterate_examples, render_example
from benchmarks.hellaswag.eval import evaluate_hf_model

__all__ = [
    "DATA_CACHE_DIR",
    "download",
    "iterate_examples",
    "render_example",
    "evaluate_hf_model",
]

