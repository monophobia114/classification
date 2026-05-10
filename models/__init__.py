"""模型包：包含 STL-10 分类任务相关的网络定义。"""

from .baseline_cnn import BaselineCNN, count_parameters

__all__ = ["BaselineCNN", "count_parameters"]
