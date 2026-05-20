"""激活函数对比实验模型。

提供两个变体：
    - SigmoidCNN：所有 ReLU → Sigmoid
    - TanhCNN：    所有 ReLU → Tanh

权重初始化从 Kaiming 改为 Xavier（Glorot），因为 Kaiming 专为 ReLU 设计，
对 Sigmoid/Tanh 不适用。

"""

from __future__ import annotations

import torch
import torch.nn as nn


class _BaseActivationCNN(nn.Module):
    """共享骨架：3 Conv + 4 MaxPool + 2 FC，激活函数由子类指定。"""

    activation: type[nn.Module]  # 子类覆盖

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        act = self.activation

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            act(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            act(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            act(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 额外下采样
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            act(),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Xavier 初始化（适用于 Sigmoid / Tanh）。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class SigmoidCNN(_BaseActivationCNN):
    """所有激活函数使用 Sigmoid。"""
    activation = nn.Sigmoid


class TanhCNN(_BaseActivationCNN):
    """所有激活函数使用 Tanh。"""
    activation = nn.Tanh
