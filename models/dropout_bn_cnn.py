"""Dropout + Batch Normalization 组合实验模型。

同时加入 BN（Conv 后）和 Dropout（FC1 后）。
注意：BN 和 Dropout 组合使用时存在"方差偏移"问题（训练/推理不一致），
因此 Dropout rate 从 0.5 调低至 0.3。

架构：Conv2d → BatchNorm2d → ReLU → MaxPool2d → ... → FC1 → ReLU → Dropout(0.3) → FC2

"""

from __future__ import annotations

import torch
import torch.nn as nn


class DropoutBatchNormCNN(nn.Module):
    """同时使用 BatchNorm 和 Dropout 的 CNN。

    3 Conv + BN + 4 MaxPool + 2 FC + Dropout(0.3)。
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 额外下采样
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),  # 较单独 Dropout 实验降低 rate，缓解 BN+Dropout 方差偏移
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming 初始化，与 ReLU 匹配。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
