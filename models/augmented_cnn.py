"""数据增强实验模型。

本文件是 BaselineCNN 的一个别名——模型结构完全不变，仅配合数据增强流程使用。
训练时通过 `python -m scripts.train --model models.augmented_cnn.AugmentedCNN --use-augmentation`
启用 RandomCrop + HorizontalFlip + ColorJitter 数据增强。

"""

from .baseline_cnn import BaselineCNN


class AugmentedCNN(BaselineCNN):
    """带数据增强训练的 BaselineCNN（架构完全相同，仅训练流程不同）。

    实际的数据增强在 data.stl10_dataset.build_augmented_train_valid_loaders() 中实现：
        RandomCrop(96, padding=4, padding_mode='reflect')
        RandomHorizontalFlip(p=0.5)
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    """
    pass
