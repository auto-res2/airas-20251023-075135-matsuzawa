"""src/preprocess.py
Data loading & augmentation for CIFAR-10 (cached to .cache/).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from omegaconf import DictConfig, ListConfig

CACHE_DIR = ".cache/"

# -----------------------------------------------------------------------------
# Transform builder
# -----------------------------------------------------------------------------

def _build_transforms(transform_cfgs: List[dict]):
    tfms: List[T.transforms.Compose] = []
    for cfg in transform_cfgs:
        if isinstance(cfg, (DictConfig, dict)):
            name, params = list(cfg.items())[0]
            params = params or {}
            cls = getattr(T, name)
            tfms.append(cls(**params))
        else:
            raise ValueError("Each transform entry must be a dict {Name: params}")
    return T.Compose(tfms)


# -----------------------------------------------------------------------------
# Public loader factory
# -----------------------------------------------------------------------------

def get_dataloaders(cfg, batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if batch_size is not None:
        bs = batch_size
    else:
        bs = int(cfg.run.training.batch_size)

    transform_train = _build_transforms(cfg.run.dataset.transforms)
    # Validation/test – keep only basic preprocessing (ToTensor + Normalize)
    basic = [d for d in cfg.run.dataset.transforms if list(d.keys())[0] in ("ToTensor", "Normalize")]
    transform_val = _build_transforms(basic)

    full_train = CIFAR10(root=CACHE_DIR, train=True, download=True, transform=transform_train)
    test_ds = CIFAR10(root=CACHE_DIR, train=False, download=True, transform=transform_val)

    val_size = int(cfg.run.dataset.val_split)
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size])

    loader_kwargs = dict(batch_size=bs, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
