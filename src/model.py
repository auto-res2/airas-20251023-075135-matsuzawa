"""src/model.py
Convolutional network architectures used in the experiments.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_c: int, out_c: int, k: int, s: int, p: int) -> nn.Sequential:
    """Helper to create Conv-BN-ReLU block."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    """Configurable small CNN (~1.2 M parameters)."""

    def __init__(self, cfg, dropout: float):
        super().__init__()
        layers_cfg = cfg.run.model.conv_layers
        layers = []
        in_ch = 3
        for idx, conv_cfg in enumerate(layers_cfg):
            layers.append(
                _conv_block(
                    in_c=in_ch,
                    out_c=int(conv_cfg.out_channels),
                    k=int(conv_cfg.kernel_size),
                    s=int(conv_cfg.stride),
                    p=int(conv_cfg.padding),
                )
            )
            in_ch = int(conv_cfg.out_channels)
            if (idx + 1) % 2 == 0:
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)

        # Compute flattened feature dimension ----------------------------
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            feat_dim = self.features(dummy).view(1, -1).size(1)

        fc_layers = []
        last = feat_dim
        for fc_cfg in cfg.run.model.fc_layers:
            fc_layers.extend(
                [
                    nn.Linear(last, int(fc_cfg.out_features)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            last = int(fc_cfg.out_features)
        fc_layers.append(nn.Linear(last, 10))  # CIFAR-10 has 10 classes
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):  # type: ignore[override]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_model(cfg, dropout: float):
    name = cfg.run.model.name.lower()
    if name.startswith("small-cnn"):
        return SmallCNN(cfg, dropout)
    raise ValueError(f"Unknown model {cfg.run.model.name}")
