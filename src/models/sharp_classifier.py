"""
Full per-antenna SHARP classifier head.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.inception_module import SimplifiedInceptionModule


class SHARPClassifier(nn.Module):
    """
    Single-antenna SHARP activity classifier.

    Args:
        n_classes: number of output activity classes
        nw: input Doppler trace time dimension
        nd: input Doppler trace bin dimension
        reduced_channels: output channels of 1x1 reduction convolution
        dropout_rate: dropout probability before dense layer
    """

    def __init__(
        self,
        n_classes: int = 5,
        nw: int = 340,
        nd: int = 100,
        reduced_channels: int = 3,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_extractor = SimplifiedInceptionModule(in_channels=1)
        self.reduction_conv = nn.Conv2d(self.feature_extractor.out_channels, reduced_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        pooled_nw, pooled_nd = nw // 2, nd // 2
        self.classifier_head = nn.Linear(reduced_channels * pooled_nw * pooled_nd, n_classes)

    def forward(self, doppler_trace: torch.Tensor) -> torch.Tensor:
        """
        Produces activity classification logits from a single antenna Doppler trace.

        Args:
            doppler_trace: input Doppler trace tensor

        Returns:
            raw activity classification logits
        """
        features = self.feature_extractor(doppler_trace)
        reduced = self.relu(self.reduction_conv(features))
        flattened = torch.flatten(reduced, start_dim=1)
        dropped = self.dropout(flattened)
        return self.classifier_head(dropped)

    def count_parameters(self) -> int:
        """
        Calculates total trainable parameter count.

        Returns:
            number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)