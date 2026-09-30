"""
Simplified Inception module used as SHARP's feature extractor.
"""

from __future__ import annotations

import torch
from torch import nn

# Per-branch feature-map counts (the "N" in the paper's "N@(ixi)" notation).
BRANCH_CHANNELS_A = 9 
BRANCH_CHANNELS_B = 5
BRANCH_CHANNELS_C = 1

class SimplifiedInceptionModule(nn.Module):
    """
    Three-branch Inception-style feature extractor for Doppler traces.

    Args:
        in_channels: number of input channels
        branch_a_channels: output channels for branch A
        branch_b_channels: output channels for branch B
        branch_c_channels: output channels for branch C
    """

    def __init__(
        self,
        in_channels: int = 1,
        branch_a_channels: int = BRANCH_CHANNELS_A,
        branch_b_channels: int = BRANCH_CHANNELS_B,
        branch_c_channels: int = BRANCH_CHANNELS_C,
    ) -> None:
        super().__init__()
        self.branch_a_channels = branch_a_channels
        self.branch_b_channels = branch_b_channels
        self.branch_c_channels = branch_c_channels

        self.branch_a = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 6, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 9, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
        )
        self.branch_b = nn.Sequential(
            nn.Conv2d(in_channels, 5, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.branch_c = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    @property
    def out_channels(self) -> int:
        """
        Calculates total output channels across all branches.

        Returns:
            sum of output channels from all three parallel branches
        """
        return self.branch_a_channels + self.branch_b_channels + self.branch_c_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs inputs through parallel branches and concatenates outputs.

        Args:
            x: input tensor of shape (batch, in_channels, Nw, ND)

        Returns:
            concatenated multi-scale feature tensor
        """
        a = self.branch_a(x)
        b = self.branch_b(x)
        c = self.branch_c(x)
        return torch.cat([c, b, a], dim=1)