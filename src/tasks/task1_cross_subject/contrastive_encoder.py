"""Contrastive encoder for task 1
"""

from __future__ import annotations

from src.training.train_utils import flatten_antennas

import torch
from torch import nn
from torch.utils.data import DataLoader

import time
from tqdm import tqdm


from src.models.inception_module import SimplifiedInceptionModule

class FineTunedModel(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.encoder.extract_features=True
        
    def forward(self, doppler_trace):
        features = self.encoder(doppler_trace)
        return self.classifier(features)

class ContrastiveEncoder(nn.Module):
    """Encoder used for contrastive pretraining on Doppler traces.

    Reuses SimplifiedInceptionModule as the backbone so the pretrained
    weights can later be transplanted into a SHARPClassifier-compatible
    feature extractor, but keeps its own projection head, entirely separate
    from the baseline's classifier_head.
    """

    def __init__(
        self,
        nw: int = 340,
        nd: int = 100,
        reduced_channels: int = 3,
        dropout_rate: float = 0.2,
        projection_dim: int = 128,
        hidden_dim: int = 512,
        extract_features: bool = False
    ) -> None:
        super().__init__()
        self.extract_features = extract_features
        self.feature_extractor = SimplifiedInceptionModule(in_channels=1)
        self.reduction_conv = nn.Conv2d(self.feature_extractor.out_channels, reduced_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.reduced_channels = reduced_channels
        self.pooled_nw, self.pooled_nd = nw // 2, nd // 2
        
        # Convolutional projector to gently reduce spatial dimensions
        self.projector = nn.Sequential(
            nn.Conv2d(reduced_channels, 16, kernel_size=5, stride=5), # 170x50 -> 34x10
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=2, stride=2),               # 34x10 -> 17x5
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 17 * 5, hidden_dim),                       # 2,720 inputs (reduces dense params to ~1.4M)
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, doppler_trace: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(doppler_trace) 
        reduced = self.relu(self.reduction_conv(features)) 
        
        # Preserve the massive flattened path specifically for the downstream fine-tuning classifier
        flattened = torch.flatten(reduced, start_dim=1)
        dropped = self.dropout(flattened)

        if self.extract_features:
            return dropped

        # Pass the 2D spatial grid to the convolutional projector during contrastive pretraining
        return self.projector(reduced)
    
    def count_parameters(self) -> int:
            """Returns the total trainable parameter count, for comparison."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def train_contrastive_pretraining(    
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Runs one epoch of training or evaluation.

    Args:
        model: SHARPClassifier
        dataloader: Yields (batch_x, batch_y) with batch_x shape (batch, Nant, Nw, ND)
        loss_fn: Used loss function
        device: "cuda" or "cpu"
        optimizer: runs backward()+step()

    Returns:
        (mean_loss, accuracy) for the epoch.
    """
    model.train()

    total_loss, total_count = 0.0, 0
    context = torch.enable_grad()
    pbar = tqdm(dataloader, desc="Training", leave=False)
    t0 = time.perf_counter()

    with context:
        for batch_x, batch_y in pbar:
            data_time = time.perf_counter() - t0

            flattened_x1, flattened_y1 = flatten_antennas(batch_x[0], batch_y["label"])
            flattened_x2, _ = flatten_antennas(batch_x[1], batch_y["label"])
            flattened_x1, flattened_x2 = flattened_x1.to(device), flattened_x2.to(device)
            flattened_y1 = flattened_y1.to(device)

            t2 = time.perf_counter()
            logits1 = model(flattened_x1)
            logits2 = model(flattened_x2)
            loss = loss_fn(logits1, logits2, flattened_y1)
            forward_time = time.perf_counter() - t2

            t3 = time.perf_counter()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            backward_time = time.perf_counter() - t3

            total_loss += loss.item() * flattened_x1.size(0)
            total_count += flattened_x1.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "data(s)": f"{data_time:.2f}",
                "fwd(s)": f"{forward_time:.2f}",
                "bwd(s)": f"{backward_time:.2f}"
            })
            t0 = time.perf_counter()

    return total_loss / total_count

def evaluate_encoder(model, val_loader, loss_fn, device):
    total_loss=0

    model.eval()
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for inputs, targets in pbar:
            batch_size, Nant, Nw, ND = inputs[0].shape
            labels = targets["label"].to(device, non_blocking=True)
            labels_flat = labels.repeat_interleave(Nant)

            inputs_flat_1 = inputs[0].view(batch_size * Nant, 1, Nw, ND).to(device)
            inputs_flat_2 = inputs[1].view(batch_size * Nant, 1, Nw, ND).to(device)

            logits_flat_1=model(inputs_flat_1)
            logits_flat_2=model(inputs_flat_2)

            loss = loss_fn(logits_flat_1, logits_flat_2, labels_flat)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss