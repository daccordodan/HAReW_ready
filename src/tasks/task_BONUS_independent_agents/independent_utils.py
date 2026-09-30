"""
Utility functions for running independent per-antenna model training and evaluation.
"""

from tqdm import tqdm

import time

import torch
from torch.nn.functional import softmax as softmax
from torch.utils.data import DataLoader
from pathlib import Path
import logging

from src.data.doppler_trace_dataset import DopplerTraceDataset
from src.models.decision_fusion import fuse_batch


def get_independent_data_loaders(
    logger: logging.Logger, 
    config: dict, 
    set_id: str
) -> tuple[DataLoader, DataLoader]:
    """
    Builds data loaders retaining multi-antenna structure without flattening.

    Args:
        logger: logger instance for output messages
        config: configuration parameters dictionary
        set_id: identifier string for dataset subset

    Returns:
        train and validation dataloaders
    """
    root_dir = Path(config["paths"]["doppler_traces_dir"])
    common = dict(
        root_dir=root_dir,
        sets_to_include=(set_id,),
        window_size=config["doppler"]["stacked_vectors_nw"],
        stride=config["doppler"]["window_stride"],
        n_antennas=config["hardware"]["n_antennas"],
        train_split=config["training"]["train_split"],
        val_split=config["training"]["val_split"],
        logger=logger,
    )
    train_dataset = DopplerTraceDataset(
        **common,
        temporal_split="train",
        expand_train_antennas=False,
    )
    val_dataset = DopplerTraceDataset(**common, temporal_split="val")
    loader_kwargs = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": config["hardware"].get("num_workers", 2),
        "pin_memory": config["hardware"].get("pin_memory", True),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = config["hardware"].get(
            "persistent_workers", True
        )
    return DataLoader(train_dataset, shuffle=True, **loader_kwargs), DataLoader(
        val_dataset, shuffle=False, **loader_kwargs
    )


def run_independent_epochs(
    models: list[torch.nn.Module],
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: str,
    optimizers: list[torch.optim.Optimizer]
) -> tuple[list[float], list[float]]:
    """
    Trains multiple antenna-specific models independently for one epoch.

    Args:
        models: list of model instances, one per antenna
        dataloader: dataloader yielding multi-antenna inputs
        loss_fn: loss function instance
        device: "cuda" or "cpu"
        optimizers: list of optimizers, one per model

    Returns:
        average losses and accuracies for each model
    """
    total_losses, total_corrects = [0.0] * len(models), [0] * len(models)
    total_counts = 0
    context = torch.enable_grad()

    for model in models:
        model.train()

    pbar = tqdm(dataloader, desc="Training", leave=False)
    t0 = time.perf_counter()
    
    with context:
        for batch_x, batch_y in pbar:
            data_time = time.perf_counter() - t0
            
            # Transfer to GPU
            batch_x, batch_y["label"] = batch_x.to(device, non_blocking=True), batch_y["label"].to(device, non_blocking=True)
            if batch_x.ndim != 4 or batch_x.shape[1] != len(models):
                raise ValueError(
                    f"Expected batch_x shape (batch, {len(models)}, Nw, ND), "
                    f"got {tuple(batch_x.shape)}"
                )

            # Forward Pass
            for i in range(len(models)):
                t2 = time.perf_counter()
                logits = models[i](batch_x[:, i, :, :].unsqueeze(1))
                loss = loss_fn(logits, batch_y["label"])
                forward_time = time.perf_counter() - t2

                # Backward Pass
                t3 = time.perf_counter()
                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()
                backward_time = time.perf_counter() - t3

                total_losses[i] += loss.item() * batch_y["label"].size(0)
                total_corrects[i] += (logits.argmax(dim=1) == batch_y["label"]).sum().item()

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "data(s)": f"{data_time:.2f}",
                    "fwd(s)": f"{forward_time:.2f}",
                    "bwd(s)": f"{backward_time:.2f}"
                })
                t0 = time.perf_counter()

            total_counts += batch_y["label"].size(0)
    total_losses[:] = [x / total_counts for x in total_losses]
    total_corrects[:] = [x / total_counts for x in total_corrects]
    return total_losses, total_corrects

def evaluate_with_fusion_independent(
    models: list[torch.nn.Module], 
    val_loader: DataLoader, 
    loss_fn: torch.nn.Module, 
    device: str
) -> tuple[list[float], list[float], float]:
    """
    Evaluates independent antenna models individually and combined via decision fusion.

    Args:
        models: list of trained antenna-specific models
        val_loader: validation dataloader
        loss_fn: loss function instance
        device: "cuda" or "cpu"

    Returns:
        average losses, per-model accuracies, and fused accuracy
    """
    for model in models:
        model.eval()
    
    n_models = len(models)

    total_loss = [0.0] * n_models
    indep_acc = [0] * n_models
    correct_fused = 0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for inputs, targets in pbar:
            batch_size, _, _, _ = inputs.shape

            inputs = inputs.to(device, non_blocking=True)
            labels = targets["label"].to(device, non_blocking=True)
            per_antenna_probabilities = []

            for i, model in enumerate(models):
                antenna_input = inputs[:, i, :, :].unsqueeze(1)
                logits = model(antenna_input)
                loss = loss_fn(logits, labels)
                total_loss[i] += loss.item() * batch_size
                per_antenna_probabilities.append(softmax(logits, dim=1))
                preds = logits.argmax(dim=1)
                indep_acc[i] += (preds == labels).sum().item()

            probabilities = torch.stack(per_antenna_probabilities, dim=1)
            fused_preds = fuse_batch(probabilities.cpu(), n_antennas=n_models)

            targets_cpu = targets["label"].cpu()
            correct_fused += (fused_preds == targets_cpu).sum().item()
            total_samples += batch_size

    avg_loss = [x / total_samples for x in total_loss]
    fused_acc = correct_fused / total_samples
    indep_acc = [x / total_samples for x in indep_acc]
    return avg_loss, indep_acc, fused_acc