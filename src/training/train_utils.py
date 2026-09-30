"""
Useful functions used during the training of all the different models.
"""

from __future__ import annotations

import logging
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from src.models.decision_fusion import fuse_batch
from torch.nn.functional import softmax as softmax

from src.data.doppler_trace_dataset import build_train_val_split
from src.data.label_mapping import TARGET_CLASSES

from pathlib import Path
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
import matplotlib.pyplot as plt

def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None
) -> tuple[float, float]:
    """
    Function that runs an entire training epoch

    Args:
        model: model to be trained, instance of torch.nn.Module
        dataloader: DataLoader instance
        loss_fn: loss function used, instance of torch.nn.Module
        device: "cuda" or "cpu"
        optimizer: optimizer used, instance of torch.optim.Optimizer

    Returns:
        (average training loss, average accuracy over training set)
    """
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    context = torch.enable_grad()

    pbar = tqdm(dataloader, desc="Training", leave=False)
    t0 = time.perf_counter()
    
    with context:
        for batch_x, batch_y in pbar:
            data_time = time.perf_counter() - t0
            
            flattened_x, flattened_y = flatten_antennas(batch_x, batch_y["label"])
            flattened_x, flattened_y = flattened_x.to(device, non_blocking=True), flattened_y.to(device, non_blocking=True)
            
            t2 = time.perf_counter()
            logits = model(flattened_x)
            loss = loss_fn(logits, flattened_y)
            forward_time = time.perf_counter() - t2

            t3 = time.perf_counter()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            backward_time = time.perf_counter() - t3

            total_loss += loss.item() * flattened_y.size(0)
            total_correct += (logits.argmax(dim=1) == flattened_y).sum().item()
            total_count += flattened_y.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "data(s)": f"{data_time:.2f}",
                "fwd(s)": f"{forward_time:.2f}",
                "bwd(s)": f"{backward_time:.2f}"
            })
            t0 = time.perf_counter()

    return total_loss / total_count, total_correct / total_count

def flatten_antennas(batch_x: torch.Tensor, batch_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reshapes a (batch, Nant, Nw, ND) batch into (batch*Nant, 1, Nw, ND).

    The single shared SHARPClassifier is trained on every antenna's window
    as an independent training example.

    Args:
        batch_x: Tensor of shape (batch, Nant, Nw, ND).
        batch_y: Tensor of shape (batch, 1, Nw, ND).

    Returns:
        (flattened_x, flattened_y)
    """
    batch, n_ant, nw, nd = batch_x.shape
    flattened_x = batch_x.reshape(batch * n_ant, 1, nw, nd)
    flattened_y = batch_y.unsqueeze(1).expand(batch, n_ant).reshape(batch * n_ant)
    return flattened_x, flattened_y


def evaluate_with_fusion(model: torch.nn.Module, val_loader: DataLoader, loss_fn: torch.nn.Module, device: str) -> tuple[float, float]:
    """
    Evaluates the model using the SHARP Decision strategy.

    Args:
        model: model to be trained, instance of torch.nn.Module
        val_loader: DataLoader instance
        loss_fn: loss function used, instance of torch.nn.Module
        device: "cuda" or "cpu"

    Returns:
        (average loss, average accuracy) computed over the validation set
    """
    model.eval()
    total_loss = 0.0
    correct_fused = 0
    total_samples = 0

    with torch.no_grad():
        tqdm(val_loader, desc="Validation", leave=False)
        for inputs, targets in val_loader:
            batch_size, Nant, Nw, ND = inputs.shape
            inputs_flat = inputs.view(batch_size * Nant, 1, Nw, ND).to(device)
            targets_flat = targets["label"].repeat_interleave(Nant).to(device)

            logits_flat = model(inputs_flat)
            loss = loss_fn(logits_flat, targets_flat)
            total_loss += loss.item()

            probs_flat = softmax(logits_flat, dim=1)
            probs_reshaped = probs_flat.view(batch_size, Nant, -1)
            fused_preds = fuse_batch(probs_reshaped.cpu(), n_antennas=Nant)

            targets_cpu = targets["label"].cpu()
            correct_fused += (fused_preds == targets_cpu).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / len(val_loader)
    fused_acc = correct_fused / total_samples
    
    return avg_loss, fused_acc

def load_checkpoint(model: torch.nn.Module, optimizer: torch.nn.Module, config: dict, checkpoint_path: Path, local: bool=False) -> tuple[float, float]:
    """
    Loads a checkpoint on the Hugging Face related repository or locally.

    Args:
        model: model used, instance of torch.nn.Module,
        optimizer: optimizer used, instance of torch.optim.Optimizer
        config: configurations
        checkpoint_path: path directed to the locally stored checkpoint,
        local: flag used in order to search locally or online

    Returns:
        (epoch, validation accuracy, history) stored in the checkpoint
    """
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }
    
    if local and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch=checkpoint["epoch"]+1
        val_acc=checkpoint["val_acc"]
        history=checkpoint["history"]
    elif local:
        epoch=1
        val_acc = 0.0
    else:
        try:
            checkpoint = torch.load(hf_hub_download(
                repo_id="danieledaccordo/HAReW",
                filename="checkpoints_dir/"+checkpoint_path.name,
                repo_type="model",
            ), map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch=checkpoint["epoch"]+1
            val_acc=checkpoint["val_acc"]
            history=checkpoint["history"]
            torch.save(checkpoint, checkpoint_path)
        except EntryNotFoundError:
            epoch=1
            val_acc = 0.0

    return epoch, val_acc, history

def get_data_loaders(logger: logging.Logger, config: dict, set_id: str, transform=None) -> tuple[DataLoader, DataLoader]:
    """
    Creates the dataloaders given the set id used in the dataset

    Args:
        logger: logger used to print the output messages,
        config: configurations,
        set_id: string containing the id of the set taken into consideration,
        transform: transformation that could be used to do data augmentation,

    Returns:
        (train set data loader, validation set dataloader) instances
    """
    hw_config = config.get("hardware", {})
    num_workers = int(hw_config.get("num_workers", 2))
    pin_memory = bool(hw_config.get("pin_memory", True))
    persistent_workers = bool(hw_config.get("persistent_workers", True) and num_workers > 0)

    logger.info(
        "Preparing data loaders: workers=%d, pin_memory=%s, persistent_workers=%s",
        num_workers, pin_memory, persistent_workers
    )

    _, train_subset, val_subset, _ = build_train_val_split(
        Path(config["paths"]["doppler_traces_dir"]),
        set_id,
        window_size=config["doppler"]["stacked_vectors_nw"],
        stride=config["doppler"]["window_stride"],
        n_antennas=config["hardware"]["n_antennas"],
        train_split=config["training"]["train_split"],
        val_split=config["training"]["val_split"],
        logger=logger,
        transform=transform
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader


@torch.no_grad()
def evaluate_with_fusion_metrics(model: torch.nn.Module, val_loader: DataLoader, loss_fn, device: str) -> tuple[float, float, float]:
    """
    Evaluates per-antenna and fused metrics on grouped validation windows.
    Args:
        model: model used, instance of torch.nn.Module
        val_loader: dataloader for the validation set
        loss_fn: loss function used for training
        device: "cuda" or "cpu"

    Returns:
        (Average loss, Average per-antenna accuracy, Average fused accuracy) computed with the fusion metrics
    """
    model.eval()
    total_loss = 0.0
    total_antenna_correct = 0
    total_antenna_samples = 0
    total_fused_correct = 0
    total_samples = 0

    for inputs, targets in val_loader:
        batch_size, n_antennas, nw, nd = inputs.shape
        labels = targets["label"].to(device, non_blocking=True)
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs.reshape(batch_size * n_antennas, 1, nw, nd))
        labels_flat = labels.repeat_interleave(n_antennas)
        total_loss += loss_fn(logits, labels_flat).item() * labels_flat.numel()

        predictions = logits.argmax(dim=1)
        total_antenna_correct += (predictions == labels_flat).sum().item()
        total_antenna_samples += labels_flat.numel()

        probabilities = softmax(logits, dim=1).reshape(batch_size, n_antennas, -1)
        fused_predictions = fuse_batch(probabilities.cpu(), n_antennas=n_antennas)
        total_fused_correct += (fused_predictions == labels.cpu()).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Validation loader produced no samples")

    return (
        total_loss / total_antenna_samples,
        total_antenna_correct / total_antenna_samples,
        total_fused_correct / total_samples,
    )

def update_checkpoints(
    logger: logging.Logger, 
    api, 
    model: torch.nn.Module, 
    optimizer: torch.nn.Module, 
    config: dict, 
    epoch: int, 
    val_acc: float, 
    history: dict,
    checkpoint_path: Path, 
    checkpoint_name: str, 
    local: bool=False
) -> None:
    """
    Update the checkpoint, it could be online or local.
    Args:
        logger: logger used to print the output messages,
        api: api handler used for hugging face, 
        model: model used, instance of torch.nn.Module, 
        optimizer: optimizer used, instance of torch.optim.Optimizer
        config: configurations,
        epoch: current epoch, 
        val_acc: validation accuracy performed in the current epoch, 
        history: dictionary containing the history of the training of the model,
        checkpoint_path: path directed to the checkpoint file position, 
        checkpoint_name: name of the checkpoint, 
        local: flag used in order to search locally or online
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "class_names": list(TARGET_CLASSES),
            "val_acc" : val_acc,
            "history": history
        },
        checkpoint_path,
    )
    logger.info("Saved new best checkpoint (val_acc=%.4f) -> %s", val_acc, checkpoint_path)
    if not local:
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo="checkpoints_dir/" + checkpoint_name,
            repo_id="danieledaccordo/HAReW",
            repo_type="model",
        )
        logger.info("Uploaded new best checkpoint to Hugging Face")

def plot_train_val_history(history: dict, figures_dir: Path, figure_name: str) -> None:
    """
    Function used to print and save the plot related to the validation and training loss over the different epochs.
    Args:
        history: dictionary containing the history of the training of the model,
        figures_dir: path directed to the output folder position, 
        figure_name: name of the plot
    """
    
    plt.figure(figsize=(8,5))
    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Train loss",
        color="blue",
        marker="o"
    )
    plt.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation loss",
        color="red",
        marker="o"
    )
    plt.title("Training and validation loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(figures_dir / figure_name, bbox_inches='tight', dpi=300)
    plt.show()
