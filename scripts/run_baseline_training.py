"""Train the five-class SHARP baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars

import torch
from torch import nn

from src.models.sharp_classifier import SHARPClassifier
from src.training.train_utils import (
    evaluate_with_fusion_metrics,
    plot_train_val_history,
    update_checkpoints,
    get_data_loaders,
    load_checkpoint,
    run_epoch,
)
from src.utils.utils import get_logger, load_config

disable_progress_bars()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logger = get_logger(__name__)
api = HfApi()

def main(config_path: str, local: bool = False) -> None:
    """
    Trains the five-class SHARP baseline model

    Args:
        config_path: Path to the configuration file
        local: Flag to use local checkpoint files instead of Hugging Face

    Returns:
        None
    """
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_root = Path(config["paths"]["output_dir"])
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = "sharp_baseline_best.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name

    model = SHARPClassifier(
        n_classes=config["model"]["n_classes_primary"],
        nw=config["doppler"]["stacked_vectors_nw"],
        nd=config["doppler"]["velocity_bins_nd"],
        dropout_rate=config["model"].get("dropout_rate", 0.2),
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )
    train_loader, val_loader = get_data_loaders(logger, config, "S1")

    start_epoch, best_val_accuracy, history = load_checkpoint(
        model, optimizer, config, checkpoint_path, local=local
    )
    epochs = int(config["training"]["epochs"])
    tolerance_epochs = int(config["training"]["early_stopping_patience"])
    epochs_without_improvement = 0

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_fn, str(device), optimizer
        )
        val_loss, val_antenna_accuracy, val_accuracy = evaluate_with_fusion_metrics(
            model, val_loader, loss_fn, str(device)
        )
        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | "
            "val_loss=%.4f val_antenna_acc=%.4f val_fused_acc=%.4f",
            epoch,
            epochs,
            train_loss,
            train_accuracy,
            val_loss,
            val_antenna_accuracy,
            val_accuracy,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_accuracy)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            update_checkpoints(
                logger, api, model, optimizer, config, epoch, val_accuracy, history,
                checkpoint_path, checkpoint_name, local=local
            )
            logger.info("Saved best checkpoint to %s", checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= tolerance_epochs:
                logger.info(
                    "Early stopping at epoch %d after %d epochs without "
                    "fused validation improvement.",
                    epoch,
                    tolerance_epochs,
                )
                break
    
    figure_name="train_validation_over_epoch_baseline.png"
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_train_val_history(history, figures_dir, figure_name)
    logger.info("Training complete. Best val_acc=%.4f", best_val_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SHARP baseline.")
    parser.add_argument("--config", default="config/base_config.yaml")
    parser.add_argument("--local", action="store_true", help="Use local checkpoint files instead of Hugging Face.")
    args = parser.parse_args()
    main(args.config, local=args.local)