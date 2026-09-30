"""CLI: trains the SHARP baseline classifier on set S1.
"""

from __future__ import annotations

import argparse
import torch
from torch import nn

from src.models.sharp_classifier import SHARPClassifier
from src.tasks.task_BONUS_independent_agents.independent_utils import (
    evaluate_with_fusion_independent,
    get_independent_data_loaders,
    run_independent_epochs,
)
from src.training.train_utils import load_checkpoint, update_checkpoints, plot_train_val_history
from src.utils.utils import load_config, get_logger

from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()
logger = get_logger(__name__)
api = HfApi()

def main(config_path: str, local: bool = False) -> None:
    """
    Trains independent SHARP baseline models on set S1

    Args:
        config_path: Path to the configuration file
        local: Flag to use local checkpoint files instead of Hugging Face

    Returns:
        None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config(config_path)

    output_root=Path(config["paths"]["output_dir_indep"])
    output_root.mkdir(parents=True, exist_ok=True)

    models = [None] * 4
    optimizers = [None] * 4
    best_val_accuracies = [None] *4
    loss_fn = nn.CrossEntropyLoss()
    for i in range(config["hardware"]["n_antennas"]):
        models[i] =  SHARPClassifier(
            n_classes=config["model"]["n_classes_primary"],
            nw=config["doppler"]["stacked_vectors_nw"],
            nd=config["doppler"]["velocity_bins_nd"],
        ).to(device)

        optimizers[i] = torch.optim.Adam(models[i].parameters(), lr=config["training"]["learning_rate"])

        checkpoint_name=f"sharp_best_antenna_{i}.pt"
        checkpoints_dir = output_root / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoints_dir / checkpoint_name
        start_epoch, best_val_accuracies[i], history=load_checkpoint(
            models[i], optimizers[i], config, checkpoint_path, local=local
        )
    
    logger.info("Starting dataset parsing and recording loading...")
    train_loader, val_loader = get_independent_data_loaders(logger, config, "S1")
    logger.info("Dataset preparation complete.")
    tolerance_epochs = int(config["training"]["early_stopping_patience_indep"])

    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        logger.info("Starting training epoch %d/%d...", epoch, config["training"]["epochs"])
        train_losses, training_accuracies = run_independent_epochs(
            models, train_loader, loss_fn, device, optimizers
        )
        val_losses, val_accuracies, total_val_accuracy = evaluate_with_fusion_independent(models, val_loader, loss_fn, device)

        for i in range(4):
            logger.info(
                "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f tot_val_acc=%.4f",
                epoch, config["training"]["epochs"], train_losses[i], training_accuracies[i], val_losses[i], val_accuracies[i], total_val_accuracy
            )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_losses)
        history["val_loss"].append(val_losses)
        history["val_acc"].append(val_accuracies)

        updated = False
        for i in range(4):
            if val_accuracies[i] > best_val_accuracies[i]:
                epochs_without_improvement=0
                updated=True

                best_val_accuracies[i] = val_accuracies[i]
                checkpoint_name=f"sharp_best_antenna_{i}.pt"
                checkpoints_dir = output_root / "checkpoints"
                checkpoints_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoints_dir / checkpoint_name

                update_checkpoints(
                    logger, api, models[i], optimizers[i], config, epoch, val_accuracies[i], history,
                    checkpoint_path, checkpoint_name, local=local
                )

        if not updated:
            epochs_without_improvement += 1
            if epochs_without_improvement >= tolerance_epochs:
                logger.info(
                    "Early stopping at epoch %d after %d epochs without "
                    "fused validation improvement.",
                    epoch,
                    tolerance_epochs,
                )
                break

    figure_name="train_validation_over_epoch_independent_baseline.png"
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_train_val_history(history, figures_dir, figure_name)
    for i in range(4):
        logger.info(f"Training complete. Best val_acc_{i}=%.4f", best_val_accuracies[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SHARP baseline.")
    parser.add_argument("--config", type=str, default="config/base_config.yaml")
    parser.add_argument("--local", action="store_true", help="Use local checkpoint files instead of Hugging Face.")
    args = parser.parse_args()
    main(args.config, local=args.local)