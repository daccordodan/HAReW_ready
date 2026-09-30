"""CLI: runs Task 1 (cross-subject generalization) contrastive pretraining.

Usage:
    python scripts/run_task1.py --config config/task1_cross_subject.yaml
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from pathlib import Path

from src.tasks.task1_cross_subject.contrastive_encoder import ContrastiveEncoder, FineTunedModel, evaluate_encoder, train_contrastive_pretraining, freeze
from src.models.transform import dopplerTraceTransformation
from src.models.losses import SupConLoss
from src.training.train_utils import run_epoch, evaluate_with_fusion, load_checkpoint, get_data_loaders, update_checkpoints, plot_train_val_history
from src.utils.utils import load_config, get_logger

from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()
logger = get_logger(__name__)
api = HfApi()

def main(
    config_path: str,
    local: bool = False,
    phase: str = "all",
    pretrained_checkpoint: str = "pretraining_t13_best.pt",
) -> None:
    """
    Runs contrastive pretraining and fine-tuning for Task 1

    Args:
        config_path: path to configuration file
        local: flag to use local checkpoints instead of Hugging Face
        phase: execution phase (all, pretrain, or finetune)
        pretrained_checkpoint: filename of the pretrained checkpoint

    Returns:
        None
    """
    if phase not in {"all", "pretrain", "finetune"}:
        raise ValueError(f"Unknown Task 1 phase: {phase}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(config_path)

    n_classes=config["model"]["n_classes_primary"]

    output_root=Path(config["paths"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_name = pretrained_checkpoint
    checkpoints_dir = output_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / checkpoint_name

    contrastive_enc = ContrastiveEncoder(
        nw=config["doppler"]["stacked_vectors_nw"],
        nd=config["doppler"]["velocity_bins_nd"],
    ).to(device)
    loss_fn = SupConLoss(temperature=config["contrastive"]["temperature"])
    optimizer = torch.optim.Adam(contrastive_enc.parameters(), lr=config["training"]["learning_rate"])

    transform = dopplerTraceTransformation

    logger.info("Model parameter count: %d (paper reference: 128,535)", contrastive_enc.count_parameters())

    logger.info("Starting dataset parsing and recording loading...")
    train_loader, val_loader = get_data_loaders(logger, config, "S1", transform)
    logger.info("Dataset preparation complete.")

    if phase in {"all", "pretrain"}:
        tolerance_epochs = int(config["training"]["pretraining_early_stopping"])
        epochs_without_improvement = 0
        
        start_epoch, _, history = load_checkpoint(
            contrastive_enc, optimizer, config, checkpoint_path, local=local
        )
        best_val_loss = min(history["val_loss"], default=float("inf"))        

        for epoch in range(start_epoch, config["training"]["pretraining_epochs"] + 1):
            logger.info(
                "Starting pretraining epoch %d/%d...",
                epoch,
                config["training"]["pretraining_epochs"],
            )
            train_loss = train_contrastive_pretraining(
                contrastive_enc, train_loader, loss_fn, device, optimizer
            )
            val_loss = evaluate_encoder(contrastive_enc, val_loader, loss_fn, device)
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f",
                epoch,
                config["training"]["pretraining_epochs"],
                train_loss,
                val_loss,
            )

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                epochs_without_improvement = 0
                best_val_loss = val_loss
                update_checkpoints(
                    logger,
                    api,
                    contrastive_enc,
                    optimizer,
                    config,
                    epoch,
                    0.0,
                    history,
                    checkpoint_path,
                    checkpoint_name,
                    local=local,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= tolerance_epochs:
                    logger.info(
                        "Early stopping at epoch %d after %d epochs without "
                        "validation improvement.",
                        epoch,
                        tolerance_epochs,
                    )
                    break

        figure_name = "pretrain_validation_over_epoch_t13.png"
        figures_dir = output_root / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_train_val_history(history, figures_dir, figure_name)
        logger.info("Pre-training complete. Best val_loss=%.4f", best_val_loss)

        if phase == "pretrain":
            return
    else:
        pretrain_optimizer = torch.optim.Adam(
            contrastive_enc.parameters(),
            lr=config["training"]["learning_rate"],
        )
        load_checkpoint(
            contrastive_enc,
            pretrain_optimizer,
            config,
            checkpoint_path,
            local=local,
        )

    freeze(contrastive_enc)
    loss_fn = nn.CrossEntropyLoss()
    train_loader, val_loader = get_data_loaders(logger, config, "S1")

    classifier = nn.Linear(contrastive_enc.reduced_channels * contrastive_enc.pooled_nw * contrastive_enc.pooled_nd, n_classes)
    HAReW_model= FineTunedModel(
        contrastive_enc,
        classifier
    ).to(device)

    checkpoint_name = "training_t13_best.pt"
    checkpoint_path = checkpoints_dir / checkpoint_name
    finetune_optimizer = torch.optim.Adam(
        (parameter for parameter in HAReW_model.parameters() if parameter.requires_grad),
        lr=config["training"]["learning_rate"],
    )
    start_epoch, best_val_acc, history = load_checkpoint(
        HAReW_model, finetune_optimizer, config, checkpoint_path, local=local
    )

    for epoch in range(start_epoch, config["training"]["finetune_epochs"] + 1):
        logger.info(
            "Starting fine-tuning epoch %d/%d...",
            epoch,
            config["training"]["finetune_epochs"],
        )
        train_loss, train_acc = run_epoch(
            HAReW_model,
            train_loader,
            loss_fn,
            device,
            finetune_optimizer,
        )
        val_loss, val_acc = evaluate_with_fusion(HAReW_model, val_loader, loss_fn, device)
        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch, config["training"]["finetune_epochs"], train_loss, train_acc, val_loss, val_acc,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            update_checkpoints(
                logger, api, HAReW_model, finetune_optimizer, config, epoch, val_acc, history,
                checkpoint_path, checkpoint_name, local=local
            )

    figure_name="train_validation_over_epoch_task1.png"
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_train_val_history(history, figures_dir, figure_name)
    logger.info("Training complete. Best val_acc=%.4f", best_val_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 1: cross-subject contrastive pretraining.")
    parser.add_argument("--config", type=str, default="config/task1_cross_subject.yaml")
    parser.add_argument("--local", action="store_true", help="Use local checkpoint files instead of Hugging Face.")
    parser.add_argument(
        "--phase",
        choices=("all", "pretrain", "finetune"),
        default="all",
        help="Run both stages, only pretraining, or fine-tuning from a pretrained encoder.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default="pretraining_t13_best.pt",
        help="Pretraining checkpoint filename used by the fine-tuning-only phase.",
    )
    args = parser.parse_args()
    main(
        args.config,
        local=args.local,
        phase=args.phase,
        pretrained_checkpoint=args.pretrained_checkpoint,
    )