"""CLI: evaluates the 4 independent SHARP models across sets S1-S7.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.sharp_classifier import SHARPClassifier
import torch
from torch.nn.functional import softmax as softmax
from torch.utils.data import DataLoader
import numpy as np

from src.data.label_mapping import TARGET_CLASSES
from src.data.doppler_trace_dataset import build_train_val_split, build_zero_shot_test_set
from src.evaluation.metrics import compute_accuracy_per_activity, compute_confusion_matrix, compute_f1_per_activity
from src.utils.utils import load_config, get_logger
from src.evaluation.evaluation_utils import load_checkpoint_to_model, plot_acc_f1_results_pa, plot_conf_mat_results_pa, write_report_performances
from src.models.decision_fusion import fuse_batch


S7_REFERENCE_ACCURACY = 0.9599 # For comparison with the paper

logger = get_logger(__name__)

def main(config_path: str, local: bool = False) -> None:
    """
    Evaluates trained checkpoints across all scenarios

    Args:
        config_path: path to configuration file
        local: flag to load checkpoints from local disk

    Returns:
        None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(config_path)

    models = [None] * 4
    optimizers = [None] * 4

    data_root=Path(config["paths"]["doppler_traces_dir"])
    output_root=Path(config["paths"]["output_dir_indep"])

    for i in range(config["hardware"]["n_antennas"]):
        models[i] =  SHARPClassifier(
            n_classes=config["model"]["n_classes_primary"],
            nw=config["doppler"]["stacked_vectors_nw"],
            nd=config["doppler"]["velocity_bins_nd"],
        ).to(device)
    
        optimizers[i] = torch.optim.Adam(models[i].parameters(), lr=config["training"]["learning_rate"])
    
        checkpoint_name=f"sharp_best_antenna_{i}.pt"
        models[i] = load_checkpoint_to_model(
            config, models[i], checkpoint_name, device, logger, local=local
        )

    accuracy_by_set: dict[str, float] = {}
    accuracy_by_set_pa: dict[str, dict[str, float]] = {}
    fscore_by_set_pa: dict[str, dict[str, float]] = {}
    conf_mat_by_set: dict[str, np.ndarray] = {}

    logger.info("=== Starting evaluation ===")
    logger.info("Evaluated set S1")
    _, _, _, s1_test_subset = build_train_val_split(
        data_root,
        set_id="S1",
        window_size=config["doppler"]["stacked_vectors_nw"],
        stride=config["doppler"].get("window_stride"),
        n_antennas=config["hardware"]["n_antennas"],
        train_split=config["training"]["train_split"],
        val_split=config["training"]["val_split"],
    )
    s1_loader = DataLoader(s1_test_subset, batch_size=config["training"]["batch_size"], shuffle=False)
    y_true, y_pred = evaluate_set(models, s1_loader, device)
    accuracy_by_set_pa["S1"],  accuracy_by_set["S1"] = compute_accuracy_per_activity(y_true, y_pred,TARGET_CLASSES)
    fscore_by_set_pa["S1"] = compute_f1_per_activity(y_true,y_pred,TARGET_CLASSES)
    conf_mat_by_set["S1"] = compute_confusion_matrix(y_true, y_pred, len(TARGET_CLASSES))

    set_ids=["S2", "S3", "S4", "S5", "S6", "S7"]
    for set_id in set_ids:
        logger.info(f"Evaluated set {set_id}")
        test_dataset = build_zero_shot_test_set(
            data_root,
            set_id=set_id,
            window_size=config["doppler"]["stacked_vectors_nw"],
            stride=config["doppler"].get("window_stride"),
            n_antennas=config["hardware"]["n_antennas"],
        )
        test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
        y_true, y_pred = evaluate_set(models, test_loader, device)
        accuracy_by_set_pa[set_id], accuracy_by_set[set_id] = compute_accuracy_per_activity(y_true, y_pred,TARGET_CLASSES)
        fscore_by_set_pa[set_id] = compute_f1_per_activity(y_true, y_pred, TARGET_CLASSES)
        conf_mat_by_set[set_id] = compute_confusion_matrix(y_true, y_pred, len(TARGET_CLASSES))

    logger.info("=== Per-set accuracy (fused, decision-level) ===")
    for set_id, acc in accuracy_by_set.items():
        logger.info("  %s: %.4f", set_id, acc)

    if "S7" in accuracy_by_set:
        logger.info(
            "S7 vs. paper reference: measured=%.4f, paper=%.4f, diff=%.4f",
            accuracy_by_set["S7"], S7_REFERENCE_ACCURACY, accuracy_by_set["S7"] - S7_REFERENCE_ACCURACY,
        )

    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    set_ids.insert(0,"S1")
    plot_acc_f1_results_pa(accuracy_by_set_pa,fscore_by_set_pa,figures_dir)
    plot_conf_mat_results_pa(conf_mat_by_set, set_ids, TARGET_CLASSES, figures_dir)

    files_dir = output_root / "text"
    files_dir.mkdir(parents=True, exist_ok=True)
    write_report_performances(accuracy_by_set, accuracy_by_set_pa, fscore_by_set_pa, files_dir)

@torch.no_grad()
def evaluate_set(models: list(), dataloader: DataLoader, device: str) -> tuple[list[int], list[int]]:
    """
    Runs fused-prediction inference over a dataset

    Args:
        models: list of trained PyTorch models
        dataloader: yields data batches
        device: "cuda" or "cpu"

    Returns:
        ground-truth and fused-predicted class indices
    """
    for model in models:
        model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        per_antenna_probabilities = []

        for i, model in enumerate(models):
            antenna_input = inputs[:, i, :, :].unsqueeze(1)
            logits = model(antenna_input)
            per_antenna_probabilities.append(softmax(logits, dim=1))

        probabilities = torch.stack(per_antenna_probabilities, dim=1)
        fused_preds = fuse_batch(probabilities.cpu(), n_antennas=len(models))

        y_true.extend(targets["label"].tolist())
        y_pred.extend(fused_preds.tolist())

    return y_true, y_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the SHARP baseline.")
    parser.add_argument("--config", type=str, default="config/base_config.yaml")
    parser.add_argument("--local", action="store_true", help="Load the checkpoint from local disk.")
    args = parser.parse_args()
    main(args.config, local=args.local)