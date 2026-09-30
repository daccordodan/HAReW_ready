"""
Baseline SHARP reproduction: zero-shot evaluation across S1-S7.
"""

from __future__ import annotations

from pathlib import Path
import logging

import torch

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from huggingface_hub import hf_hub_download


def load_checkpoint_to_model(
    config: dict, 
    model: torch.nn.Module, 
    checkpoint_name: str, 
    device: str, 
    logger: logging.Logger, 
    local: bool = False
) -> torch.nn.Module:
    """
    Loads saved model state parameters from a local checkpoint or Hugging Face.

    Args:
        config: configuration parameter dictionary
        model: neural network model instance
        checkpoint_name: checkpoint file path or identifier name
        device: "cuda" or "cpu"
        logger: logger instance for output messages
        local: flag to load from local directory

    Returns:
        model loaded with saved checkpoint weights
    """
    if local:
        checkpoint_path = Path(checkpoint_name)
        if not checkpoint_path.is_absolute() and not checkpoint_path.exists():
            checkpoint_path = Path(config["paths"]["output_dir"]) / "checkpoints" / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Local checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(hf_hub_download(
            repo_id="danieledaccordo/HAReW",
            filename="checkpoints_dir/" + checkpoint_name,
            repo_type="model",
        ), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded checkpoint from epoch %d", checkpoint.get("epoch", -1))
    return model


def plot_acc_f1_results_pa(accuracy: dict[str, dict[str, float]], fscore: dict[str, dict[str, float]], figures_dir: Path) -> None:
    """
    Generates and saves a table plot of per-activity accuracy and F1 scores.

    Args:
        accuracy: nested dict mapping set and class to accuracy values
        fscore: nested dict mapping set and class to F1 scores
        figures_dir: output directory for saving the plot
    """
    df_acc = pd.DataFrame(accuracy)
    df_f1 = pd.DataFrame(fscore)

    df_combined = "Acc: " + df_acc.round(2).astype(str) + "\nF1: " + df_f1.round(2).astype(str)
    _, ax = plt.subplots(figsize=(6, 2.5))
    ax.set_title('Accuracy and F1-scores of the baseline')
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df_combined.values,
        rowLabels=df_combined.index,
        colLabels=df_combined.columns,
        loc='center',
        cellLoc='center'
    )
    table.scale(1, 2.2)

    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold')

    plt.savefig(figures_dir / "accuracy_f1_pa_table.png", bbox_inches='tight', dpi=300)
    plt.show()


def plot_conf_mat_results_pa(confmat: dict, set_ids: list[str], class_names: list[str], figures_dir: Path) -> None:
    """
    Generates and saves confusion matrix plots for each dataset subset.

    Args:
        confmat: dict mapping set IDs to confusion matrices
        set_ids: list of set identifiers
        class_names: list of class label names
        figures_dir: output directory for saving plot figures
    """
    for set_id in set_ids:
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=confmat[set_id], display_labels=class_names)
        disp.plot(cmap='viridis', ax=ax, xticks_rotation='horizontal')

        ax.set_title(f'Confusion Matrix of the baseline: set {set_id}')

        plt.tight_layout()
        plt.savefig(figures_dir / f"conf_mat_{set_id}.png", bbox_inches='tight', dpi=300)
        plt.close(fig)


def write_report_performances(accuracy_by_set: dict, accuracy_by_set_pa: dict, fscore_by_set_pa: dict, files_dir: Path) -> None:
    """
    Writes evaluation accuracy and F1 score performance reports to text files.

    Args:
        accuracy_by_set: dict mapping set ID to overall accuracy
        accuracy_by_set_pa: dict mapping set ID to per-activity accuracy
        fscore_by_set_pa: dict mapping set ID to per-activity F1 scores
        files_dir: output directory path for text reports
    """
    with open(files_dir / "per_set_accuracy.txt", "w", encoding="utf-8") as f:
        for set_id, acc in accuracy_by_set.items():
            f.write(f"{set_id}\t{acc:.4f}\n")
            
    with open(files_dir / "per_activity_accuracy.txt", "w", encoding="utf-8") as f:
        for set_id, acts in accuracy_by_set_pa.items():
            f.write(f"{set_id}\n")
            for act, acc in acts.items():
                f.write(f"{act}\t{acc:.4f}\n")

    with open(files_dir / "per_set_f1.txt", "w", encoding="utf-8") as f:
        for set_id, acts in fscore_by_set_pa.items():
            f.write(f"{set_id}\n")
            for act, fs in acts.items():
                f.write(f"{act}\t{fs:.4f}\n")