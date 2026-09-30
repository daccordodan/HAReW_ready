"""
Evaluation metrics matching Paper 2's reported benchmark tables.
"""

from __future__ import annotations

import numpy as np

S7_REFERENCE_ACCURACY = 0.9599 


def compute_accuracy_per_activity(
    y_true: list[int], y_pred: list[int], class_names: list[str]
) -> tuple[dict[str, float], float]:
    """
    Computes overall and per-activity classification accuracy scores.

    Args:
        y_true: ground-truth class index targets
        y_pred: predicted class index values
        class_names: list of target class names

    Returns:
        dictionary of per-activity accuracies and overall accuracy
    """

    accuracy_pa: dict[str, float] = {}
    if len(y_true) == 0:
        return {class_name: 0.0 for class_name in class_names}, 0.0

    correct = 0
    correct_pa = [0] * len(class_names)
    counts_pa = [0] * len(class_names)
    for t, p in zip(y_true, y_pred):
        counts_pa[t] += 1
        if t == p:
            correct_pa[t] += 1
            correct += 1
            
    for class_idx, class_name in enumerate(class_names):
        accuracy_pa[class_name] = (
            correct_pa[class_idx] / counts_pa[class_idx]
            if counts_pa[class_idx] > 0
            else 0.0
        )

    correct /= len(y_pred)

    return accuracy_pa, correct


def compute_f1_per_activity(y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict[str, float]:
    """
    Computes per-activity F1 scores from ground truth and predictions.

    Args:
        y_true: ground-truth class index targets
        y_pred: predicted class index values
        class_names: ordered list of class label names

    Returns:
        dictionary mapping class name to F1 score
    """
    f1_scores: dict[str, float] = {}
    for class_idx, class_name in enumerate(class_names):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == class_idx and p == class_idx)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != class_idx and p == class_idx)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == class_idx and p != class_idx)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores[class_name] = f1
    return f1_scores


def compute_confusion_matrix(y_true: list[int], y_pred: list[int], n_classes: int) -> np.ndarray:
    """
    Computes confusion matrix across true and predicted class labels.

    Args:
        y_true: ground-truth class index targets
        y_pred: predicted class index values
        n_classes: total number of target activity classes

    Returns:
        confusion matrix array of shape (n_classes, n_classes)
    """
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def report_per_set_accuracy(results_by_set: dict[str, float]) -> str:
    """
    Formats per-set accuracy results alongside reference benchmark values.

    Args:
        results_by_set: dict mapping set identifier to measured accuracy

    Returns:
        formatted multi-line summary report string
    """
    lines = ["Per-set accuracy:"]
    for set_id, acc in results_by_set.items():
        if set_id == "S7":
            lines.append(f"  {set_id}: {acc:.4f}  (paper reference: {S7_REFERENCE_ACCURACY:.4f})")
        else:
            lines.append(f"  {set_id}: {acc:.4f}")
    mean_acc = sum(results_by_set.values()) / len(results_by_set) if results_by_set else 0.0
    lines.append(f"Mean across sets: {mean_acc:.4f}  (paper reference: >0.95)")
    return "\n".join(lines)