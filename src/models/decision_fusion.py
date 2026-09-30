"""
Multi-antenna Decision Fusion strategy for SHARP.
"""

from __future__ import annotations

from collections import Counter

import torch

DEFAULT_N_ANTENNAS = 4


def majority_vote_fusion(per_antenna_predictions: list[int], n_antennas: int = DEFAULT_N_ANTENNAS) -> int | None:
    """
    Attempts majority-vote fusion across per-antenna class predictions.

    Args:
        per_antenna_predictions: predicted class indices per antenna
        n_antennas: total number of antennas

    Returns:
        majority class index if consensus threshold met, else None
    """
    counts = Counter(per_antenna_predictions)
    top_class, top_count = counts.most_common(1)[0]
    if top_count >= n_antennas - 1:
        return top_class
    return None


def summed_vector_fusion(per_antenna_activity_vectors: torch.Tensor) -> int:
    """
    Fuses per-antenna probability vectors via element-wise summation and argmax.

    Args:
        per_antenna_activity_vectors: softmax probability tensor across antennas

    Returns:
        fused predicted class index
    """
    summed = per_antenna_activity_vectors.sum(dim=0)
    return int(torch.argmax(summed).item())


def fuse_predictions(
    per_antenna_activity_vectors: torch.Tensor,
    n_antennas: int = DEFAULT_N_ANTENNAS,
) -> int:
    """
    Executes decision fusion pipeline using majority vote or summed vectors fallback.

    Args:
        per_antenna_activity_vectors: softmax probability tensor for one sample
        n_antennas: total number of antennas

    Returns:
        fused predicted class index
    """
    per_antenna_predictions = torch.argmax(per_antenna_activity_vectors, dim=1).tolist()
    majority_result = majority_vote_fusion(per_antenna_predictions, n_antennas)
    if majority_result is None:
        return summed_vector_fusion(per_antenna_activity_vectors)
    return majority_result


def fuse_batch(batch_activity_vectors: torch.Tensor, n_antennas: int = DEFAULT_N_ANTENNAS) -> torch.Tensor:
    """
    Applies decision fusion strategy across a batch of multi-antenna predictions.

    Args:
        batch_activity_vectors: softmax probability tensor for batch
        n_antennas: total number of antennas

    Returns:
        tensor of fused predicted class indices per sample
    """
    fused = [fuse_predictions(batch_activity_vectors[i], n_antennas) for i in range(batch_activity_vectors.shape[0])]
    return torch.tensor(fused, dtype=torch.long)