"""
Centralizes the mapping activity-code -> class.
"""

from __future__ import annotations

TARGET_CLASSES: list[str] = ["E", "W", "R", "J", "L"] #index position IS the model's output index.

RAW_TO_TARGET: dict[str, str] = {
    "E": "E",
    "W": "W",
    "R": "R",
    "R1": "R",
    "J1": "J",
    "J2": "J",
    "J3": "J",
    "L": "L",
    "L1": "L",
    "L2": "L",
}

SCENARIO_TO_SUBJECT: dict[str, int] = {
    "S1" : 1,
    "S2" : 1,
    "S3" : 2,
    "S4" : 1,
    "S5" : 2,
    "S6" : 1,
    "S7" : 3,
}

TARGET_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(TARGET_CLASSES)}
INDEX_TO_TARGET: dict[int, str] = {i: name for name, i in TARGET_TO_INDEX.items()}


def is_in_scope(raw_activity_code: str) -> bool:
    """
    Checks if raw activity code maps to target class.

    Args:
        raw_activity_code: activity code parsed from filename

    Returns:
        True if code is in target scope, else False
    """
    return raw_activity_code in RAW_TO_TARGET


def raw_to_class_index(raw_activity_code: str) -> int:
    """
    Maps raw activity code string to integer class index.

    Args:
        raw_activity_code: activity code parsed from filename

    Returns:
        integer index position in TARGET_CLASSES
    """
    target_name = RAW_TO_TARGET[raw_activity_code]
    return TARGET_TO_INDEX[target_name]