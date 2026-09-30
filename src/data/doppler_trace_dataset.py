"""Dataset for pre-computed, per-antenna Doppler traces."""

from __future__ import annotations

import logging
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.label_mapping import SCENARIO_TO_SUBJECT, is_in_scope, raw_to_class_index

DEFAULT_NW = 340
DEFAULT_NANT = 4
DEFAULT_STRIDE = 1

_FILENAME_RE = re.compile(
    r"^(?P<set_num>S\d+)(?P<repetition>[a-z])_"
    r"(?P<activity>[A-Za-z0-9]+)_stream_(?P<antenna>\d+)\.txt$"
)


@dataclass(frozen=True)
class StreamFileInfo:
    """
    Metadata parsed from one per-antenna stream file.

    Args:
        path: file path to stream
        set_id: dataset set identifier
        repetition: recording repetition letter
        activity_code: activity raw code string
        antenna_idx: antenna stream index
    """

    path: Path
    set_id: str
    repetition: str
    activity_code: str
    antenna_idx: int


def discover_stream_files(root_dir: Path) -> list[StreamFileInfo]:
    """
    Finds all stream files following the project naming convention.

    Args:
        root_dir: root directory path to scan

    Returns:
        list of stream file metadata objects
    """
    infos: list[StreamFileInfo] = []
    for path in sorted(root_dir.rglob("*.txt")):
        match = _FILENAME_RE.fullmatch(path.name)
        if match is None:
            continue
        infos.append(
            StreamFileInfo(
                path=path,
                set_id=match.group("set_num"),
                repetition=match.group("repetition"),
                activity_code=match.group("activity"),
                antenna_idx=int(match.group("antenna")),
            )
        )
    return infos


def count_stream_files(root_dir: str | Path) -> int:
    """
    Counts recognized stream files under root directory.

    Args:
        root_dir: root directory path to scan

    Returns:
        total count of valid stream files
    """
    n = 0
    root = Path(root_dir)
    for txt_path in sorted(root.rglob("*.txt")):
        match = _FILENAME_RE.match(txt_path.name)
        if match is not None:
            n += 1

    return n


def _load_pickled_array(path: Path) -> np.ndarray:
    """
    Loads and mean-centers pickled Doppler trace array.

    Args:
        path: path to pickled array file

    Returns:
        mean-centered 2D float32 array
    """
    with path.open("rb") as file:
        value = pickle.load(file)

    array = np.asarray(value)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(
            f"Expected a numeric 2-D NumPy array in {path}, "
            f"got {type(value).__name__} with shape {getattr(array, 'shape', None)}"
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"Empty Doppler trace in {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite values in {path}")
    array = np.asarray(array, dtype=np.float32)
    return array - array.mean(axis=0, keepdims=True)


class DopplerTraceDataset(Dataset):
    """
    Windowed dataset for multi-antenna Doppler traces.

    Args:
        root_dir: root path to dataset recordings
        sets_to_include: sequence of set IDs to include
        window_size: temporal window length
        stride: sliding window step size
        n_antennas: expected number of antenna streams
        temporal_split: split type ('train', 'val', 'test', 'all')
        train_split: ratio of recording allocated for training
        val_split: ratio of recording allocated for validation
        expand_train_antennas: flag to index antennas independently during training
        transform: data augmentation function
        logger: logger instance for output messages
    """

    def __init__(
        self,
        root_dir: str | Path,
        sets_to_include: Sequence[str],
        window_size: int = DEFAULT_NW,
        stride: int = DEFAULT_STRIDE,
        n_antennas: int = DEFAULT_NANT,
        temporal_split: Literal["train", "val", "test", "all"] = "all",
        train_split: float = 0.6,
        val_split: float = 0.2,
        expand_train_antennas: bool = True,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive or None")
        if n_antennas <= 0:
            raise ValueError("n_antennas must be positive")
        if temporal_split not in {"train", "val", "test", "all"}:
            raise ValueError(f"Unknown temporal split: {temporal_split}")
        if not 0.0 <= train_split <= 1.0 or not 0.0 <= val_split <= 1.0:
            raise ValueError("train_split and val_split must be between 0 and 1")
        if train_split + val_split > 1.0:
            raise ValueError("train_split + val_split must not exceed 1")

        self.root_dir = Path(root_dir)
        self.sets_to_include = tuple(sets_to_include)
        self.window_size = window_size
        self.stride = stride
        self.n_antennas = n_antennas
        self.temporal_split = temporal_split
        self.train_split = train_split
        self.val_split = val_split
        self.expand_train_antennas = expand_train_antennas
        self.transform = transform
        self.logger = logger

        self._recordings: list[np.ndarray] = []
        self._window_indices: list[tuple[int, int, int, int, int | None]] = []
        self._build_index()

    def _build_index(self) -> None:
        """
        Loads recordings and builds window index array.
        """
        started_at = time.perf_counter()
        groups: dict[tuple[str, str, str], dict[int, Path]] = {}

        for info in discover_stream_files(self.root_dir):
            if info.set_id not in self.sets_to_include or not is_in_scope(info.activity_code):
                continue
            key = (info.set_id, info.repetition, info.activity_code)
            streams = groups.setdefault(key, {})
            if info.antenna_idx in streams:
                raise ValueError(f"Duplicate antenna {info.antenna_idx} in {key}")
            streams[info.antenna_idx] = info.path

        for (set_id, repetition, activity_code), paths in sorted(groups.items()):
            missing = [index for index in range(self.n_antennas) if index not in paths]
            if missing:
                raise ValueError(
                    f"Missing antenna stream(s) {missing} for "
                    f"{set_id}{repetition}_{activity_code}"
                )

            streams = [_load_pickled_array(paths[index]) for index in range(self.n_antennas)]
            feature_dims = {stream.shape[1] for stream in streams}
            if len(feature_dims) != 1:
                raise ValueError(
                    f"Antenna Doppler dimensions disagree for "
                    f"{set_id}{repetition}_{activity_code}: {feature_dims}"
                )

            length = min(stream.shape[0] for stream in streams)
            start, end = self.evaluate_temp_split(length)
            if end - start < self.window_size:
                continue

            recording = np.stack(
                [stream[start:end] for stream in streams],
                axis=0,
            )
            recording_index = len(self._recordings)
            self._recordings.append(recording)

            label = raw_to_class_index(activity_code)
            subject = SCENARIO_TO_SUBJECT.get(set_id, -1)
            n_windows = 1 + (recording.shape[1] - self.window_size) // self.stride
            for offset in range(0, n_windows * self.stride, self.stride):
                if self.temporal_split == "train" and self.expand_train_antennas:
                    self._window_indices.extend(
                        (recording_index, offset, label, subject, antenna)
                        for antenna in range(self.n_antennas)
                    )
                else:
                    self._window_indices.append(
                        (recording_index, offset, label, subject, None)
                    )

        if self.logger is not None:
            self.logger.info(
                "Loaded %d recordings and %d %s windows in %.2fs",
                len(self._recordings),
                len(self._window_indices),
                self.temporal_split,
                time.perf_counter() - started_at,
            )

    def __len__(self) -> int:
        """
        Calculates total number of window samples.

        Returns:
            number of indexed window samples
        """
        return len(self._window_indices)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], dict[str, int]]:
        """
        Retrieves window tensor sample and metadata target labels.

        Args:
            index: window sample index

        Returns:
            window tensor or transformed tuple and target metadata dict
        """
        recording_index, start, label, subject, antenna = self._window_indices[index]
        if antenna is None:
            window = self._recordings[recording_index][
                :, start : start + self.window_size, :
            ]
        else:
            window = self._recordings[recording_index][
                antenna : antenna + 1, start : start + self.window_size, :
            ]

        if self.transform is None:
            sample: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = torch.from_numpy(
                window.copy()
            )
        else:
            first = torch.as_tensor(self.transform(window.copy()), dtype=torch.float32)
            second = torch.as_tensor(self.transform(window.copy()), dtype=torch.float32)
            sample = first, second

        return sample, {"label": label, "subject": subject}

    def evaluate_temp_split(self, length: int) -> tuple[int, int]:
        """
        Calculates temporal boundary indices for configured split.

        Args:
            length: total recording time length

        Returns:
            start and end time index tuple
        """
        if self.temporal_split == "all":
            return 0, length

        train_end = int(length * self.train_split)
        val_end = train_end + int(length * self.val_split)
        if self.temporal_split == "train":
            return 0, train_end
        if self.temporal_split == "val":
            return train_end, val_end
        return val_end, length


def build_train_val_split(
    root_dir: str | Path,
    set_id: Literal["S1", "S2", "S3", "S4", "S5", "S6", "S7"],
    window_size: int = DEFAULT_NW,
    stride: int = DEFAULT_STRIDE,
    n_antennas: int = DEFAULT_NANT,
    train_split: float = 0.6,
    val_split: float = 0.2,
    logger: logging.Logger | None = None,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[DopplerTraceDataset, torch.utils.data.Subset, torch.utils.data.Subset, torch.utils.data.Subset]:
    """
    Builds full, train, validation, and test subsets for a set ID.

    Args:
        root_dir: root directory path
        set_id: dataset set identifier
        window_size: temporal window length
        stride: sliding window stride
        n_antennas: number of antenna streams
        train_split: training set ratio
        val_split: validation set ratio
        logger: logger instance
        transform: data augmentation transform function

    Returns:
        tuple containing full dataset and train, val, test subsets
    """
    common = dict(
        root_dir=root_dir,
        sets_to_include=(set_id,),
        window_size=window_size,
        stride=stride,
        n_antennas=n_antennas,
        train_split=train_split,
        val_split=val_split,
        logger=logger,
        transform=transform,
    )
    train_dataset = DopplerTraceDataset(**common, temporal_split="train")
    val_dataset = DopplerTraceDataset(**common, temporal_split="val")
    test_dataset = DopplerTraceDataset(**common, temporal_split="test")
    full_dataset = DopplerTraceDataset(**common, temporal_split="all")
    return (
        full_dataset,
        torch.utils.data.Subset(train_dataset, range(len(train_dataset))),
        torch.utils.data.Subset(val_dataset, range(len(val_dataset))),
        torch.utils.data.Subset(test_dataset, range(len(test_dataset))),
    )


def build_zero_shot_test_set(
    root_dir: str | Path,
    set_id: Literal["S1", "S2", "S3", "S4", "S5", "S6", "S7"],
    window_size: int = DEFAULT_NW,
    stride: int = DEFAULT_STRIDE,
    n_antennas: int = DEFAULT_NANT,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> DopplerTraceDataset:
    """
    Builds an unsplit dataset for zero-shot evaluation on one set.

    Args:
        root_dir: root directory path
        set_id: dataset set identifier
        window_size: temporal window length
        stride: sliding window stride
        n_antennas: number of antenna streams
        transform: data augmentation transform function

    Returns:
        unsplit DopplerTraceDataset instance
    """
    return DopplerTraceDataset(
        root_dir,
        sets_to_include=(set_id,),
        window_size=window_size,
        stride=stride,
        n_antennas=n_antennas,
        transform=transform,
    )