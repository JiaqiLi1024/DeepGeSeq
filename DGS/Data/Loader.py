"""High-level DataLoader builders for genomic sequence workflows.

Purpose:
    Expose first-class PyTorch data loading APIs for FASTA/BED driven workflows.

Main Responsibilities:
    - Build sequence-only dataloaders directly from FASTA and BED intervals.
    - Build supervised dataloaders by aligning BED/BigWig targets to intervals.
    - Provide streaming datasets that lazily fetch sequences per sample.

Key Runtime Notes:
    - Streaming mode is the default and does not cache all extracted sequences.
    - Each DataLoader worker lazily opens its own FASTA reader for safe parallel IO.
    - Single-sample sequence shape is `(sequence_length, 4)` before collation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..IO import BigWigReader
from .Dataset import GenomicDataset, SeqDataset, create_dataloader
from .Interval import Interval
from .Sampler import chromosome_split, random_split
from .Sequence import Genome, one_hot_encode
from .Target import Target

logger = logging.getLogger("dgs.loader")

IntervalLike = Union[str, Path, pd.DataFrame, Interval]
TargetTasks = List[Dict[str, Any]]
ProfileTasks = List[Dict[str, Any]]

__all__ = [
    "StreamingSeqDataset",
    "StreamingGenomicDataset",
    "StreamingProfileDataset",
    "build_sequence_dataloader",
    "build_profile_dataloader",
    "build_profile_dataloaders",
    "build_supervised_dataloader",
    "build_supervised_dataloaders",
]


def _coerce_intervals(
    intervals: IntervalLike,
    chrom_col: str = "chrom",
    start_col: str = "start",
    end_col: str = "end",
    **kwargs: Any,
) -> Interval:
    """Normalize supported interval inputs to a standard `Interval` object."""
    if isinstance(intervals, Interval):
        data = intervals.data.copy()
        rename_map = {}
        if intervals.chrom_col != "chrom":
            rename_map[intervals.chrom_col] = "chrom"
        if intervals.start_col != "start":
            rename_map[intervals.start_col] = "start"
        if intervals.end_col != "end":
            rename_map[intervals.end_col] = "end"
        if rename_map:
            data = data.rename(columns=rename_map)
        return Interval(data)

    if isinstance(intervals, pd.DataFrame):
        data = intervals.copy()
        rename_map = {}
        if chrom_col != "chrom":
            rename_map[chrom_col] = "chrom"
        if start_col != "start":
            rename_map[start_col] = "start"
        if end_col != "end":
            rename_map[end_col] = "end"
        if rename_map:
            data = data.rename(columns=rename_map)
        return Interval(data)

    return Interval(intervals, **kwargs)


def _normalize_mode(mode: str) -> str:
    normalized = mode.lower()
    if normalized not in {"streaming", "cached"}:
        raise ValueError("mode must be either 'streaming' or 'cached'")
    return normalized


def _validate_fasta_path(fasta_path: Union[str, Path]) -> Path:
    """Validate that the requested FASTA file exists before building loaders."""
    path = Path(fasta_path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")
    return path


def _validate_intervals_for_fasta(
    intervals: Interval,
    fasta_path: Union[str, Path],
    strand_aware: bool,
) -> None:
    """Validate interval content against a FASTA index with clear user errors."""
    data = intervals.data
    if data.empty:
        raise ValueError("No intervals were provided; cannot build a DataLoader.")

    required = {"chrom", "start", "end"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Intervals are missing required columns: {missing}")

    genome = Genome(fasta_path)
    try:
        lengths = genome._reader.lengths
    finally:
        genome.close()

    lengths = {str(chrom): length for chrom, length in lengths.items()}
    interval_chroms = set(data["chrom"].astype(str))
    missing_chroms = sorted(interval_chroms - set(lengths))
    if missing_chroms:
        preview = ", ".join(missing_chroms[:5])
        extra = "" if len(missing_chroms) <= 5 else f" and {len(missing_chroms) - 5} more"
        raise ValueError(
            "Intervals contain chromosomes absent from the FASTA index: "
            f"{preview}{extra}."
        )

    chrom_lengths = data["chrom"].astype(str).map(lengths)
    out_of_bounds = data["end"] > chrom_lengths
    if out_of_bounds.any():
        row = data.loc[out_of_bounds].iloc[0]
        chrom = str(row["chrom"])
        raise ValueError(
            "Interval extends beyond FASTA chromosome length: "
            f"{row['chrom']}:{int(row['start'])}-{int(row['end'])} "
            f"(chrom length {int(lengths[chrom])})."
        )

    if strand_aware and "strand" in data.columns:
        invalid_strands = sorted(set(data["strand"].dropna()) - {"+", "-", "."})
        if invalid_strands:
            raise ValueError(
                "Invalid strand values in intervals. Expected '+', '-', or '.', "
                f"found: {invalid_strands}."
            )


def _validate_target_tasks(target_tasks: TargetTasks) -> None:
    """Validate target task configuration before expensive data loading."""
    if not target_tasks:
        raise ValueError("target_tasks must contain at least one task definition.")

    for idx, task in enumerate(target_tasks):
        missing = sorted({"task_name", "file_path", "file_type"} - set(task))
        if missing:
            raise ValueError(
                f"target_tasks[{idx}] is missing required fields: {missing}."
            )

        file_path = Path(task["file_path"])
        if not file_path.exists():
            raise FileNotFoundError(
                f"Target file for task '{task['task_name']}' not found: {file_path}"
            )

        file_type = str(task["file_type"]).lower()
        if file_type not in {"bed", "bigwig"}:
            raise ValueError(
                f"Unsupported file_type for task '{task['task_name']}': {file_type}. "
                "Expected 'bed' or 'bigwig'."
            )


def _validate_profile_tasks(profile_tasks: ProfileTasks) -> None:
    """Validate BigWig-backed profile target definitions."""
    if not profile_tasks:
        raise ValueError("profile_tasks must contain at least one task definition.")

    for idx, task in enumerate(profile_tasks):
        missing = sorted({"task_name", "file_path", "file_type"} - set(task))
        if missing:
            raise ValueError(
                f"profile_tasks[{idx}] is missing required fields: {missing}."
            )

        file_path = Path(task["file_path"])
        if not file_path.exists():
            raise FileNotFoundError(
                f"Profile target file for task '{task['task_name']}' not found: {file_path}"
            )

        file_type = str(task["file_type"]).lower()
        if file_type != "bigwig":
            raise ValueError(
                f"Unsupported profile file_type for task '{task['task_name']}': {file_type}. "
                "Profile targets currently require 'bigwig'."
            )


def _prepare_loader_inputs(
    fasta_path: Union[str, Path],
    intervals_path: IntervalLike,
    strand_aware: bool,
) -> Tuple[Path, Interval]:
    """Normalize and validate common FASTA/interval loader inputs."""
    fasta_path = _validate_fasta_path(fasta_path)
    intervals = _coerce_intervals(intervals_path)
    _validate_intervals_for_fasta(intervals, fasta_path, strand_aware)
    return fasta_path, intervals


def _validate_fixed_interval_width(intervals: Interval, context: str) -> int:
    """Return interval width and require all intervals to have the same width."""
    widths = (intervals.data["end"].astype(int) - intervals.data["start"].astype(int))
    if (widths <= 0).any():
        raise ValueError(f"{context} requires intervals with positive width.")
    unique_widths = sorted(set(widths.tolist()))
    if len(unique_widths) != 1:
        raise ValueError(
            f"{context} requires fixed-width intervals; found widths {unique_widths[:5]}."
        )
    return int(unique_widths[0])


def _read_profile_targets(
    intervals: Interval,
    profile_tasks: ProfileTasks,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, List[str]]:
    """Read profile target tracks into ``(N, C, L)`` arrays."""
    arrays = []
    task_names = []
    interval_data = intervals.data[["chrom", "start", "end"]].copy()

    for task in profile_tasks:
        task_names.append(str(task["task_name"]))
        reader = BigWigReader(task["file_path"])
        values = reader.read(
            interval_data,
            bin_size=task.get("bin_size", None),
            aggfunc=task.get("aggfunc", None),
        ).astype(dtype, copy=False)
        if values.ndim != 2:
            raise ValueError(
                f"Profile task '{task['task_name']}' must produce a 2D "
                f"(intervals, positions) array, got shape {values.shape}."
            )
        if values.shape[0] != len(intervals.data):
            raise ValueError(
                f"Profile task '{task['task_name']}' produced {values.shape[0]} rows "
                f"for {len(intervals.data)} intervals."
            )
        arrays.append(values)

    lengths = {array.shape[1] for array in arrays}
    if len(lengths) != 1:
        raise ValueError(
            "All profile tasks must produce the same output length; "
            f"found lengths {sorted(lengths)}."
        )

    return np.stack(arrays, axis=1).astype(dtype, copy=False), task_names


class StreamingSeqDataset(Dataset):
    """Sequence dataset that fetches FASTA intervals on demand.

    Args:
        intervals: Genomic intervals as a BED path, DataFrame, or `Interval`.
        genome_path: Reference genome FASTA path.
        strand_aware: Whether to reverse-complement negative-strand intervals.
        dtype: Output dtype for one-hot encoded sequences.

    Notes:
        The dataset stores only interval metadata and the FASTA path. The FASTA
        reader is opened lazily per process so PyTorch workers do not share file
        handles inherited from the parent process.
    """

    def __init__(
        self,
        intervals: IntervalLike,
        genome_path: Union[str, Path],
        strand_aware: bool = True,
        dtype: np.dtype = np.float32,
        chrom_col: str = "chrom",
        start_col: str = "start",
        end_col: str = "end",
        **interval_kwargs: Any,
    ):
        self.intervals = _coerce_intervals(
            intervals,
            chrom_col=chrom_col,
            start_col=start_col,
            end_col=end_col,
            **interval_kwargs,
        )
        self.genome_path = Path(genome_path)
        self.strand_aware = strand_aware
        self.dtype = dtype
        self._genome: Optional[Genome] = None

    def __len__(self) -> int:
        """Return the number of intervals in the dataset."""
        return len(self.intervals.data)

    def __getstate__(self) -> Dict[str, Any]:
        """Drop open FASTA handles before PyTorch worker pickling."""
        state = self.__dict__.copy()
        state["_genome"] = None
        return state

    def close(self) -> None:
        """Close the lazily opened genome reader, if present."""
        if self._genome is not None:
            self._genome.close()
            self._genome = None

    def _get_genome(self) -> Genome:
        if self._genome is None:
            self._genome = Genome(self.genome_path)
        return self._genome

    def _interval_at(self, idx: int) -> pd.DataFrame:
        return self.intervals.data.iloc[[idx]]

    def __getitem__(self, idx: int) -> np.ndarray:
        """Fetch, strand-normalize, and one-hot encode one interval sequence."""
        idx = int(idx)
        seq = self._get_genome().extract_sequences(
            self._interval_at(idx),
            strand_aware=self.strand_aware,
        )[0]
        return one_hot_encode(seq.sequence, dtype=self.dtype)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class StreamingGenomicDataset(StreamingSeqDataset):
    """Streaming FASTA dataset with interval-aligned target labels."""

    def __init__(
        self,
        intervals: IntervalLike,
        genome_path: Union[str, Path],
        targets: Union[Target, TargetTasks],
        strand_aware: bool = True,
        dtype: np.dtype = np.float32,
        chrom_col: str = "chrom",
        start_col: str = "start",
        end_col: str = "end",
        **interval_kwargs: Any,
    ):
        super().__init__(
            intervals,
            genome_path,
            strand_aware=strand_aware,
            dtype=dtype,
            chrom_col=chrom_col,
            start_col=start_col,
            end_col=end_col,
            **interval_kwargs,
        )
        self.targets = targets if isinstance(targets, Target) else Target(self.intervals.data, targets)
        self.labels = self.targets.get_labels()
        self.task_info = self.targets.get_task_info()

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return one encoded sequence and its float32 label vector."""
        idx = int(idx)
        seq = super().__getitem__(idx)
        label = self.labels[idx]
        return seq, label.astype(np.float32)


class StreamingProfileDataset(StreamingSeqDataset):
    """Streaming FASTA dataset with interval-aligned profile BigWig targets.

    Profile targets are read from BigWig files as fixed-width per-position
    arrays and returned in channel-first shape ``(channels, length)`` for each
    sample. Batched targets therefore have shape ``(batch, channels, length)``.
    """

    def __init__(
        self,
        intervals: IntervalLike,
        genome_path: Union[str, Path],
        profile_tasks: ProfileTasks,
        strand_aware: bool = True,
        dtype: np.dtype = np.float32,
        profile_dtype: np.dtype = np.float32,
        return_counts: bool = False,
        counts_log1p: bool = True,
        reverse_negative_strand_targets: bool = True,
        chrom_col: str = "chrom",
        start_col: str = "start",
        end_col: str = "end",
        **interval_kwargs: Any,
    ):
        super().__init__(
            intervals,
            genome_path,
            strand_aware=strand_aware,
            dtype=dtype,
            chrom_col=chrom_col,
            start_col=start_col,
            end_col=end_col,
            **interval_kwargs,
        )
        _validate_profile_tasks(profile_tasks)
        self.profile_width = _validate_fixed_interval_width(
            self.intervals,
            "Profile target loading",
        )
        self.profile_targets, self.task_names = _read_profile_targets(
            self.intervals,
            profile_tasks,
            dtype=profile_dtype,
        )
        self.return_counts = bool(return_counts)
        self.counts_log1p = bool(counts_log1p)
        self.reverse_negative_strand_targets = bool(reverse_negative_strand_targets)

        if (
            self.reverse_negative_strand_targets
            and "strand" in self.intervals.data.columns
        ):
            negative = self.intervals.data["strand"].astype(str).to_numpy() == "-"
            if np.any(negative):
                self.profile_targets[negative] = self.profile_targets[negative, :, ::-1]
                self.profile_targets = np.ascontiguousarray(self.profile_targets)

        counts = self.profile_targets.sum(axis=(1, 2), dtype=np.float64)[:, None]
        if self.counts_log1p:
            counts = np.log1p(counts)
        self.count_targets = counts.astype(profile_dtype, copy=False)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """Return one encoded sequence and its profile target."""
        idx = int(idx)
        seq = super().__getitem__(idx)
        profile = self.profile_targets[idx]
        if self.return_counts:
            return seq, (profile, self.count_targets[idx])
        return seq, profile


def _build_sequence_dataset(
    fasta_path: Union[str, Path],
    intervals: Interval,
    mode: str,
    strand_aware: bool,
) -> Dataset:
    if mode == "streaming":
        return StreamingSeqDataset(intervals, fasta_path, strand_aware=strand_aware)

    genome = Genome(fasta_path)
    return SeqDataset(intervals, genome, strand_aware=strand_aware)


def _build_supervised_dataset(
    fasta_path: Union[str, Path],
    intervals: Interval,
    target_tasks: TargetTasks,
    mode: str,
    strand_aware: bool,
) -> Dataset:
    targets = Target(intervals.data, target_tasks)
    if mode == "streaming":
        return StreamingGenomicDataset(
            intervals,
            fasta_path,
            targets,
            strand_aware=strand_aware,
        )

    genome = Genome(fasta_path)
    return GenomicDataset(intervals, genome, targets, strand_aware=strand_aware)


def _build_profile_dataset(
    fasta_path: Union[str, Path],
    intervals: Interval,
    profile_tasks: ProfileTasks,
    strand_aware: bool,
    return_counts: bool,
    counts_log1p: bool,
    reverse_negative_strand_targets: bool,
) -> Dataset:
    return StreamingProfileDataset(
        intervals,
        fasta_path,
        profile_tasks,
        strand_aware=strand_aware,
        return_counts=return_counts,
        counts_log1p=counts_log1p,
        reverse_negative_strand_targets=reverse_negative_strand_targets,
    )


def build_sequence_dataloader(
    fasta_path: Union[str, Path],
    intervals_path: IntervalLike,
    batch_size: int = 32,
    mode: str = "streaming",
    strand_aware: bool = True,
    shuffle: bool = False,
    **dataloader_kwargs: Any,
) -> torch.utils.data.DataLoader:
    """Build a sequence-only dataloader from FASTA and BED-like intervals.

    Args:
        fasta_path: Reference genome FASTA path.
        intervals_path: BED path, DataFrame, or `Interval`.
        batch_size: Number of sequences per batch.
        mode: `"streaming"` for on-demand FASTA reads or `"cached"` for eager extraction.
        strand_aware: Whether to reverse-complement negative-strand intervals.
        shuffle: Whether to shuffle interval order.
        **dataloader_kwargs: Runtime options forwarded to `create_dataloader`.

    Returns:
        PyTorch DataLoader yielding one-hot encoded sequence batches.
    """
    mode = _normalize_mode(mode)
    fasta_path, intervals = _prepare_loader_inputs(fasta_path, intervals_path, strand_aware)
    dataset = _build_sequence_dataset(fasta_path, intervals, mode, strand_aware)
    return create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **dataloader_kwargs,
    )


def build_profile_dataloader(
    fasta_path: Union[str, Path],
    intervals_path: IntervalLike,
    profile_tasks: ProfileTasks,
    batch_size: int = 32,
    strand_aware: bool = True,
    shuffle: bool = True,
    return_counts: bool = False,
    counts_log1p: bool = True,
    reverse_negative_strand_targets: bool = True,
    **dataloader_kwargs: Any,
) -> torch.utils.data.DataLoader:
    """Build a profile-model dataloader from FASTA, BED intervals, and BigWigs.

    The returned loader yields ``(sequences, profiles)`` by default, where
    sequences have shape ``(batch, length, 4)`` and profiles have shape
    ``(batch, tasks, profile_length)``. Set ``return_counts=True`` to yield
    ``(sequences, (profiles, counts))`` for BPNet-style profile/count losses.
    """
    fasta_path, intervals = _prepare_loader_inputs(fasta_path, intervals_path, strand_aware)
    _validate_profile_tasks(profile_tasks)
    dataset = _build_profile_dataset(
        fasta_path,
        intervals,
        profile_tasks,
        strand_aware,
        return_counts,
        counts_log1p,
        reverse_negative_strand_targets,
    )
    return create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **dataloader_kwargs,
    )


def build_supervised_dataloader(
    fasta_path: Union[str, Path],
    intervals_path: IntervalLike,
    target_tasks: TargetTasks,
    batch_size: int = 32,
    mode: str = "streaming",
    strand_aware: bool = True,
    shuffle: bool = True,
    **dataloader_kwargs: Any,
) -> torch.utils.data.DataLoader:
    """Build a supervised dataloader from FASTA, intervals, and target tracks."""
    mode = _normalize_mode(mode)
    fasta_path, intervals = _prepare_loader_inputs(fasta_path, intervals_path, strand_aware)
    _validate_target_tasks(target_tasks)
    dataset = _build_supervised_dataset(
        fasta_path,
        intervals,
        target_tasks,
        mode,
        strand_aware,
    )
    return create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **dataloader_kwargs,
    )


def build_profile_dataloaders(
    fasta_path: Union[str, Path],
    intervals_path: IntervalLike,
    profile_tasks: ProfileTasks,
    batch_size: int = 32,
    split: str = "random",
    test_size: float = 0.2,
    val_size: float = 0.2,
    test_chroms: Optional[List[str]] = None,
    val_chroms: Optional[List[str]] = None,
    random_state: Optional[int] = None,
    strand_aware: bool = True,
    train_shuffle: bool = True,
    return_counts: bool = False,
    counts_log1p: bool = True,
    reverse_negative_strand_targets: bool = True,
    **dataloader_kwargs: Any,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train/validation/test dataloaders for profile-model workflows."""
    fasta_path, intervals = _prepare_loader_inputs(fasta_path, intervals_path, strand_aware)
    _validate_profile_tasks(profile_tasks)
    split_name = split.lower()

    if split_name in {"random", "random_split"}:
        train_intervals, val_intervals, test_intervals = random_split(
            intervals,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )
    elif split_name in {"chromosome", "chromosome_split"}:
        if test_chroms is None or val_chroms is None:
            raise ValueError("test_chroms and val_chroms are required for chromosome splitting")
        train_intervals, val_intervals, test_intervals = chromosome_split(
            intervals,
            test_chroms=test_chroms,
            val_chroms=val_chroms,
        )
    else:
        raise ValueError("split must be 'random', 'random_split', 'chromosome', or 'chromosome_split'")

    train_dataset = _build_profile_dataset(
        fasta_path,
        train_intervals,
        profile_tasks,
        strand_aware,
        return_counts,
        counts_log1p,
        reverse_negative_strand_targets,
    )
    val_dataset = _build_profile_dataset(
        fasta_path,
        val_intervals,
        profile_tasks,
        strand_aware,
        return_counts,
        counts_log1p,
        reverse_negative_strand_targets,
    )
    test_dataset = _build_profile_dataset(
        fasta_path,
        test_intervals,
        profile_tasks,
        strand_aware,
        return_counts,
        counts_log1p,
        reverse_negative_strand_targets,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        **dataloader_kwargs,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )
    return train_loader, val_loader, test_loader


def build_supervised_dataloaders(
    fasta_path: Union[str, Path],
    intervals_path: IntervalLike,
    target_tasks: TargetTasks,
    batch_size: int = 32,
    mode: str = "streaming",
    split: str = "random",
    test_size: float = 0.2,
    val_size: float = 0.2,
    test_chroms: Optional[List[str]] = None,
    val_chroms: Optional[List[str]] = None,
    random_state: Optional[int] = None,
    strand_aware: bool = True,
    train_shuffle: bool = True,
    **dataloader_kwargs: Any,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train/validation/test supervised dataloaders.

    Args:
        fasta_path: Reference genome FASTA path.
        intervals_path: BED path, DataFrame, or `Interval`.
        target_tasks: Task definitions consumed by `Target`.
        batch_size: Number of samples per batch.
        mode: `"streaming"` or `"cached"`.
        split: `"random"`/`"random_split"` or `"chromosome"`/`"chromosome_split"`.
        test_size: Test fraction for random splitting.
        val_size: Validation fraction for random splitting.
        test_chroms: Held-out test chromosomes for chromosome splitting.
        val_chroms: Held-out validation chromosomes for chromosome splitting.
        random_state: Optional random seed for random splitting.
        strand_aware: Whether to reverse-complement negative-strand intervals.
        train_shuffle: Whether to shuffle the training loader.
        **dataloader_kwargs: Runtime options forwarded to `create_dataloader`.

    Returns:
        Tuple `(train_loader, val_loader, test_loader)`.
    """
    mode = _normalize_mode(mode)
    fasta_path, intervals = _prepare_loader_inputs(fasta_path, intervals_path, strand_aware)
    _validate_target_tasks(target_tasks)
    split_name = split.lower()

    if split_name in {"random", "random_split"}:
        train_intervals, val_intervals, test_intervals = random_split(
            intervals,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )
    elif split_name in {"chromosome", "chromosome_split"}:
        if test_chroms is None or val_chroms is None:
            raise ValueError("test_chroms and val_chroms are required for chromosome splitting")
        train_intervals, val_intervals, test_intervals = chromosome_split(
            intervals,
            test_chroms=test_chroms,
            val_chroms=val_chroms,
        )
    else:
        raise ValueError("split must be 'random', 'random_split', 'chromosome', or 'chromosome_split'")

    train_dataset = _build_supervised_dataset(
        fasta_path,
        train_intervals,
        target_tasks,
        mode,
        strand_aware,
    )
    val_dataset = _build_supervised_dataset(
        fasta_path,
        val_intervals,
        target_tasks,
        mode,
        strand_aware,
    )
    test_dataset = _build_supervised_dataset(
        fasta_path,
        test_intervals,
        target_tasks,
        mode,
        strand_aware,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        **dataloader_kwargs,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )
    return train_loader, val_loader, test_loader
