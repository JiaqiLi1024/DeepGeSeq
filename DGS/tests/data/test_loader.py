"""Tests for high-level FASTA/BED dataloader builders."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import pysam
except ImportError:  # pragma: no cover - optional dependency
    pysam = None

from DGS.Data import (
    StreamingGenomicDataset,
    StreamingSeqDataset,
    build_sequence_dataloader,
    build_supervised_dataloader,
    build_supervised_dataloaders,
    one_hot_decode,
)
from DGS.Data.Interval import Interval


@pytest.fixture
def loader_files(tmp_path):
    fasta_path = tmp_path / "mini.fa"
    fasta_path.write_text(
        ">chr1\n"
        "ACGTACGTACGTACGTACGTACGT\n"
        ">chr2\n"
        "TTTTCCCCAAAAGGGGTTTTCCCC\n"
        ">chr3\n"
        "AAAACCCCGGGGTTTTAAAACCCC\n"
    )
    pysam.faidx(str(fasta_path))

    intervals_path = tmp_path / "regions.bed"
    intervals = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2", "chr2", "chr3", "chr3"],
            "start": [0, 1, 0, 4, 0, 8],
            "end": [4, 5, 4, 8, 4, 12],
            "name": [f"region_{i}" for i in range(6)],
            "score": [0] * 6,
            "strand": ["+", "-", "+", "+", "+", "+"],
        }
    )
    intervals.to_csv(intervals_path, sep="\t", index=False, header=False)

    target_path = tmp_path / "targets.bed"
    targets = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2", "chr3"],
            "start": [0, 0, 8],
            "end": [6, 10, 12],
            "value": [1, 1, 1],
        }
    )
    targets.to_csv(target_path, sep="\t", index=False, header=False)

    tasks = [
        {
            "task_name": "binding",
            "file_path": str(target_path),
            "file_type": "bed",
            "task_type": "binary",
            "target_column": "name",
        }
    ]
    return fasta_path, intervals_path, tasks


pytestmark = pytest.mark.skipif(pysam is None, reason="pysam is required for loader tests")


def _decode(encoded):
    return one_hot_decode(np.asarray(encoded), include_n=True)


def test_streaming_seq_dataset_reads_fasta_on_demand(loader_files):
    fasta_path, intervals_path, _ = loader_files
    dataset = StreamingSeqDataset(intervals_path, fasta_path, strand_aware=False)

    assert len(dataset) == 6
    assert dataset[0].shape == (4, 4)
    assert _decode(dataset[0]) == "ACGT"


def test_streaming_seq_dataset_respects_negative_strand(loader_files):
    fasta_path, intervals_path, _ = loader_files
    dataset = StreamingSeqDataset(intervals_path, fasta_path, strand_aware=True)

    assert _decode(dataset[1]) == "TACG"


def test_streaming_genomic_dataset_returns_float32_labels(loader_files):
    fasta_path, intervals_path, tasks = loader_files
    dataset = StreamingGenomicDataset(intervals_path, fasta_path, tasks, strand_aware=False)

    seq, label = dataset[0]
    assert seq.shape == (4, 4)
    assert label.shape == (1,)
    assert label.dtype == np.float32
    assert label[0] == 1.0


def test_streaming_dataset_drops_open_genome_handles_for_worker_pickling(loader_files):
    fasta_path, intervals_path, _ = loader_files
    dataset = StreamingSeqDataset(intervals_path, fasta_path, strand_aware=False)

    assert dataset._genome is None
    _ = dataset[0]
    assert dataset._genome is not None
    state = dataset.__getstate__()
    assert state["_genome"] is None


def test_build_sequence_dataloader_batches_streaming_sequences(loader_files):
    fasta_path, intervals_path, _ = loader_files
    loader = build_sequence_dataloader(
        fasta_path,
        intervals_path,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))
    assert batch.shape == (2, 4, 4)
    assert _decode(batch[0].numpy()) == "ACGT"


def test_build_sequence_dataloader_supports_worker_options(loader_files):
    """Streaming FASTA loaders should work with PyTorch worker options."""
    fasta_path, intervals_path, _ = loader_files
    loader = build_sequence_dataloader(
        fasta_path,
        intervals_path,
        batch_size=2,
        shuffle=False,
        num_workers=1,
        persistent_workers=False,
        prefetch_factor=2,
    )

    try:
        batch = next(iter(loader))
    except RuntimeError as exc:
        if "torch_shm_manager" in str(exc) or "Operation not permitted" in str(exc):
            pytest.skip("PyTorch shared-memory worker process is not permitted in this sandbox")
        raise
    assert batch.shape == (2, 4, 4)


def test_build_sequence_dataloader_reports_missing_fasta(loader_files, tmp_path):
    _, intervals_path, _ = loader_files
    missing_fasta = tmp_path / "missing.fa"

    with pytest.raises(FileNotFoundError, match="FASTA file not found"):
        build_sequence_dataloader(missing_fasta, intervals_path)


def test_build_sequence_dataloader_reports_unknown_chromosome(loader_files):
    fasta_path, intervals_path, _ = loader_files
    intervals = pd.read_csv(
        intervals_path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "name", "score", "strand"],
    )
    intervals.loc[0, "chrom"] = "chrMissing"

    with pytest.raises(ValueError, match="absent from the FASTA index"):
        build_sequence_dataloader(fasta_path, intervals)


def test_build_sequence_dataloader_reports_empty_intervals(loader_files):
    fasta_path, _, _ = loader_files
    intervals = pd.DataFrame(columns=["chrom", "start", "end"])

    with pytest.raises(ValueError, match="No intervals"):
        build_sequence_dataloader(fasta_path, intervals)


def test_build_sequence_dataloader_reports_missing_interval_columns(loader_files):
    fasta_path, _, _ = loader_files
    intervals = pd.DataFrame({"chrom": ["chr1"], "start": [0]})

    with pytest.raises(ValueError, match="Missing required columns|missing required columns"):
        build_sequence_dataloader(fasta_path, intervals)


def test_build_sequence_dataloader_reports_invalid_coordinates(loader_files):
    fasta_path, _, _ = loader_files
    intervals = pd.DataFrame({"chrom": ["chr1"], "start": [5], "end": [5]})

    with pytest.raises(ValueError, match="end <= start"):
        build_sequence_dataloader(fasta_path, intervals)


def test_build_sequence_dataloader_reports_out_of_bounds_interval(loader_files):
    fasta_path, intervals_path, _ = loader_files
    intervals = pd.read_csv(
        intervals_path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "name", "score", "strand"],
    )
    intervals.loc[0, "end"] = 10_000

    with pytest.raises(ValueError, match="beyond FASTA chromosome length"):
        build_sequence_dataloader(fasta_path, intervals)


def test_build_sequence_dataloader_reports_invalid_strand(loader_files):
    fasta_path, intervals_path, _ = loader_files
    intervals = pd.read_csv(
        intervals_path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "name", "score", "strand"],
    )
    intervals.loc[0, "strand"] = "?"

    with pytest.raises(ValueError, match="Invalid strand values"):
        build_sequence_dataloader(fasta_path, intervals, strand_aware=True)


def test_build_supervised_dataloader_batches_sequences_and_labels(loader_files):
    fasta_path, intervals_path, tasks = loader_files
    loader = build_supervised_dataloader(
        fasta_path,
        intervals_path,
        tasks,
        batch_size=3,
        shuffle=False,
        num_workers=0,
    )

    seqs, labels = next(iter(loader))
    assert seqs.shape == (3, 4, 4)
    assert labels.shape == (3, 1)
    assert labels.dtype.is_floating_point


def test_build_supervised_dataloader_reports_missing_target_file(loader_files, tmp_path):
    fasta_path, intervals_path, tasks = loader_files
    bad_tasks = [dict(tasks[0], file_path=str(tmp_path / "missing.bed"))]

    with pytest.raises(FileNotFoundError, match="Target file"):
        build_supervised_dataloader(fasta_path, intervals_path, bad_tasks)


def test_build_supervised_dataloader_reports_bad_task_schema(loader_files):
    fasta_path, intervals_path, _ = loader_files

    with pytest.raises(ValueError, match="missing required fields"):
        build_supervised_dataloader(
            fasta_path,
            intervals_path,
            [{"task_name": "binding", "file_type": "bed"}],
        )


def test_build_supervised_dataloader_reports_unsupported_target_type(loader_files):
    fasta_path, intervals_path, tasks = loader_files
    bad_tasks = [dict(tasks[0], file_type="wig")]

    with pytest.raises(ValueError, match="Unsupported file_type"):
        build_supervised_dataloader(fasta_path, intervals_path, bad_tasks)


def test_build_supervised_dataloaders_random_split(loader_files):
    fasta_path, intervals_path, tasks = loader_files
    train_loader, val_loader, test_loader = build_supervised_dataloaders(
        fasta_path,
        intervals_path,
        tasks,
        batch_size=2,
        split="random",
        test_size=0.33,
        val_size=0.33,
        random_state=11,
        train_shuffle=False,
        num_workers=0,
    )

    lengths = [len(loader.dataset) for loader in (train_loader, val_loader, test_loader)]
    assert sum(lengths) == 6
    assert lengths == [3, 2, 1]

    train_idx = set(train_loader.dataset.intervals.data.index)
    val_idx = set(val_loader.dataset.intervals.data.index)
    test_idx = set(test_loader.dataset.intervals.data.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_build_supervised_dataloaders_chromosome_split(loader_files):
    fasta_path, intervals_path, tasks = loader_files
    train_loader, val_loader, test_loader = build_supervised_dataloaders(
        fasta_path,
        intervals_path,
        tasks,
        batch_size=2,
        split="chromosome",
        val_chroms=["chr2"],
        test_chroms=["chr3"],
        train_shuffle=False,
        num_workers=0,
    )

    assert set(train_loader.dataset.intervals.data["chrom"]) == {"chr1"}
    assert set(val_loader.dataset.intervals.data["chrom"]) == {"chr2"}
    assert set(test_loader.dataset.intervals.data["chrom"]) == {"chr3"}


def test_cached_mode_uses_existing_dataset_stack(loader_files):
    fasta_path, intervals_path, _ = loader_files
    loader = build_sequence_dataloader(
        fasta_path,
        Interval(intervals_path),
        batch_size=2,
        mode="cached",
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))
    assert batch.shape == (2, 4, 4)
