"""Tests for CLI data preprocessing entry points."""

from pathlib import Path

import pandas as pd
import pytest

try:
    import pysam
except ImportError:  # pragma: no cover - optional dependency
    pysam = None

from DGS.Cli import preprocess_data_for_train
from DGS.Data.Loader import StreamingGenomicDataset


pytestmark = pytest.mark.skipif(pysam is None, reason="pysam is required for CLI data tests")


def _write_files(tmp_path: Path):
    fasta_path = tmp_path / "mini.fa"
    fasta_path.write_text(
        ">chr1\nACGTACGTACGTACGT\n"
        ">chr2\nTTTTCCCCAAAAGGGG\n"
        ">chr3\nAAAACCCCGGGGTTTT\n"
    )
    pysam.faidx(str(fasta_path))

    intervals = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2", "chr2", "chr3", "chr3"],
            "start": [0, 4, 0, 4, 0, 4],
            "end": [4, 8, 4, 8, 4, 8],
            "name": [f"region_{idx}" for idx in range(6)],
            "score": [0] * 6,
            "strand": ["+"] * 6,
        }
    )
    intervals_path = tmp_path / "regions.bed"
    intervals.to_csv(intervals_path, sep="\t", header=False, index=False)

    targets = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2", "chr3"],
            "start": [0, 0, 0],
            "end": [10, 10, 10],
            "name": [1, 1, 1],
        }
    )
    targets_path = tmp_path / "targets.bed"
    targets.to_csv(targets_path, sep="\t", header=False, index=False)

    tasks = [
        {
            "task_name": "binding",
            "file_path": str(targets_path),
            "file_type": "bed",
            "target_column": "name",
        }
    ]
    return fasta_path, intervals_path, tasks


def test_preprocess_data_for_train_uses_high_level_streaming_loader(tmp_path):
    """CLI training data path should use the first-class DataLoader builder."""
    fasta_path, intervals_path, tasks = _write_files(tmp_path)

    train_loader, val_loader, test_loader = preprocess_data_for_train(
        genome_path=fasta_path,
        target_tasks=tasks,
        intervals_path=intervals_path,
        train_test_split="chromosome_split",
        val_chroms=["chr2"],
        test_chroms=["chr3"],
        batch_size=2,
        loader_mode="streaming",
        dataloader_config={"num_workers": 0, "pin_memory": False},
    )

    assert isinstance(train_loader.dataset, StreamingGenomicDataset)
    assert isinstance(val_loader.dataset, StreamingGenomicDataset)
    assert isinstance(test_loader.dataset, StreamingGenomicDataset)

    seqs, labels = next(iter(train_loader))
    assert seqs.shape == (2, 4, 4)
    assert labels.shape == (2, 1)
