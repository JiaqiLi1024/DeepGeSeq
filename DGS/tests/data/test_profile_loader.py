"""Tests for profile-model DataLoader builders."""

import numpy as np
import pandas as pd
import pytest

try:
    import pyBigWig
except ImportError:  # pragma: no cover - optional dependency
    pyBigWig = None

try:
    import pysam
except ImportError:  # pragma: no cover - optional dependency
    pysam = None

from DGS.Data import (
    StreamingProfileDataset,
    build_profile_dataloader,
    build_profile_dataloaders,
)


pytestmark = pytest.mark.skipif(
    pyBigWig is None or pysam is None,
    reason="pyBigWig and pysam are required for profile loader tests",
)


@pytest.fixture
def profile_files(tmp_path):
    fasta_path = tmp_path / "mini.fa"
    fasta_path.write_text(
        ">chr1\n"
        "ACGTACGTACGTACGT\n"
        ">chr2\n"
        "TTTTCCCCAAAAGGGG\n"
    )
    pysam.faidx(str(fasta_path))

    intervals = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2", "chr2"],
            "start": [0, 4, 0, 4],
            "end": [4, 8, 4, 8],
            "name": ["a", "b", "c", "d"],
            "score": [0, 0, 0, 0],
            "strand": ["+", "-", "+", "+"],
        }
    )

    bw_path = tmp_path / "signal.bw"
    bw = pyBigWig.open(str(bw_path), "w")
    bw.addHeader([("chr1", 16), ("chr2", 16)])
    for chrom, offset in [("chr1", 0.0), ("chr2", 100.0)]:
        starts = list(range(16))
        bw.addEntries(
            [chrom] * 16,
            starts,
            ends=[start + 1 for start in starts],
            values=[offset + float(start) for start in starts],
        )
    bw.close()

    tasks = [
        {
            "task_name": "signal",
            "file_path": str(bw_path),
            "file_type": "bigwig",
            "task_type": "profile",
        }
    ]
    return fasta_path, intervals, tasks


def test_streaming_profile_dataset_reads_bigwig_profiles(profile_files):
    fasta_path, intervals, tasks = profile_files
    dataset = StreamingProfileDataset(
        intervals,
        fasta_path,
        tasks,
        strand_aware=False,
        reverse_negative_strand_targets=False,
    )

    seq, profile = dataset[0]
    assert seq.shape == (4, 4)
    assert profile.shape == (1, 4)
    np.testing.assert_allclose(profile[0], [0.0, 1.0, 2.0, 3.0])
    assert dataset.task_names == ["signal"]


def test_streaming_profile_dataset_reverses_negative_strand_targets(profile_files):
    fasta_path, intervals, tasks = profile_files
    dataset = StreamingProfileDataset(intervals, fasta_path, tasks, strand_aware=True)

    _, profile = dataset[1]
    np.testing.assert_allclose(profile[0], [7.0, 6.0, 5.0, 4.0])


def test_profile_dataset_can_return_count_targets(profile_files):
    fasta_path, intervals, tasks = profile_files
    dataset = StreamingProfileDataset(
        intervals,
        fasta_path,
        tasks,
        return_counts=True,
        counts_log1p=False,
        reverse_negative_strand_targets=False,
    )

    _, (profile, counts) = dataset[0]
    assert profile.shape == (1, 4)
    assert counts.shape == (1,)
    assert counts[0] == pytest.approx(6.0)


def test_build_profile_dataloader_batches_sequences_and_profiles(profile_files):
    fasta_path, intervals, tasks = profile_files
    loader = build_profile_dataloader(
        fasta_path,
        intervals,
        tasks,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    seqs, profiles = next(iter(loader))
    assert seqs.shape == (2, 4, 4)
    assert profiles.shape == (2, 1, 4)
    np.testing.assert_allclose(profiles[0, 0].numpy(), [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(profiles[1, 0].numpy(), [7.0, 6.0, 5.0, 4.0])


def test_build_profile_dataloaders_random_split(profile_files):
    fasta_path, intervals, tasks = profile_files
    train_loader, val_loader, test_loader = build_profile_dataloaders(
        fasta_path,
        intervals,
        tasks,
        batch_size=2,
        split="random",
        test_size=0.25,
        val_size=0.25,
        random_state=7,
        train_shuffle=False,
        num_workers=0,
    )

    lengths = [len(loader.dataset) for loader in (train_loader, val_loader, test_loader)]
    assert lengths == [2, 1, 1]
    seqs, profiles = next(iter(train_loader))
    assert seqs.shape[-1] == 4
    assert profiles.shape[1:] == (1, 4)


def test_profile_loader_rejects_non_bigwig_tasks(profile_files):
    fasta_path, intervals, tasks = profile_files
    bad_tasks = [dict(tasks[0], file_type="bed")]

    with pytest.raises(ValueError, match="Profile targets currently require 'bigwig'"):
        build_profile_dataloader(fasta_path, intervals, bad_tasks)


def test_profile_loader_requires_fixed_width_intervals(profile_files):
    fasta_path, intervals, tasks = profile_files
    intervals = intervals.copy()
    intervals.loc[0, "end"] = 5

    with pytest.raises(ValueError, match="fixed-width intervals"):
        build_profile_dataloader(fasta_path, intervals, tasks)


def test_profile_loader_reports_profile_task_length_mismatch(profile_files):
    fasta_path, intervals, tasks = profile_files
    mixed_resolution_tasks = [
        tasks[0],
        dict(
            tasks[0],
            task_name="signal_binned",
            bin_size=2,
            aggfunc="mean",
        ),
    ]

    with pytest.raises(ValueError, match="same output length"):
        build_profile_dataloader(fasta_path, intervals, mixed_resolution_tasks)
