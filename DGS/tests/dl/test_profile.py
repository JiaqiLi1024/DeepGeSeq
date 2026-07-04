"""Tests for profile-model losses, metrics, and writers."""

import numpy as np
import pandas as pd
import pytest
import torch

from DGS.DL import (
    ProfileCountLoss,
    calculate_profile_metrics,
    count_targets_from_profile,
    ensure_profile_ncl,
    profile_multinomial_nll_loss,
    profile_poisson_loss,
    save_profile_predictions_h5,
    save_profile_predictions_npz,
    write_profile_predictions_bigwig,
)


def test_ensure_profile_ncl_accepts_channel_last_inputs():
    channel_last = torch.zeros(2, 5, 1)
    converted = ensure_profile_ncl(channel_last)

    assert converted.shape == (2, 1, 5)


def test_profile_multinomial_nll_loss_matches_manual_value():
    logits = torch.tensor([[[0.0, 1.0, 0.0]]])
    targets = torch.tensor([[[0.0, 2.0, 1.0]]])

    loss = profile_multinomial_nll_loss(logits, targets, reduction="none")
    expected = -(targets.reshape(1, -1) * torch.log_softmax(logits.reshape(1, -1), dim=1)).sum(dim=1)

    torch.testing.assert_close(loss, expected)


def test_profile_multinomial_nll_loss_rejects_negative_targets():
    with pytest.raises(ValueError, match="non-negative"):
        profile_multinomial_nll_loss(
            torch.zeros(1, 1, 3),
            torch.tensor([[[1.0, -1.0, 0.0]]]),
        )


def test_profile_count_loss_derives_count_targets_and_backpropagates():
    profile_logits = torch.zeros(2, 1, 4, requires_grad=True)
    count_pred = torch.zeros(2, 1, requires_grad=True)
    profile_targets = torch.tensor(
        [
            [[0.0, 1.0, 2.0, 0.0]],
            [[1.0, 0.0, 0.0, 3.0]],
        ]
    )
    criterion = ProfileCountLoss(profile_weight=1.0, count_weight=0.5)

    loss = criterion((profile_logits, count_pred), profile_targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert profile_logits.grad is not None
    assert count_pred.grad is not None


def test_profile_poisson_loss_uses_torch_poisson_nll():
    prediction = torch.log(torch.tensor([[2.0], [3.0]]))
    targets = torch.tensor([[2.0], [4.0]])

    loss = profile_poisson_loss(prediction, targets, log_input=True)

    assert torch.isfinite(loss)


def test_count_targets_from_profile_can_return_raw_or_log_counts():
    profile = torch.tensor([[[1.0, 2.0, 3.0]]])

    raw = count_targets_from_profile(profile, log1p=False)
    logged = count_targets_from_profile(profile, log1p=True)

    torch.testing.assert_close(raw, torch.tensor([[6.0]]))
    torch.testing.assert_close(logged, torch.log1p(torch.tensor([[6.0]])))


def test_calculate_profile_metrics_returns_expected_columns():
    targets = np.array([[[1.0, 2.0, 3.0]], [[0.0, 1.0, 0.0]]], dtype=np.float32)
    predictions = targets + 0.5

    metrics = calculate_profile_metrics(targets, predictions)

    assert set(
        [
            "profile_mse",
            "profile_mae",
            "profile_pearson_r_mean",
            "count_pearson_r",
            "count_mse",
        ]
    ).issubset(metrics.columns)
    assert metrics.loc[0, "profile_mse"] == pytest.approx(0.25)


def test_save_profile_predictions_npz_and_h5(tmp_path):
    predictions = np.arange(8, dtype=np.float32).reshape(2, 1, 4)
    intervals = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 4],
            "end": [4, 8],
        }
    )

    npz_path = save_profile_predictions_npz(
        tmp_path / "profiles.npz",
        predictions,
        intervals=intervals,
        track_names=["signal"],
    )
    loaded = np.load(npz_path)
    assert loaded["predictions"].shape == (2, 1, 4)
    assert loaded["track_names"].tolist() == ["signal"]

    h5_path = save_profile_predictions_h5(
        tmp_path / "profiles.h5",
        predictions,
        intervals=intervals,
        track_names=["signal"],
    )
    h5py = pytest.importorskip("h5py")
    with h5py.File(h5_path, "r") as handle:
        assert handle["predictions"].shape == (2, 1, 4)
        assert handle.attrs["shape_convention"] == "NCL"


def test_write_profile_predictions_bigwig_roundtrip(tmp_path):
    pyBigWig = pytest.importorskip("pyBigWig")
    predictions = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
    intervals = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [4]})

    paths = write_profile_predictions_bigwig(
        tmp_path / "pred",
        predictions,
        intervals,
        chrom_sizes={"chr1": 10},
        track_names=["signal"],
    )

    assert len(paths) == 1
    bw = pyBigWig.open(paths[0])
    try:
        values = bw.values("chr1", 0, 4, numpy=True)
    finally:
        bw.close()
    np.testing.assert_allclose(values, [1.0, 2.0, 3.0, 4.0])
