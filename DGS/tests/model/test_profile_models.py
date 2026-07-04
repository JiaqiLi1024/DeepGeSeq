"""Tests for profile-model exports and adapters."""

import importlib.util

import numpy as np
import pytest
import torch

from DGS.Model import (
    Borzoi,
    ChromBPNet,
    Enformer,
    KerasProfileAdapter,
    get_model_zoo,
    load_keras_profile_model,
)


def test_chrombpnet_returns_profile_and_count_outputs():
    """ChromBPNet should follow the DGS profile/count output convention."""
    model = ChromBPNet(
        input_len=128,
        output_len=64,
        n_filters=8,
        n_layers=2,
        n_outputs=2,
    )
    x = torch.zeros(3, 128, 4)
    x[:, :, 0] = 1.0

    profile, counts = model(x)

    assert profile.shape == (3, 2, 64)
    assert counts.shape == (3, 2)


class ConstantBias(torch.nn.Module):
    """Tiny bias model returning profile and count offsets."""

    def forward(self, x):
        batch = x.shape[0]
        length = x.shape[-1]
        return torch.ones(batch, 1, length), torch.zeros(batch, 1)


def test_chrombpnet_can_add_frozen_bias_model():
    """Bias-factorized ChromBPNet should combine TF and bias outputs."""
    model = ChromBPNet(
        output_len=16,
        n_filters=4,
        n_layers=1,
        n_outputs=1,
        bias_model=ConstantBias(),
    )

    profile, counts = model(torch.zeros(2, 4, 64))

    assert profile.shape == (2, 1, 16)
    assert counts.shape == (2, 1)
    assert all(not param.requires_grad for param in model.bias_model.parameters())


def test_borzoi_returns_selected_head_and_embeddings():
    """Borzoi should expose profile heads and optional trunk embeddings."""
    model = Borzoi(
        input_len=256,
        output_heads={"human": 3, "mouse": 2},
        target_length=32,
        stem_filters=16,
        filters_init=24,
        filters_end=32,
        res_blocks=2,
        transformer_depth=1,
        transformer_heads=4,
        upsample_blocks=2,
        final_filters=32,
    )
    x = torch.zeros(2, 256, 4)
    x[:, :, 1] = 1.0

    profile, embeddings = model(x, head="human", return_embeddings=True)
    all_heads = model(x, head=None)

    assert profile.shape == (2, 3, 32)
    assert embeddings.shape == (2, 32, 32)
    assert set(all_heads) == {"human", "mouse"}
    assert all_heads["mouse"].shape == (2, 2, 32)


class DummyKerasModel:
    """Keras-like object with a predict method for adapter tests."""

    def predict(self, sequence_nlc, verbose=0):
        batch, length, _ = sequence_nlc.shape
        profile = np.ones((batch, length), dtype=np.float32)
        counts = np.full((batch, 1), 2.0, dtype=np.float32)
        return [profile, counts]


class DummyTrackLastKerasModel:
    """Keras-like model returning track-last profile predictions."""

    def predict(self, sequence_nlc, verbose=0):
        batch, length, _ = sequence_nlc.shape
        return np.ones((batch, length, 5), dtype=np.float32)


def test_keras_profile_adapter_normalizes_outputs_without_tensorflow():
    """Adapter should wrap loaded Keras-like objects without importing TensorFlow."""
    adapter = KerasProfileAdapter(DummyKerasModel())

    profile, counts = adapter(torch.zeros(2, 4, 10))

    assert profile.shape == (2, 1, 10)
    assert counts.shape == (2, 1)
    torch.testing.assert_close(counts, torch.full((2, 1), 2.0))


def test_keras_profile_adapter_uses_profile_channels_for_track_last_outputs():
    """Borzoi/Enformer-style Keras outputs may be shaped as (N, L, tracks)."""
    adapter = KerasProfileAdapter(DummyTrackLastKerasModel(), profile_channels=5)

    profile = adapter(torch.zeros(2, 16, 4))

    assert profile.shape == (2, 5, 16)


def test_keras_profile_loader_reports_missing_tensorflow_for_paths(tmp_path):
    """Path loading should be lazy and fail with a useful optional-dependency error."""
    if importlib.util.find_spec("tensorflow") is not None:
        pytest.skip("tensorflow is installed in this environment")

    with pytest.raises(RuntimeError, match="TensorFlow"):
        load_keras_profile_model(tmp_path / "missing.h5")


def test_enformer_export_accepts_tensor_inputs():
    """The Enformer export should run without requiring external einops."""
    model = Enformer(
        dim=384,
        depth=1,
        heads=8,
        output_heads={"test": 2},
        target_length=-1,
        dropout_rate=0.0,
        attn_dropout=0.0,
        pos_dropout=0.0,
    )
    model.eval()
    x = torch.zeros(1, 256, 4)
    x[:, :, 2] = 1.0

    with torch.no_grad():
        output = model(x, head="test")

    assert output.shape == (1, 2, 2)


def test_model_zoo_lists_profile_model_support():
    """Model support metadata should list the new profile-model exports."""
    zoo = get_model_zoo()

    for name in ["ChromBPNet", "Enformer", "Borzoi", "KerasProfileAdapter"]:
        assert name in zoo
        assert zoo[name]["class_name"] == name
        assert "workflow_status" in zoo[name]
