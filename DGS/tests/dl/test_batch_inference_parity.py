"""Unit tests for batch inference parity."""

import unittest
from unittest import mock
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from DGS.DL import Predict as predict_module
from DGS.DL import Explain as explain_module

HAS_CAPTUM = importlib.util.find_spec("captum") is not None


class _ToyModel(torch.nn.Module):
    """Helper class used by local tests."""
    def forward(self, x):
        """Compute forward outputs for `_ToyModel`."""
        task1 = x[:, 0, :].sum(dim=1)
        task2 = x[:, 1, :].sum(dim=1)
        return torch.stack([task1, task2], dim=1)


class _ChannelFirstToyModel(torch.nn.Module):
    """Helper model that only accepts channel-first inputs."""
    def forward(self, x):
        """Compute outputs and reject channel-last sequence tensors."""
        if x.ndim != 3 or x.shape[1] != 4:
            raise RuntimeError("expected channel-first NCL input")
        task1 = x[:, 0, :].sum(dim=1)
        task2 = x[:, 1, :].sum(dim=1)
        return torch.stack([task1, task2], dim=1)


class _VariantPairDataset:
    """Helper class used by local tests."""
    def __init__(self):
        """Initialize `_VariantPairDataset`."""
        self._ref = [
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
            np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
            np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=np.float32),
        ]
        self._alt = [
            np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
            np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.float32),
            np.array([[0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32),
        ]

    def __len__(self):
        """Return dataset length for flattened indexing."""
        return len(self._ref)

    def __getitem__(self, idx):
        """Return one sample for the given flattened index."""
        return self._ref[idx], self._alt[idx]


class _SeqDataset:
    """Helper class used by local tests."""
    def __init__(self):
        """Initialize `_SeqDataset`."""
        self._data = [
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
            np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
            np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=np.float32),
        ]

    def __len__(self):
        """Return dataset length for flattened indexing."""
        return len(self._data)

    def __getitem__(self, idx):
        """Return one sample for the given flattened index."""
        return self._data[idx]


class TestBatchInferenceParity(unittest.TestCase):
    """Test cases for batch inference parity."""
    def test_predict_batch_path_matches_legacy(self):
        """Test predict batch path matches legacy."""
        model = _ToyModel()
        dataset = _VariantPairDataset()

        legacy = predict_module.vep_centred_on_ds(
            model,
            dataset,
            metric_func="diff",
            mean_by_tasks=True,
            device=torch.device("cpu"),
            batch_size=None,
        )
        batched = predict_module.vep_centred_on_ds(
            model,
            dataset,
            metric_func="diff",
            mean_by_tasks=True,
            device=torch.device("cpu"),
            batch_size=2,
        )
        batched_with_loader_cfg = predict_module.vep_centred_on_ds(
            model,
            dataset,
            metric_func="diff",
            mean_by_tasks=True,
            device=torch.device("cpu"),
            batch_size=2,
            dataloader_config={"num_workers": 0, "pin_memory": False},
        )
        np.testing.assert_allclose(legacy, batched)
        np.testing.assert_allclose(legacy, batched_with_loader_cfg)

    def test_predict_layout_fallback_supports_channel_first_models(self):
        """Test VEP paths can retry channel-last batches as channel-first."""
        model = _ChannelFirstToyModel()
        dataset = _VariantPairDataset()

        legacy = predict_module.vep_centred_on_ds(
            model,
            dataset,
            metric_func="diff",
            mean_by_tasks=True,
            device=torch.device("cpu"),
            batch_size=None,
        )
        batched = predict_module.vep_centred_on_ds(
            model,
            dataset,
            metric_func="diff",
            mean_by_tasks=True,
            device=torch.device("cpu"),
            batch_size=2,
        )

        np.testing.assert_allclose(legacy, batched)

    def test_explain_batch_path_matches_legacy(self):
        """Test explain batch path matches legacy."""
        dataset = _SeqDataset()
        model = _ToyModel()

        with mock.patch.object(explain_module, "_TANGERMEME_IMPORT_ERROR", None), \
             mock.patch.object(explain_module, "deep_lift_shap", side_effect=lambda _m, x, target=0: x * 0.25):
            legacy_x, legacy_attr = explain_module.calculate_shap_on_ds(
                model,
                dataset,
                target=0,
                device=torch.device("cpu"),
                batch_size=None,
            )
            batched_x, batched_attr = explain_module.calculate_shap_on_ds(
                model,
                dataset,
                target=0,
                device=torch.device("cpu"),
                batch_size=2,
            )

        np.testing.assert_allclose(legacy_x, batched_x)
        np.testing.assert_allclose(legacy_attr, batched_attr)

    @unittest.skipUnless(HAS_CAPTUM, "Captum attribution tests require captum")
    def test_captum_explain_batch_path_matches_legacy(self):
        """Test Captum attribution batch path matches sample-wise path."""
        dataset = _SeqDataset()
        model = _ToyModel()

        for method in ["deeplift", "integrated_gradients"]:
            legacy_x, legacy_attr = explain_module.calculate_attributions_on_ds(
                model,
                dataset,
                target=0,
                device=torch.device("cpu"),
                batch_size=None,
                method=method,
                n_steps=8,
            )
            batched_x, batched_attr = explain_module.calculate_attributions_on_ds(
                model,
                dataset,
                target=0,
                device=torch.device("cpu"),
                batch_size=2,
                method=method,
                n_steps=8,
            )

            np.testing.assert_allclose(legacy_x, batched_x)
            np.testing.assert_allclose(legacy_attr, batched_attr, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
