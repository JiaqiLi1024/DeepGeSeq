"""Unit tests for trainer failure accounting and checkpoint loading behavior."""

import tempfile
import unittest
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from DGS.DL.Trainer import Trainer, TrainerState


class _FailingModel(torch.nn.Module):
    """Model that raises on every forward pass."""

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, _x):
        raise RuntimeError("intentional forward failure")


class TestTrainerFailureHandling(unittest.TestCase):
    """Validate robust error handling in train/validate loops."""

    def setUp(self):
        dataset = TensorDataset(torch.randn(4, 2), torch.randn(4, 1))
        self.loader = DataLoader(dataset, batch_size=2, shuffle=False)
        self.device = torch.device("cpu")

    def test_train_epoch_raises_when_all_batches_fail(self):
        """All failed train batches should raise a clear error."""
        model = _FailingModel()
        trainer = Trainer(
            model=model,
            criterion=torch.nn.MSELoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            device=self.device,
        )

        with self.assertRaisesRegex(RuntimeError, "All training batches failed"):
            trainer.train_epoch(self.loader, epoch=0)

    def test_validate_raises_when_all_batches_fail(self):
        """All failed validation batches should raise a clear error."""
        model = _FailingModel()
        trainer = Trainer(
            model=model,
            criterion=torch.nn.MSELoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            device=self.device,
        )

        with self.assertRaisesRegex(RuntimeError, "All validation batches failed"):
            trainer.validate(self.loader)

    def test_load_checkpoint_can_skip_optimizer_state(self):
        """Evaluation workflows can load model weights without optimizer restore."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "checkpoint.pt"

            model_src = torch.nn.Linear(2, 1)
            trainer_src = Trainer(
                model=model_src,
                criterion=torch.nn.MSELoss(),
                optimizer=torch.optim.Adam(model_src.parameters(), lr=1e-3),
                device=self.device,
                checkpoint_dir=tmp_dir,
            )
            trainer_src.save_checkpoint(ckpt_path)

            model_dst = torch.nn.Linear(2, 1)
            trainer_dst = Trainer(
                model=model_dst,
                criterion=torch.nn.MSELoss(),
                optimizer=torch.optim.SGD(model_dst.parameters(), lr=1e-2),
                device=self.device,
                checkpoint_dir=tmp_dir,
            )
            trainer_dst.load_checkpoint(ckpt_path, load_optimizer=False)

            for src_param, dst_param in zip(model_src.parameters(), trainer_dst.model.parameters()):
                self.assertTrue(torch.allclose(src_param, dst_param))

    def test_load_checkpoint_restores_optimizer_state_for_resume(self):
        """Resume workflows should restore both model weights and optimizer state."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "resume.pt"
            data = torch.randn(4, 2)
            target = torch.randn(4, 1)
            loader = DataLoader(TensorDataset(data, target), batch_size=2, shuffle=False)

            model_src = torch.nn.Linear(2, 1)
            trainer_src = Trainer(
                model=model_src,
                criterion=torch.nn.MSELoss(),
                optimizer=torch.optim.Adam(model_src.parameters(), lr=1e-3),
                device=self.device,
                checkpoint_dir=tmp_dir,
            )
            trainer_src.train_epoch(loader, epoch=0)
            trainer_src.save_checkpoint(ckpt_path)

            model_dst = torch.nn.Linear(2, 1)
            trainer_dst = Trainer(
                model=model_dst,
                criterion=torch.nn.MSELoss(),
                optimizer=torch.optim.Adam(model_dst.parameters(), lr=1e-3),
                device=self.device,
                checkpoint_dir=tmp_dir,
            )
            trainer_dst.load_checkpoint(ckpt_path, load_optimizer=True)

            self.assertTrue(trainer_dst.optimizer.state)
            self.assertEqual(trainer_dst.state.global_step, trainer_src.state.global_step)
            for src_param, dst_param in zip(model_src.parameters(), trainer_dst.model.parameters()):
                self.assertTrue(torch.allclose(src_param, dst_param))

    def test_save_checkpoint_uses_plain_dict_format(self):
        """New checkpoints avoid dataclass pickling for PyTorch 2.x loading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "checkpoint.pt"

            model = torch.nn.Linear(2, 1)
            trainer = Trainer(
                model=model,
                criterion=torch.nn.MSELoss(),
                optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
                device=self.device,
                checkpoint_dir=tmp_dir,
            )
            trainer.save_checkpoint(ckpt_path)

            try:
                checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.assertIsInstance(checkpoint, dict)
            self.assertEqual(checkpoint["format_version"], 2)
            self.assertIn("model_state", checkpoint)
            self.assertIsInstance(checkpoint["metrics"], dict)

    def test_load_checkpoint_supports_legacy_trainer_state(self):
        """Legacy dataclass checkpoints still restore for existing users."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "legacy.pt"

            model_src = torch.nn.Linear(2, 1)
            trainer_src = Trainer(
                model=model_src,
                criterion=torch.nn.MSELoss(),
                optimizer=torch.optim.Adam(model_src.parameters(), lr=1e-3),
                device=self.device,
                checkpoint_dir=tmp_dir,
            )
            legacy_state = TrainerState(
                model_state={k: v.cpu() for k, v in model_src.state_dict().items()},
                optimizer_state=trainer_src.optimizer.state_dict(),
                metrics=trainer_src.metrics,
            )
            torch.save(legacy_state, ckpt_path)

            model_dst = torch.nn.Linear(2, 1)
            trainer_dst = Trainer(
                model=model_dst,
                criterion=torch.nn.MSELoss(),
                optimizer=torch.optim.Adam(model_dst.parameters(), lr=1e-3),
                device=self.device,
                checkpoint_dir=tmp_dir,
            )
            trainer_dst.load_checkpoint(ckpt_path)

            for src_param, dst_param in zip(model_src.parameters(), trainer_dst.model.parameters()):
                self.assertTrue(torch.allclose(src_param, dst_param))

    def test_load_checkpoint_supports_plain_model_state_dict(self):
        """Model-only state_dict files are accepted for lightweight restores."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "weights.pt"

            model_src = torch.nn.Linear(2, 1)
            torch.save(model_src.state_dict(), ckpt_path)

            model_dst = torch.nn.Linear(2, 1)
            trainer_dst = Trainer(
                model=model_dst,
                criterion=torch.nn.MSELoss(),
                optimizer=torch.optim.Adam(model_dst.parameters(), lr=1e-3),
                device=self.device,
                checkpoint_dir=tmp_dir,
            )
            trainer_dst.load_checkpoint(ckpt_path, load_optimizer=False)

            for src_param, dst_param in zip(model_src.parameters(), trainer_dst.model.parameters()):
                self.assertTrue(torch.allclose(src_param, dst_param))

    def test_amp_request_is_safe_on_cpu_and_non_blocking_is_recorded(self):
        """PyTorch 2.x CPU runs keep AMP disabled and preserve transfer options."""
        model = torch.nn.Linear(2, 1)
        trainer = Trainer(
            model=model,
            criterion=torch.nn.MSELoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            device=self.device,
            use_amp=True,
            non_blocking=True,
        )

        self.assertFalse(trainer.use_amp)
        self.assertTrue(trainer.non_blocking)
        loss = trainer.train_epoch(self.loader, epoch=0)
        self.assertGreaterEqual(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
