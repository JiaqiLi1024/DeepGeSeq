"""Unit tests for configuration normalization."""

import unittest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from DGS.Config import (
    ConfigError,
    ConfigManager,
    get_config_schema_reference,
    normalize_config,
    validate_config,
)


class TestConfigNormalization(unittest.TestCase):
    """Test cases for config normalization."""
    def test_legacy_optimizer_and_criterion_are_normalized(self):
        """Test legacy optimizer and criterion are normalized."""
        config = {
            "train": {
                "optimizer": {
                    "type": "Adam",
                    "lr": 1e-3,
                    "weight_decay": 0.1,
                },
                "criterion": {
                    "type": "MSELoss",
                },
            }
        }

        normalized, notes = normalize_config(config)
        self.assertIn("params", normalized["train"]["optimizer"])
        self.assertEqual(normalized["train"]["optimizer"]["params"]["lr"], 1e-3)
        self.assertEqual(normalized["train"]["optimizer"]["params"]["weight_decay"], 0.1)
        self.assertIn("params", normalized["train"]["criterion"])
        self.assertEqual(normalized["train"]["criterion"]["params"], {})
        self.assertTrue(notes)

    def test_normalization_is_idempotent(self):
        """Test normalization is idempotent."""
        config = {
            "train": {
                "optimizer": {"type": "Adam", "params": {"lr": 1e-3}},
                "criterion": {"type": "MSELoss", "params": {}},
            }
        }
        normalized_once, notes_once = normalize_config(config)
        normalized_twice, notes_twice = normalize_config(normalized_once)
        self.assertEqual(normalized_once, normalized_twice)
        self.assertEqual(notes_once, [])
        self.assertEqual(notes_twice, [])

    def test_config_manager_load_applies_normalization(self):
        """Test config manager load applies normalization."""
        config_manager = ConfigManager()
        normalized = config_manager.load_config(
            {
                "train": {
                    "optimizer": {"type": "Adam", "lr": 0.002},
                    "criterion": {"type": "MSELoss"},
                }
            }
        )
        self.assertEqual(normalized["train"]["optimizer"]["params"]["lr"], 0.002)
        self.assertIn("params", normalized["train"]["criterion"])
        self.assertTrue(config_manager.get_compat_notes())

    def test_schema_reference_exposes_recommended_sections(self):
        """Schema reference documents the public runtime sections."""
        schema = get_config_schema_reference()
        self.assertIn("data", schema)
        self.assertIn("model", schema)
        self.assertIn("train", schema)
        self.assertIn("loader_mode", schema["data"]["fields"])
        self.assertIn("predict", schema)

    def test_validate_config_reports_missing_required_fields(self):
        """Config validation raises user-facing errors before runtime work."""
        config = {
            "modes": ["predict"],
            "data": {"genome_path": "genome.fa"},
            "model": {"type": "CNN", "args": {"output_size": 1}},
            "predict": {},
        }

        with self.assertRaisesRegex(ConfigError, "predict.vcf_path"):
            validate_config(config)

    def test_validate_config_accepts_current_train_schema(self):
        """Current nested optimizer/loss schema validates with existing files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            genome_path = tmp_path / "genome.fa"
            intervals_path = tmp_path / "regions.bed"
            target_path = tmp_path / "targets.bed"
            for path in (genome_path, intervals_path, target_path):
                path.write_text("")

            config = {
                "modes": ["train"],
                "data": {
                    "genome_path": str(genome_path),
                    "intervals_path": str(intervals_path),
                    "target_tasks": [
                        {
                            "task_name": "binding",
                            "file_path": str(target_path),
                            "file_type": "bed",
                        }
                    ],
                    "loader_mode": "streaming",
                },
                "model": {"type": "CNN", "args": {"output_size": 1}},
                "train": {
                    "optimizer": {"type": "Adam", "params": {"lr": 1e-3}},
                    "criterion": {"type": "MSELoss", "params": {}},
                },
            }

            validate_config(config, check_files=True)


if __name__ == "__main__":
    unittest.main()
