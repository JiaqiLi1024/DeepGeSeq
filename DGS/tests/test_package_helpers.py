"""Tests for top-level DGS compatibility helpers."""

import logging

import DGS


def test_initialize_logger_writes_legacy_tutorial_log(tmp_path):
    log_path = tmp_path / "Log" / "tutorial.log"

    logger = DGS.initialize_logger(log_path, verbosity=1)
    logger.info("tutorial logger ready")

    assert logger is logging.getLogger("DGS")
    assert log_path.exists()
    assert "tutorial logger ready" in log_path.read_text()
