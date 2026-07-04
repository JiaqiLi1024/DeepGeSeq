"""Tests for sequence-design APIs."""

import torch

from DGS.DL import (
    SequenceDesignResult,
    gradient_ascent_sequence_design,
    greedy_ism_sequence_design,
)


class PositionalPreferenceModel(torch.nn.Module):
    """Tiny deterministic model with base preferences at fixed positions."""

    def forward(self, x):
        # x is expected in (batch, 4, length): prefer A at pos 1 and G at pos 3.
        return (x[:, 0, 1] + x[:, 2, 3]).unsqueeze(1)


class TwoPositionModel(torch.nn.Module):
    """Prefer A at pos 0 and C at pos 1."""

    def forward(self, x):
        return (x[:, 0, 0] + x[:, 1, 1]).unsqueeze(1)


class AAtFirstPositionModel(torch.nn.Module):
    """Return whether the first position is A."""

    def forward(self, x):
        return x[:, 0, 0].unsqueeze(1)


def test_gradient_ascent_sequence_design_improves_sequence():
    model = PositionalPreferenceModel()

    result = gradient_ascent_sequence_design(
        model,
        "TTTT",
        steps=80,
        lr=0.4,
        seed=3,
    )

    assert isinstance(result, SequenceDesignResult)
    assert result.sequence[1] == "A"
    assert result.sequence[3] == "G"
    assert result.score > result.history[0]["score"]
    assert result.one_hot.shape == (1, 4, 4)


def test_gradient_ascent_sequence_design_respects_fixed_mask():
    model = TwoPositionModel()

    result = gradient_ascent_sequence_design(
        model,
        "TTTT",
        steps=60,
        lr=0.4,
        fixed_mask=[True, False, False, False],
    )

    assert result.sequence[0] == "T"
    assert result.sequence[1] == "C"


def test_greedy_ism_sequence_design_applies_best_single_base_edits():
    model = TwoPositionModel()

    result = greedy_ism_sequence_design(
        model,
        "TTTT",
        max_steps=2,
    )

    assert result.sequence[:2] == "AC"
    assert result.score == 2.0
    assert len(result.history) == 3


def test_greedy_ism_sequence_design_can_match_target_output():
    model = AAtFirstPositionModel()

    result = greedy_ism_sequence_design(
        model,
        "A",
        target=0.0,
        max_steps=1,
    )

    assert result.sequence != "A"
    assert result.score == 0.0


def test_greedy_ism_sequence_design_validates_mutable_positions():
    model = TwoPositionModel()

    try:
        greedy_ism_sequence_design(model, "TTTT", mutable_positions=[99])
    except ValueError as exc:
        assert "outside the sequence length" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected mutable position validation to fail")
