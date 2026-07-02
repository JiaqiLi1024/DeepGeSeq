"""Minimal sequence-design APIs for differentiable DGS models.

The routines in this module optimize short regulatory DNA candidates against a
caller-provided model objective. They are intended for in silico exploration and
model debugging, not for asserting biological function.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..Data.Sequence import one_hot_decode, one_hot_encode


ObjectiveFn = Callable[[Any], torch.Tensor]
SequenceLike = Union[str, np.ndarray, torch.Tensor]


@dataclass
class SequenceDesignResult:
    """Container returned by sequence-design routines."""

    sequence: str
    one_hot: torch.Tensor
    score: float
    history: List[Dict[str, Any]]


def _first_tensor_output(output: Any) -> torch.Tensor:
    """Extract the primary tensor from common model output conventions."""
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Model output tuple/list is empty.")
        output = output[0]
    if not isinstance(output, torch.Tensor):
        output = torch.as_tensor(output)
    return output


def _default_objective(
    output: Any,
    target: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
    output_index: Optional[Union[int, Sequence[int]]] = None,
) -> torch.Tensor:
    """Score model outputs by maximizing selected outputs or matching targets."""
    tensor = _first_tensor_output(output)
    if tensor.ndim == 0:
        selected = tensor.reshape(1, 1)
    else:
        selected = tensor.reshape(tensor.shape[0], -1)

    if output_index is not None:
        selected = selected[:, output_index]
        if selected.ndim == 1:
            selected = selected.unsqueeze(1)

    if target is None:
        return selected.mean()

    target_t = torch.as_tensor(target, dtype=selected.dtype, device=selected.device)
    if target_t.numel() == 1:
        target_t = target_t.reshape(1, 1).expand_as(selected)
    else:
        target_t = target_t.reshape_as(selected)
    return -F.mse_loss(selected, target_t, reduction="mean")


def _objective_score(
    output: Any,
    objective_fn: Optional[ObjectiveFn],
    target: Optional[Union[float, Sequence[float], torch.Tensor]],
    output_index: Optional[Union[int, Sequence[int]]],
) -> torch.Tensor:
    """Return a scalar differentiable objective score."""
    score = (
        objective_fn(output)
        if objective_fn is not None
        else _default_objective(output, target=target, output_index=output_index)
    )
    if not isinstance(score, torch.Tensor):
        score = torch.as_tensor(score)
    return score.mean()


def _sequence_to_ncl(sequence: SequenceLike, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert one sequence to ``(1, 4, length)`` float tensor format."""
    if isinstance(sequence, str):
        tensor = torch.as_tensor(one_hot_encode(sequence), dtype=torch.float32)
    elif isinstance(sequence, torch.Tensor):
        tensor = sequence.detach().clone().to(dtype=torch.float32)
    else:
        tensor = torch.as_tensor(sequence, dtype=torch.float32)

    if tensor.ndim == 2:
        if tensor.shape[0] == 4 and tensor.shape[1] != 4:
            tensor = tensor.unsqueeze(0)
        elif tensor.shape[-1] == 4:
            tensor = tensor.transpose(0, 1).unsqueeze(0)
        elif tensor.shape[0] == 4:
            tensor = tensor.unsqueeze(0)
        else:
            raise ValueError("One-hot sequence must have shape (L, 4) or (4, L).")
    elif tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise ValueError("Sequence design currently optimizes one sequence at a time.")
        if tensor.shape[1] == 4:
            pass
        elif tensor.shape[2] == 4:
            tensor = tensor.transpose(1, 2)
        else:
            raise ValueError("One-hot batch must have shape (1, 4, L) or (1, L, 4).")
    else:
        raise ValueError("initial_sequence must be a DNA string or one-hot array/tensor.")

    return tensor.to(device=device) if device is not None else tensor


def _hard_one_hot(probs: torch.Tensor) -> torch.Tensor:
    """Discretize ``(N, 4, L)`` probabilities with channel-wise argmax."""
    indices = probs.argmax(dim=1)
    return F.one_hot(indices, num_classes=4).permute(0, 2, 1).to(dtype=probs.dtype)


def _ncl_to_sequence(one_hot: torch.Tensor) -> str:
    """Decode a single ``(1, 4, L)`` one-hot tensor to DNA sequence."""
    array = one_hot.detach().cpu()[0].transpose(0, 1).numpy()
    decoded = one_hot_decode(array, include_n=False)
    return str(decoded)


def _initial_logits(one_hot: torch.Tensor) -> torch.Tensor:
    """Initialize optimizable logits from a hard or sparse one-hot sequence."""
    logits = torch.zeros_like(one_hot)
    logits = torch.where(one_hot > 0, torch.full_like(logits, 2.0), logits)
    return logits


def _fixed_mask_to_ncl(
    fixed_mask: Optional[Union[Sequence[bool], torch.Tensor]],
    length: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Normalize a per-position fixed mask to ``(1, 1, L)`` boolean format."""
    if fixed_mask is None:
        return None
    mask = torch.as_tensor(fixed_mask, dtype=torch.bool, device=device)
    if mask.ndim != 1 or mask.numel() != length:
        raise ValueError("fixed_mask must be a one-dimensional boolean mask with sequence length.")
    return mask.reshape(1, 1, length)


def gradient_ascent_sequence_design(
    model: torch.nn.Module,
    initial_sequence: SequenceLike,
    target: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
    objective_fn: Optional[ObjectiveFn] = None,
    output_index: Optional[Union[int, Sequence[int]]] = None,
    steps: int = 100,
    lr: float = 0.1,
    temperature: float = 1.0,
    fixed_mask: Optional[Union[Sequence[bool], torch.Tensor]] = None,
    device: Optional[Union[str, torch.device]] = None,
    seed: Optional[int] = None,
) -> SequenceDesignResult:
    """Design a sequence by gradient ascent over soft one-hot logits.

    Args:
        model: PyTorch model accepting inputs shaped ``(batch, 4, length)``.
        initial_sequence: DNA string or one-hot sequence used as initialization.
        target: Optional target output. If provided, the objective maximizes
            negative mean-squared error to the target.
        objective_fn: Optional callable taking raw model output and returning a
            differentiable score. This overrides ``target`` and ``output_index``.
        output_index: Optional flattened output index or indices to maximize.
        steps: Number of optimization steps.
        lr: Adam learning rate for sequence logits.
        temperature: Softmax temperature applied to logits.
        fixed_mask: Optional boolean mask where ``True`` keeps a position fixed.
        device: Optional device for model inputs and optimization state.
        seed: Optional PyTorch RNG seed.

    Returns:
        ``SequenceDesignResult`` with the final discrete sequence, final one-hot
        tensor in ``(1, 4, L)`` format, scalar objective score, and per-step log.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if seed is not None:
        torch.manual_seed(seed)

    resolved_device = torch.device(device) if device is not None else next(model.parameters(), torch.empty(0)).device
    initial_one_hot = _sequence_to_ncl(initial_sequence, device=resolved_device)
    mask = _fixed_mask_to_ncl(fixed_mask, initial_one_hot.shape[-1], resolved_device)
    logits = _initial_logits(initial_one_hot).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=lr)
    history: List[Dict[str, Any]] = []
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            initial_score = _objective_score(
                model(initial_one_hot),
                objective_fn,
                target,
                output_index,
            )
            history.append({"step": 0, "score": float(initial_score.detach().cpu())})

        for step in range(1, steps + 1):
            optimizer.zero_grad()
            probs = F.softmax(logits / temperature, dim=1)
            if mask is not None:
                probs = torch.where(mask, initial_one_hot, probs)
            score = _objective_score(model(probs), objective_fn, target, output_index)
            (-score).backward()
            optimizer.step()
            history.append({"step": step, "score": float(score.detach().cpu())})

        with torch.no_grad():
            probs = F.softmax(logits / temperature, dim=1)
            final_one_hot = _hard_one_hot(probs)
            if mask is not None:
                final_one_hot = torch.where(mask, initial_one_hot, final_one_hot)
            final_score = _objective_score(
                model(final_one_hot),
                objective_fn,
                target,
                output_index,
            )
    finally:
        model.train(was_training)

    return SequenceDesignResult(
        sequence=_ncl_to_sequence(final_one_hot),
        one_hot=final_one_hot.detach().cpu(),
        score=float(final_score.detach().cpu()),
        history=history,
    )


def greedy_ism_sequence_design(
    model: torch.nn.Module,
    initial_sequence: SequenceLike,
    target: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
    objective_fn: Optional[ObjectiveFn] = None,
    output_index: Optional[Union[int, Sequence[int]]] = None,
    mutable_positions: Optional[Sequence[int]] = None,
    max_steps: Optional[int] = None,
    min_delta: float = 1e-6,
    device: Optional[Union[str, torch.device]] = None,
) -> SequenceDesignResult:
    """Greedily optimize a discrete sequence with in silico mutagenesis.

    At each step, the function evaluates all single-base substitutions at the
    mutable positions and applies the best improving mutation.
    """
    resolved_device = torch.device(device) if device is not None else next(model.parameters(), torch.empty(0)).device
    current = _hard_one_hot(_sequence_to_ncl(initial_sequence, device=resolved_device))
    length = current.shape[-1]
    positions = list(range(length)) if mutable_positions is None else [int(pos) for pos in mutable_positions]
    if any(pos < 0 or pos >= length for pos in positions):
        raise ValueError("mutable_positions contains positions outside the sequence length.")
    if max_steps is None:
        max_steps = len(positions)
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative.")

    alphabet = "ACGT"
    history: List[Dict[str, Any]] = []
    was_training = model.training
    model.eval()

    def score_sequence(sequence_tensor: torch.Tensor) -> float:
        with torch.no_grad():
            score = _objective_score(
                model(sequence_tensor),
                objective_fn,
                target,
                output_index,
            )
        return float(score.detach().cpu())

    try:
        current_score = score_sequence(current)
        history.append({"step": 0, "score": current_score, "sequence": _ncl_to_sequence(current)})

        for step in range(1, max_steps + 1):
            best_score = current_score
            best_candidate = current
            best_position: Optional[int] = None
            best_base: Optional[str] = None

            for position in positions:
                current_base = int(current[0, :, position].argmax().item())
                for base_idx, base in enumerate(alphabet):
                    if base_idx == current_base:
                        continue
                    candidate = current.clone()
                    candidate[0, :, position] = 0
                    candidate[0, base_idx, position] = 1
                    candidate_score = score_sequence(candidate)
                    if candidate_score > best_score + min_delta:
                        best_score = candidate_score
                        best_candidate = candidate
                        best_position = position
                        best_base = base

            if best_position is None:
                break

            current = best_candidate
            current_score = best_score
            history.append(
                {
                    "step": step,
                    "score": current_score,
                    "position": best_position,
                    "base": best_base,
                    "sequence": _ncl_to_sequence(current),
                }
            )
    finally:
        model.train(was_training)

    return SequenceDesignResult(
        sequence=_ncl_to_sequence(current),
        one_hot=current.detach().cpu(),
        score=current_score,
        history=history,
    )


__all__ = [
    "SequenceDesignResult",
    "gradient_ascent_sequence_design",
    "greedy_ism_sequence_design",
]
