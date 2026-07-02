"""Profile-model losses, metrics, and output writers.

This module provides the minimal profile/count workflow pieces needed for
BPNet-style sequence models without changing the scalar Trainer/Evaluator API.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ArrayLike = Union[np.ndarray, torch.Tensor]


def _as_tensor(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert arrays or tensors to a floating tensor."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def ensure_profile_ncl(value: ArrayLike, name: str = "profile") -> torch.Tensor:
    """Return profile tensors with shape ``(batch, channels, length)``."""
    tensor = _as_tensor(value)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape (N, C, L) or (N, L, C).")
    if tensor.shape[1] <= 4:
        return tensor
    if tensor.shape[-1] <= 4:
        return tensor.transpose(1, 2)
    raise ValueError(f"{name} must include a channel axis.")


def profile_multinomial_nll_loss(
    logits: ArrayLike,
    targets: ArrayLike,
    reduction: str = "mean",
    normalize_by_counts: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute multinomial negative log-likelihood for profile logits.

    Args:
        logits: Predicted profile logits with shape ``(N, C, L)``.
        targets: Non-negative profile counts/signals with matching shape.
        reduction: ``"none"``, ``"mean"``, or ``"sum"``.
        normalize_by_counts: Divide each sample loss by its total target count.
        eps: Numerical floor used when normalizing by counts.
    """
    logits_t = ensure_profile_ncl(logits, "logits")
    targets_t = ensure_profile_ncl(targets, "targets").to(device=logits_t.device)
    if logits_t.shape != targets_t.shape:
        raise ValueError(f"logits and targets must have matching shapes; got {logits_t.shape} and {targets_t.shape}.")
    if torch.any(targets_t < 0):
        raise ValueError("targets must be non-negative for multinomial profile loss.")

    flat_logits = logits_t.reshape(logits_t.shape[0], -1)
    flat_targets = targets_t.reshape(targets_t.shape[0], -1)
    log_probs = F.log_softmax(flat_logits, dim=1)
    loss = -(flat_targets * log_probs).sum(dim=1)
    if normalize_by_counts:
        loss = loss / flat_targets.sum(dim=1).clamp_min(eps)

    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'.")


def profile_poisson_loss(
    prediction: ArrayLike,
    targets: ArrayLike,
    log_input: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute Poisson loss for profile or count predictions."""
    pred_t = _as_tensor(prediction)
    target_t = _as_tensor(targets).to(device=pred_t.device)
    if pred_t.shape != target_t.shape:
        raise ValueError(f"prediction and targets must have matching shapes; got {pred_t.shape} and {target_t.shape}.")
    return F.poisson_nll_loss(pred_t, target_t, log_input=log_input, full=False, reduction=reduction)


def count_targets_from_profile(profile_targets: ArrayLike, log1p: bool = True) -> torch.Tensor:
    """Derive total count targets from profile targets."""
    targets_t = ensure_profile_ncl(profile_targets, "profile_targets")
    counts = targets_t.sum(dim=(1, 2), keepdim=False).unsqueeze(1)
    return torch.log1p(counts) if log1p else counts


class ProfileCountLoss(nn.Module):
    """Composite BPNet-style profile/count loss.

    The module accepts model outputs as ``profile_logits`` or
    ``(profile_logits, count_prediction)``. Targets may be ``profile_targets`` or
    ``(profile_targets, count_targets)``. If count targets are omitted, they are
    derived from the summed profile target.
    """

    def __init__(
        self,
        profile_weight: float = 1.0,
        count_weight: float = 1.0,
        normalize_profile_by_counts: bool = True,
        count_loss: str = "mse",
        count_log1p_target: bool = True,
    ):
        super().__init__()
        self.profile_weight = float(profile_weight)
        self.count_weight = float(count_weight)
        self.normalize_profile_by_counts = bool(normalize_profile_by_counts)
        self.count_loss = count_loss
        self.count_log1p_target = bool(count_log1p_target)

    def forward(self, output: Any, target: Any) -> torch.Tensor:
        """Return the weighted profile/count loss tensor."""
        if isinstance(output, (tuple, list)):
            profile_logits = output[0]
            count_pred = output[1] if len(output) > 1 else None
        else:
            profile_logits = output
            count_pred = None

        if isinstance(target, (tuple, list)):
            profile_targets = target[0]
            count_targets = target[1] if len(target) > 1 else None
        else:
            profile_targets = target
            count_targets = None

        profile_loss = profile_multinomial_nll_loss(
            profile_logits,
            profile_targets,
            reduction="mean",
            normalize_by_counts=self.normalize_profile_by_counts,
        )
        total = self.profile_weight * profile_loss

        if count_pred is not None and self.count_weight:
            if count_targets is None:
                count_target_log1p = self.count_log1p_target and self.count_loss == "mse"
                count_targets = count_targets_from_profile(
                    profile_targets,
                    log1p=count_target_log1p,
                )
            count_pred_t = _as_tensor(count_pred)
            count_target_t = _as_tensor(count_targets).to(device=count_pred_t.device)
            if self.count_loss == "mse":
                count_loss = F.mse_loss(count_pred_t, count_target_t)
            elif self.count_loss == "poisson":
                count_loss = profile_poisson_loss(count_pred_t, count_target_t, log_input=True)
            else:
                raise ValueError("count_loss must be 'mse' or 'poisson'.")
            total = total + self.count_weight * count_loss

        return total


def _to_numpy_ncl(value: ArrayLike, name: str) -> np.ndarray:
    """Convert a profile array/tensor to ``(N, C, L)`` NumPy format."""
    tensor = ensure_profile_ncl(value, name)
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)


def _pearson_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute robust Pearson correlation for one flattened pair."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def calculate_profile_metrics(
    targets: ArrayLike,
    predictions: ArrayLike,
    return_dict: bool = False,
) -> Union[pd.DataFrame, Dict[str, float]]:
    """Calculate profile and count-level regression metrics."""
    y_true = _to_numpy_ncl(targets, "targets")
    y_pred = _to_numpy_ncl(predictions, "predictions")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"targets and predictions must have matching shapes; got {y_true.shape} and {y_pred.shape}.")

    per_sample_corr = [_pearson_1d(t, p) for t, p in zip(y_true, y_pred)]
    true_counts = y_true.sum(axis=(1, 2))
    pred_counts = y_pred.sum(axis=(1, 2))
    metrics = {
        "profile_mse": float(np.mean((y_pred - y_true) ** 2)),
        "profile_mae": float(np.mean(np.abs(y_pred - y_true))),
        "profile_pearson_r_mean": float(np.nanmean(per_sample_corr)),
        "count_pearson_r": _pearson_1d(true_counts, pred_counts),
        "count_mse": float(np.mean((pred_counts - true_counts) ** 2)),
    }
    if return_dict:
        return metrics
    return pd.DataFrame([metrics])


def save_profile_predictions_npz(
    output_path: Union[str, Path],
    predictions: ArrayLike,
    intervals: Optional[pd.DataFrame] = None,
    targets: Optional[ArrayLike] = None,
    track_names: Optional[Sequence[str]] = None,
) -> str:
    """Save profile predictions and optional targets to a compressed NPZ file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred = _to_numpy_ncl(predictions, "predictions")
    payload: Dict[str, Any] = {"predictions": pred, "shape_convention": "NCL"}
    if targets is not None:
        payload["targets"] = _to_numpy_ncl(targets, "targets")
    if intervals is not None:
        payload["intervals"] = intervals.astype(str).to_numpy()
        payload["interval_columns"] = np.asarray(list(intervals.columns))
    if track_names is not None:
        payload["track_names"] = np.asarray(list(track_names))
    np.savez_compressed(output_path, **payload)
    return str(output_path)


def save_profile_predictions_h5(
    output_path: Union[str, Path],
    predictions: ArrayLike,
    intervals: Optional[pd.DataFrame] = None,
    targets: Optional[ArrayLike] = None,
    track_names: Optional[Sequence[str]] = None,
) -> str:
    """Save profile predictions and optional targets to HDF5."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Saving profile predictions as HDF5 requires optional dependency 'h5py'.") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred = _to_numpy_ncl(predictions, "predictions")
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("predictions", data=pred, compression="gzip")
        handle.attrs["shape_convention"] = "NCL"
        if targets is not None:
            handle.create_dataset("targets", data=_to_numpy_ncl(targets, "targets"), compression="gzip")
        if intervals is not None:
            encoded = intervals.astype(str).to_numpy(dtype="S")
            handle.create_dataset("intervals", data=encoded)
            handle.create_dataset("interval_columns", data=np.asarray(list(intervals.columns), dtype="S"))
        if track_names is not None:
            handle.create_dataset("track_names", data=np.asarray(list(track_names), dtype="S"))
    return str(output_path)


def write_profile_predictions_bigwig(
    output_prefix: Union[str, Path],
    predictions: ArrayLike,
    intervals: pd.DataFrame,
    chrom_sizes: Dict[str, int],
    track_names: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    """Write profile predictions to one BigWig file per output channel."""
    try:
        import pyBigWig
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Writing profile predictions as BigWig requires optional dependency 'pyBigWig'.") from exc

    pred = _to_numpy_ncl(predictions, "predictions")
    required = {"chrom", "start", "end"}
    missing = required - set(intervals.columns)
    if missing:
        raise ValueError(f"intervals are missing required columns: {sorted(missing)}")
    if len(intervals) != pred.shape[0]:
        raise ValueError("Number of intervals must match number of prediction rows.")

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    names = list(track_names) if track_names is not None else [f"track_{idx}" for idx in range(pred.shape[1])]
    if len(names) != pred.shape[1]:
        raise ValueError("track_names length must match prediction channel count.")

    written = []
    header = [(str(chrom), int(size)) for chrom, size in chrom_sizes.items()]
    for channel_idx, name in enumerate(names):
        path = output_prefix.parent / f"{output_prefix.name}.{name}.bw"
        bw = pyBigWig.open(str(path), "w")
        try:
            bw.addHeader(header)
            for row_idx, row in intervals.reset_index(drop=True).iterrows():
                values = pred[row_idx, channel_idx].astype(float)
                length = int(row["end"]) - int(row["start"])
                if values.shape[0] != length:
                    raise ValueError(
                        "Prediction length must match interval length for BigWig output; "
                        f"got {values.shape[0]} and {length}."
                    )
                starts = list(range(int(row["start"]), int(row["end"])))
                ends = [start + 1 for start in starts]
                bw.addEntries([str(row["chrom"])] * length, starts, ends=ends, values=values.tolist())
        finally:
            bw.close()
        written.append(str(path))
    return tuple(written)


__all__ = [
    "ProfileCountLoss",
    "calculate_profile_metrics",
    "count_targets_from_profile",
    "ensure_profile_ncl",
    "profile_multinomial_nll_loss",
    "profile_poisson_loss",
    "save_profile_predictions_h5",
    "save_profile_predictions_npz",
    "write_profile_predictions_bigwig",
]
