"""Profile-model architectures and adapters for long regulatory tracks.

The classes here keep DGS-native training in PyTorch while making room for
official TensorFlow/Keras checkpoints through a lazy adapter. ChromBPNet and
Borzoi both produce sequence-aligned profile outputs, so their interfaces follow
the profile/count conventions used by :mod:`DGS.DL.Profile`.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TensorLike = Union[np.ndarray, torch.Tensor]


def _alias_value(
    value: Any,
    aliases: Mapping[str, Any],
    names: Sequence[str],
    default: Any,
) -> Any:
    """Return the first non-None value from ``value`` or named aliases."""
    if value is not None:
        return value
    for name in names:
        if aliases.get(name) is not None:
            return aliases[name]
    return default


def _one_hot_strings(sequences: Union[str, Sequence[str]]) -> torch.Tensor:
    """Encode DNA strings as channel-last one-hot tensors."""
    if isinstance(sequences, str):
        sequences = [sequences]
    if not sequences:
        raise ValueError("sequences must contain at least one DNA string.")
    seq_len = len(sequences[0])
    if any(len(seq) != seq_len for seq in sequences):
        raise ValueError("All sequences must have the same length.")

    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = torch.zeros((len(sequences), seq_len, 4), dtype=torch.float32)
    for row, sequence in enumerate(sequences):
        for col, base in enumerate(sequence.upper()):
            idx = mapping.get(base)
            if idx is not None:
                encoded[row, col, idx] = 1.0
    return encoded


def _to_tensor(inputs: Union[str, Sequence[str], TensorLike, Sequence[Any]]) -> torch.Tensor:
    """Convert supported input containers to a tensor without changing layout."""
    if isinstance(inputs, str) or (
        isinstance(inputs, Sequence)
        and bool(inputs)
        and isinstance(inputs[0], str)
    ):
        return _one_hot_strings(inputs)
    if isinstance(inputs, np.ndarray):
        return torch.from_numpy(inputs)
    if isinstance(inputs, torch.Tensor):
        return inputs
    return torch.as_tensor(inputs)


def _sequence_to_ncl(
    inputs: Union[str, Sequence[str], TensorLike, Sequence[Any]],
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, bool]:
    """Return one-hot sequence tensors in ``(batch, 4, length)`` format."""
    tensor = _to_tensor(inputs)
    no_batch = tensor.ndim == 2
    if no_batch:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError("sequence input must have shape (N, L, 4) or (N, 4, L).")

    if tensor.shape[1] == 4:
        tensor = tensor.float()
    elif tensor.shape[-1] == 4:
        tensor = tensor.transpose(1, 2).contiguous().float()
    else:
        raise ValueError("sequence input must include a DNA channel axis of size 4.")

    if device is not None:
        tensor = tensor.to(device)
    return tensor, no_batch


def _sequence_to_nlc(
    inputs: Union[str, Sequence[str], TensorLike, Sequence[Any]],
) -> Tuple[torch.Tensor, bool]:
    """Return one-hot sequence tensors in ``(batch, length, 4)`` format."""
    tensor = _to_tensor(inputs)
    no_batch = tensor.ndim == 2
    if no_batch:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError("sequence input must have shape (N, L, 4) or (N, 4, L).")
    if tensor.shape[-1] == 4:
        return tensor.float(), no_batch
    if tensor.shape[1] == 4:
        return tensor.transpose(1, 2).contiguous().float(), no_batch
    raise ValueError("sequence input must include a DNA channel axis of size 4.")


def _center_crop_1d(tensor: torch.Tensor, target_length: Optional[int]) -> torch.Tensor:
    """Center-crop the final sequence axis when a target length is requested."""
    if target_length is None or target_length < 0:
        return tensor
    current = tensor.shape[-1]
    if current < target_length:
        raise ValueError(
            f"Cannot crop sequence axis of length {current} to longer target_length={target_length}."
        )
    if current == target_length:
        return tensor
    left = (current - target_length) // 2
    return tensor[..., left : left + target_length]


def _crop_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Center-crop tensor to match the reference sequence length."""
    return _center_crop_1d(tensor, reference.shape[-1])


def _ensure_profile_ncl(value: Any, channels: int = 1) -> torch.Tensor:
    """Normalize profile-like outputs to ``(batch, channels, length)``."""
    tensor = torch.as_tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0).unsqueeze(1)
    elif tensor.ndim == 2:
        if channels > 1 and tensor.shape[1] % channels == 0:
            tensor = tensor.reshape(tensor.shape[0], channels, -1)
        else:
            tensor = tensor.unsqueeze(1)
    elif tensor.ndim == 3:
        if tensor.shape[1] == channels:
            pass
        elif tensor.shape[-1] == channels:
            tensor = tensor.transpose(1, 2).contiguous()
        elif tensor.shape[1] <= 8:
            pass
        elif tensor.shape[-1] <= 8:
            tensor = tensor.transpose(1, 2).contiguous()
        else:
            raise ValueError("Cannot infer profile channel axis.")
    else:
        raise ValueError("profile output must have 1, 2, or 3 dimensions.")
    return tensor.float()


def _ensure_count_nc(value: Any) -> torch.Tensor:
    """Normalize count-like outputs to ``(batch, channels)``."""
    tensor = torch.as_tensor(value).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 2:
        tensor = tensor.reshape(tensor.shape[0], -1)
    return tensor


def _split_profile_count(output: Any, profile_channels: int = 1) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extract profile and optional count tensors from common output formats."""
    if isinstance(output, Mapping):
        profile = None
        for key in ("profile_logits", "profile", "profiles", "logits_profile_predictions"):
            if key in output:
                profile = output[key]
                break
        if profile is None:
            for value in output.values():
                if isinstance(value, torch.Tensor):
                    profile = value
                    break
        count = None
        for key in ("count", "counts", "logcount", "logcounts", "logcount_predictions"):
            if key in output:
                count = output[key]
                break
        return _ensure_profile_ncl(profile, profile_channels), (
            _ensure_count_nc(count) if count is not None else None
        )

    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Model output is empty.")
        profile = _ensure_profile_ncl(output[0], profile_channels)
        count = _ensure_count_nc(output[1]) if len(output) > 1 and output[1] is not None else None
        return profile, count

    return _ensure_profile_ncl(output, profile_channels), None


class _SameDilatedResidualBlock(nn.Module):
    """Same-length dilated residual convolution block."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3, activation: Optional[nn.Module] = None):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.activation = activation or nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual convolution."""
        return x + self.activation(self.conv(x))


class _ValidDilatedResidualBlock(nn.Module):
    """Valid-padded residual block with symmetric residual cropping."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3, activation: Optional[nn.Module] = None):
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
        )
        self.activation = activation or nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply valid convolution and crop the residual path before addition."""
        conv = self.activation(self.conv(x))
        cropped = _center_crop_1d(x, conv.shape[-1])
        return cropped + conv


class _ChromBPNetCore(nn.Module):
    """Sequence-only ChromBPNet-style profile/count predictor."""

    def __init__(
        self,
        n_filters: int,
        n_layers: int,
        n_outputs: int,
        output_len: Optional[int],
        conv_kernel_size: int,
        profile_kernel_size: int,
        padding: str,
        profile_output_bias: bool,
        count_output_bias: bool,
    ):
        super().__init__()
        if padding not in {"same", "valid"}:
            raise ValueError("padding must be 'same' or 'valid'.")
        conv_padding = conv_kernel_size // 2 if padding == "same" else 0
        self.output_len = output_len
        self.padding = padding
        self.iconv = nn.Conv1d(4, n_filters, conv_kernel_size, padding=conv_padding)
        self.activation = nn.ReLU()

        block_class = _SameDilatedResidualBlock if padding == "same" else _ValidDilatedResidualBlock
        self.residual_blocks = nn.ModuleList(
            [
                block_class(n_filters, dilation=2**idx, kernel_size=3, activation=nn.ReLU())
                for idx in range(1, n_layers + 1)
            ]
        )
        profile_padding = profile_kernel_size // 2 if padding == "same" else 0
        self.profile_head = nn.Conv1d(
            n_filters,
            n_outputs,
            profile_kernel_size,
            padding=profile_padding,
            bias=profile_output_bias,
        )
        self.count_head = nn.Linear(n_filters, n_outputs, bias=count_output_bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return profile logits and log-count predictions."""
        x = self.activation(self.iconv(x))
        for block in self.residual_blocks:
            x = block(x)
        profile_logits = _center_crop_1d(self.profile_head(x), self.output_len)
        count_pred = self.count_head(x.mean(dim=-1))
        return profile_logits, count_pred


class ChromBPNet(nn.Module):
    """PyTorch ChromBPNet-style profile/count model.

    The native implementation follows the documented ChromBPNet shape contract:
    sequence input, dilated residual convolution trunk, profile logits, and
    log-count output. Set ``padding="valid"`` to more closely mirror the
    official Keras architecture; the default ``"same"`` is easier to combine
    with DGS fixed-width profile targets.
    """

    def __init__(
        self,
        input_len: Optional[int] = None,
        output_len: Optional[int] = None,
        n_filters: int = 64,
        n_layers: int = 8,
        n_outputs: Optional[int] = None,
        output_size: Optional[int] = None,
        n_tasks: Optional[int] = None,
        bias_model: Optional[nn.Module] = None,
        freeze_bias: bool = True,
        padding: str = "same",
        conv_kernel_size: int = 21,
        profile_kernel_size: int = 75,
        profile_output_bias: bool = True,
        count_output_bias: bool = True,
        **aliases: Any,
    ):
        super().__init__()
        self.input_len = _alias_value(input_len, aliases, ("input_size", "sequence_length"), None)
        self.output_len = _alias_value(output_len, aliases, ("profile_length", "target_length", "output_dim"), None)
        n_outputs = _alias_value(n_outputs, aliases, ("num_tasks",), None)
        if n_outputs is None:
            n_outputs = output_size if output_size is not None else n_tasks
        self.n_outputs = int(n_outputs or 1)

        self.tf_model = _ChromBPNetCore(
            n_filters=int(n_filters),
            n_layers=int(n_layers),
            n_outputs=self.n_outputs,
            output_len=self.output_len,
            conv_kernel_size=int(conv_kernel_size),
            profile_kernel_size=int(profile_kernel_size),
            padding=padding,
            profile_output_bias=profile_output_bias,
            count_output_bias=count_output_bias,
        )
        self.bias_model = bias_model
        self.freeze_bias = bool(freeze_bias)
        if self.bias_model is not None and self.freeze_bias:
            for parameter in self.bias_model.parameters():
                parameter.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        """Return the model device, defaulting to CPU for parameterless models."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def forward(self, inputs: Union[str, Sequence[str], TensorLike]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict profile logits and log counts for one-hot DNA inputs."""
        x, no_batch = _sequence_to_ncl(inputs, self.device)
        profile_logits, count_pred = self.tf_model(x)

        if self.bias_model is not None:
            bias_output = self.bias_model(x)
            bias_profile, bias_count = _split_profile_count(bias_output, profile_channels=self.n_outputs)
            bias_profile = _crop_like(bias_profile.to(profile_logits.device), profile_logits)
            profile_logits = profile_logits + bias_profile
            if bias_count is not None:
                bias_count = bias_count.to(count_pred.device)
                count_pred = torch.logsumexp(torch.stack([count_pred, bias_count], dim=-1), dim=-1)

        if no_batch:
            return profile_logits.squeeze(0), count_pred.squeeze(0)
        return profile_logits, count_pred


class _BorzoiResidualBlock(nn.Module):
    """Residual convolution block used by the DGS-native Borzoi module."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
        )
        self.pool = nn.MaxPool1d(pool_size) if pool_size and pool_size > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block and optional pooling."""
        x = self.proj(x) + self.conv(x)
        return self.pool(x)


class Borzoi(nn.Module):
    """DGS-native Borzoi-style long-sequence profile predictor.

    This is a PyTorch module following the public Borzoi design pattern from
    the local model notes: DNA convolution, residual downsampling tower,
    transformer tower, U-Net-like upsampling, and species/task output heads.
    Official Calico ``.h5`` checkpoints can be used through
    :class:`KerasProfileAdapter` when a TensorFlow/Borzoi environment is
    available.
    """

    def __init__(
        self,
        input_len: Optional[int] = None,
        output_heads: Optional[Dict[str, int]] = None,
        output_size: Optional[int] = None,
        target_length: Optional[int] = None,
        default_head: Optional[str] = None,
        stem_filters: int = 128,
        filters_init: int = 160,
        filters_end: int = 256,
        res_blocks: int = 3,
        transformer_depth: int = 2,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.1,
        upsample_blocks: int = 2,
        final_filters: Optional[int] = None,
        kernel_size: int = 5,
        pool_size: int = 2,
        dropout: float = 0.1,
        **aliases: Any,
    ):
        super().__init__()
        self.input_len = _alias_value(input_len, aliases, ("input_size", "sequence_length", "seq_length"), None)
        target_length = _alias_value(target_length, aliases, ("profile_length", "output_len"), None)
        if output_heads is None:
            output_heads = {"profile": int(output_size or aliases.get("n_outputs") or 1)}
        self.output_heads = dict(output_heads)
        if not self.output_heads:
            raise ValueError("output_heads must contain at least one head.")
        self.default_head = default_head or ("human" if "human" in self.output_heads else next(iter(self.output_heads)))
        self.target_length = target_length

        self.stem = nn.Sequential(
            nn.Conv1d(4, stem_filters, kernel_size=15, padding=7),
            nn.GELU(),
            nn.MaxPool1d(pool_size),
        )

        channels = [stem_filters]
        if res_blocks > 0:
            for idx in range(res_blocks):
                if res_blocks == 1:
                    value = filters_end
                else:
                    frac = idx / (res_blocks - 1)
                    value = int(round(filters_init + frac * (filters_end - filters_init)))
                channels.append(value)

        tower = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            tower.append(_BorzoiResidualBlock(in_ch, out_ch, kernel_size=kernel_size, pool_size=pool_size))
        self.res_tower = nn.Sequential(*tower)

        width = channels[-1]
        if transformer_depth > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=transformer_heads,
                dim_feedforward=width * 2,
                dropout=transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_depth)
        else:
            self.transformer = None

        unet = []
        for _ in range(upsample_blocks):
            unet.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv1d(width, width, kernel_size=3, padding=1),
                    nn.BatchNorm1d(width),
                    nn.GELU(),
                ]
            )
        self.unet = nn.Sequential(*unet)

        final_filters = int(final_filters or width)
        self.final = nn.Sequential(
            nn.Conv1d(width, final_filters, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Conv1d(final_filters, units, kernel_size=1), nn.Softplus())
                for name, units in self.output_heads.items()
            }
        )

    @property
    def device(self) -> torch.device:
        """Return the model device."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def trunk(self, inputs: Union[str, Sequence[str], TensorLike]) -> torch.Tensor:
        """Return Borzoi-style trunk embeddings in ``(batch, channels, length)``."""
        x, _ = _sequence_to_ncl(inputs, self.device)
        x = self.stem(x)
        x = self.res_tower(x)
        if self.transformer is not None:
            x = self.transformer(x.transpose(1, 2)).transpose(1, 2).contiguous()
        x = self.unet(x)
        x = _center_crop_1d(x, self.target_length)
        return self.final(x)

    def forward(
        self,
        inputs: Union[str, Sequence[str], TensorLike],
        head: Optional[str] = "__default__",
        return_embeddings: bool = False,
        return_only_embeddings: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Predict profile tracks for the selected head."""
        embeddings = self.trunk(inputs)
        if return_only_embeddings:
            return embeddings

        selected_head = self.default_head if head == "__default__" else head
        if selected_head is None:
            outputs = {name: head_module(embeddings) for name, head_module in self.heads.items()}
        else:
            if selected_head not in self.heads:
                raise ValueError(f"Unknown Borzoi head: {selected_head}")
            outputs = self.heads[selected_head](embeddings)

        if return_embeddings:
            return outputs, embeddings
        return outputs


class KerasProfileAdapter(nn.Module):
    """Wrap an external Keras profile model behind a PyTorch ``nn.Module`` API.

    The adapter is intended for inference with official ChromBPNet/Borzoi-style
    ``.h5`` models. It is deliberately lazy: TensorFlow is imported only when a
    path is supplied. Outputs are converted back to torch tensors on the input
    device and normalized to DGS profile/count conventions.
    """

    def __init__(
        self,
        keras_model: Any,
        profile_channels: int = 1,
        custom_objects: Optional[Dict[str, Any]] = None,
        compile: bool = False,
        predict_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.profile_channels = int(profile_channels)
        self.predict_kwargs = {"verbose": 0, **(predict_kwargs or {})}

        if isinstance(keras_model, (str, Path)):
            try:
                from tensorflow.keras.models import load_model
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "Loading external Keras profile models requires TensorFlow. "
                    "Install a TensorFlow environment compatible with the source model."
                ) from exc
            self.keras_model = load_model(
                str(keras_model),
                custom_objects=custom_objects,
                compile=compile,
            )
        else:
            self.keras_model = keras_model

    def _predict_numpy(self, sequence_nlc: np.ndarray) -> Any:
        """Run the wrapped Keras object on a NumPy sequence batch."""
        if hasattr(self.keras_model, "predict"):
            return self.keras_model.predict(sequence_nlc, **self.predict_kwargs)
        return self.keras_model(sequence_nlc)

    def forward(self, inputs: Union[str, Sequence[str], TensorLike]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run external model inference and return torch tensors."""
        device = inputs.device if isinstance(inputs, torch.Tensor) else torch.device("cpu")
        sequence_nlc, no_batch = _sequence_to_nlc(inputs)
        raw_output = self._predict_numpy(sequence_nlc.detach().cpu().numpy())
        profile, count = _split_profile_count(raw_output, profile_channels=self.profile_channels)
        profile = profile.to(device=device)
        if no_batch:
            profile = profile.squeeze(0)
        if count is None:
            return profile
        count = count.to(device=device)
        if no_batch:
            count = count.squeeze(0)
        return profile, count


def load_keras_profile_model(
    model_path: Union[str, Path],
    profile_channels: int = 1,
    custom_objects: Optional[Dict[str, Any]] = None,
    compile: bool = False,
    predict_kwargs: Optional[Dict[str, Any]] = None,
) -> KerasProfileAdapter:
    """Load an external Keras profile/count checkpoint as a DGS module."""
    return KerasProfileAdapter(
        model_path,
        profile_channels=profile_channels,
        custom_objects=custom_objects,
        compile=compile,
        predict_kwargs=predict_kwargs,
    )


__all__ = [
    "ChromBPNet",
    "Borzoi",
    "KerasProfileAdapter",
    "load_keras_profile_model",
]
