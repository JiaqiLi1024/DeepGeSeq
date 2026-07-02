"""Adapter interfaces for genome foundation-model integrations.

This module keeps large genomic language models behind lightweight wrappers and
lazy imports. Original DNABERT, DNABERT-2, Nucleotide Transformer, and GPN
families are exposed through Hugging Face Transformers presets, while Evo 2 is
exposed through its optional ``evo2`` package.
"""

import importlib
from collections.abc import Sequence as SequenceABC
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


TensorLike = Union[np.ndarray, torch.Tensor]


GLM_MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "dnabert": {
        "family": "DNABERT2",
        "provider": "transformers",
        "model_name_or_path": "zhihan1996/DNABERT-2-117M",
        "trust_remote_code": True,
        "pooling": "mean",
    },
    "dnabert2": {
        "family": "DNABERT2",
        "provider": "transformers",
        "model_name_or_path": "zhihan1996/DNABERT-2-117M",
        "trust_remote_code": True,
        "pooling": "mean",
    },
    "dnabert-2": {
        "family": "DNABERT2",
        "provider": "transformers",
        "model_name_or_path": "zhihan1996/DNABERT-2-117M",
        "trust_remote_code": True,
        "pooling": "mean",
    },
    "nucleotide-transformer": {
        "family": "NucleotideTransformer",
        "provider": "transformers",
        "model_name_or_path": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        "trust_remote_code": True,
        "pooling": "mean",
    },
    "nt": {
        "family": "NucleotideTransformer",
        "provider": "transformers",
        "model_name_or_path": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        "trust_remote_code": True,
        "pooling": "mean",
    },
    "nt-v2-50m": {
        "family": "NucleotideTransformer",
        "provider": "transformers",
        "model_name_or_path": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        "trust_remote_code": True,
        "pooling": "mean",
    },
    "nt-v2-100m": {
        "family": "NucleotideTransformer",
        "provider": "transformers",
        "model_name_or_path": "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        "trust_remote_code": True,
        "pooling": "mean",
    },
    "nt-v2-250m": {
        "family": "NucleotideTransformer",
        "provider": "transformers",
        "model_name_or_path": "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
        "trust_remote_code": True,
        "pooling": "mean",
    },
    "nt-v2-500m": {
        "family": "NucleotideTransformer",
        "provider": "transformers",
        "model_name_or_path": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        "trust_remote_code": True,
        "pooling": "mean",
    },
    "nt-v2-50m-3mer": {
        "family": "NucleotideTransformer",
        "provider": "transformers",
        "model_name_or_path": "InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species",
        "trust_remote_code": True,
        "pooling": "mean",
    },
    "codon-nt": {
        "family": "NucleotideTransformer",
        "provider": "transformers",
        "model_name_or_path": "InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species",
        "trust_remote_code": True,
        "pooling": "mean",
    },
    "gpn": {
        "family": "GPN",
        "provider": "transformers",
        "model_name_or_path": "songlab/gpn-brassicales",
        "auto_model_class": "AutoModelForMaskedLM",
        "register_module": "gpn.model",
        "output_key": "logits",
        "pooling": "none",
    },
    "gpn-brassicales": {
        "family": "GPN",
        "provider": "transformers",
        "model_name_or_path": "songlab/gpn-brassicales",
        "auto_model_class": "AutoModelForMaskedLM",
        "register_module": "gpn.model",
        "output_key": "logits",
        "pooling": "none",
    },
    "gpn-animal-promoter": {
        "family": "GPN",
        "provider": "transformers",
        "model_name_or_path": "songlab/gpn-animal-promoter",
        "auto_model_class": "AutoModelForMaskedLM",
        "register_module": "gpn.model",
        "output_key": "logits",
        "pooling": "none",
    },
    "gpn-msa": {
        "family": "GPN",
        "provider": "transformers",
        "model_name_or_path": "songlab/gpn-msa-sapiens",
        "auto_model_class": "AutoModelForMaskedLM",
        "register_module": "gpn.model",
        "output_key": "logits",
        "pooling": "none",
    },
    "phylogpn": {
        "family": "GPN",
        "provider": "transformers",
        "model_name_or_path": "songlab/PhyloGPN",
        "auto_model_class": "AutoModel",
        "register_module": None,
        "trust_remote_code": True,
        "output_key": "last_hidden_state",
        "pooling": "mean",
    },
    "gpn-star": {
        "family": "GPN",
        "provider": "transformers",
        "model_name_or_path": "songlab/gpn-star-hg38-p243-200m",
        "auto_model_class": "AutoModelForMaskedLM",
        "register_module": "gpn.star.model",
        "output_key": "logits",
        "pooling": "none",
    },
    "evo": {
        "family": "Evo2",
        "provider": "evo2",
        "model_name": "evo2_7b",
        "output_mode": "logits",
        "pooling": "none",
    },
    "evo2": {
        "family": "Evo2",
        "provider": "evo2",
        "model_name": "evo2_7b",
        "output_mode": "logits",
        "pooling": "none",
    },
    "evo2-7b": {
        "family": "Evo2",
        "provider": "evo2",
        "model_name": "evo2_7b",
        "output_mode": "logits",
        "pooling": "none",
    },
    "evo2-7b-base": {
        "family": "Evo2",
        "provider": "evo2",
        "model_name": "evo2_7b_base",
        "output_mode": "logits",
        "pooling": "none",
    },
    "evo2-1b-base": {
        "family": "Evo2",
        "provider": "evo2",
        "model_name": "evo2_1b_base",
        "output_mode": "logits",
        "pooling": "none",
    },
    "evo2-20b": {
        "family": "Evo2",
        "provider": "evo2",
        "model_name": "evo2_20b",
        "output_mode": "logits",
        "pooling": "none",
    },
    "evo2-40b": {
        "family": "Evo2",
        "provider": "evo2",
        "model_name": "evo2_40b",
        "output_mode": "logits",
        "pooling": "none",
    },
    "evo2-40b-base": {
        "family": "Evo2",
        "provider": "evo2",
        "model_name": "evo2_40b_base",
        "output_mode": "logits",
        "pooling": "none",
    },
    "evo2-7b-262k": {
        "family": "Evo2",
        "provider": "evo2",
        "model_name": "evo2_7b_262k",
        "output_mode": "logits",
        "pooling": "none",
    },
    "evo2-7b-microviridae": {
        "family": "Evo2",
        "provider": "evo2",
        "model_name": "evo2_7b_microviridae",
        "output_mode": "logits",
        "pooling": "none",
    },
}


for _kmer in (3, 4, 5, 6):
    _dnabert_preset = {
        "family": "DNABERT",
        "provider": "transformers",
        "model_name_or_path": f"zhihan1996/DNA_bert_{_kmer}",
        "kmer": _kmer,
        "pooling": "mean",
    }
    GLM_MODEL_PRESETS[f"dnabert-{_kmer}mer"] = dict(_dnabert_preset)
    GLM_MODEL_PRESETS[f"dnabert1-{_kmer}mer"] = dict(_dnabert_preset)
    GLM_MODEL_PRESETS[f"dna-bert-{_kmer}mer"] = dict(_dnabert_preset)

GLM_MODEL_PRESETS["dnabert1"] = dict(GLM_MODEL_PRESETS["dnabert-3mer"])
GLM_MODEL_PRESETS["dnabert-1"] = dict(GLM_MODEL_PRESETS["dnabert-3mer"])
GLM_MODEL_PRESETS["dnabert-v1"] = dict(GLM_MODEL_PRESETS["dnabert-3mer"])
del _kmer, _dnabert_preset


def _normalize_preset_key(name: str) -> str:
    """Normalize user-facing preset names."""
    return str(name).strip().lower().replace("_", "-")


def _validate_pooling(pooling: str) -> None:
    """Validate pooling mode shared by foundation-model adapters."""
    if pooling not in {"mean", "cls", "max", "none"}:
        raise ValueError("pooling must be one of: 'mean', 'cls', 'max', 'none'.")


def _pool_sequence_tensor(tensor: torch.Tensor, pooling: str) -> torch.Tensor:
    """Pool sequence embeddings when requested."""
    if pooling == "none" or tensor.ndim < 3:
        return tensor
    if pooling == "mean":
        return tensor.mean(dim=1)
    if pooling == "cls":
        return tensor[:, 0]
    if pooling == "max":
        return tensor.max(dim=1).values
    raise ValueError(f"Unsupported pooling mode: {pooling}")


def _sequence_to_kmer_string(sequence: str, kmer: int) -> str:
    """Convert a nucleotide sequence to DNABERT v1-style k-mer tokens."""
    kmer = int(kmer)
    if kmer < 1:
        raise ValueError("kmer must be a positive integer.")

    sequence = str(sequence).strip().upper()
    if len(sequence) < kmer:
        raise ValueError(
            f"DNABERT k-mer tokenization requires sequences at least {kmer} bases long."
        )
    return " ".join(sequence[idx : idx + kmer] for idx in range(len(sequence) - kmer + 1))


class FoundationSequenceAdapter(nn.Module):
    """Wrap a sequence encoder behind a stable DGS-style module interface.

    Args:
        encoder: Module or callable returning a tensor, dict, or object with a
            tensor attribute such as ``last_hidden_state``.
        tokenizer: Optional callable used when inputs are DNA strings.
        output_key: Preferred key/attribute for encoder outputs.
        pooling: ``"mean"``, ``"cls"``, ``"max"``, or ``"none"``.
        freeze_encoder: Whether to disable gradients for encoder parameters.
        tokenizer_kwargs: Default keyword arguments passed to the tokenizer.

    Notes:
        The adapter is an integration point, not a claim that DGS bundles or
        validates a particular foundation model checkpoint.
    """

    def __init__(
        self,
        encoder: Union[nn.Module, Callable[..., Any]],
        tokenizer: Optional[Callable[..., Any]] = None,
        output_key: str = "last_hidden_state",
        pooling: str = "mean",
        freeze_encoder: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        _validate_pooling(pooling)

        if isinstance(encoder, nn.Module):
            self.encoder = encoder
        else:
            self.encoder = _CallableEncoder(encoder)

        self.tokenizer = tokenizer
        self.output_key = output_key
        self.pooling = pooling
        self.tokenizer_kwargs = dict(tokenizer_kwargs or {})

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        """Return the encoder device, defaulting to CPU for parameterless modules."""
        try:
            return next(self.encoder.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _tokenize(self, sequences: Union[str, Sequence[str]]) -> Dict[str, torch.Tensor]:
        """Tokenize DNA strings and move tensor fields to the encoder device."""
        if self.tokenizer is None:
            raise ValueError("String inputs require a tokenizer.")
        if isinstance(sequences, str):
            sequences = [sequences]

        kwargs = {
            "return_tensors": "pt",
            "padding": True,
            **self.tokenizer_kwargs,
        }
        encoded = self.tokenizer(list(sequences), **kwargs)
        if hasattr(encoded, "items"):
            return {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in encoded.items()
            }
        raise ValueError("tokenizer must return a mapping with tensor fields.")

    def _prepare_tensor(self, inputs: TensorLike) -> torch.Tensor:
        """Convert array/tensor inputs to a float tensor on the encoder device."""
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs)
        return inputs.to(self.device, dtype=torch.float32)

    def _extract_tensor(self, output: Any) -> torch.Tensor:
        """Extract a tensor from common encoder output conventions."""
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, dict):
            if self.output_key in output:
                return output[self.output_key]
            for value in output.values():
                if isinstance(value, torch.Tensor):
                    return value
        if hasattr(output, self.output_key):
            value = getattr(output, self.output_key)
            if isinstance(value, torch.Tensor):
                return value
        if isinstance(output, (tuple, list)):
            for value in output:
                if isinstance(value, torch.Tensor):
                    return value
        raise ValueError("encoder output does not contain a tensor representation.")

    def _pool(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pool sequence embeddings when requested."""
        return _pool_sequence_tensor(tensor, self.pooling)

    def forward(self, inputs: Union[str, Sequence[str], TensorLike]) -> torch.Tensor:
        """Encode DNA strings or tensor inputs and return pooled embeddings."""
        if isinstance(inputs, str) or (
            isinstance(inputs, SequenceABC) and bool(inputs) and isinstance(inputs[0], str)
        ):
            output = self.encoder(**self._tokenize(inputs))
        else:
            output = self.encoder(self._prepare_tensor(inputs))
        return self._pool(self._extract_tensor(output))


class GenomicLanguageModelAdapter(FoundationSequenceAdapter):
    """Foundation adapter with explicit genomic language-model provenance."""

    family: str = "genomic-language-model"

    def __init__(
        self,
        encoder: Union[nn.Module, Callable[..., Any]],
        tokenizer: Optional[Callable[..., Any]] = None,
        output_key: str = "last_hidden_state",
        pooling: str = "mean",
        freeze_encoder: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_name_or_path: Optional[str] = None,
        family: Optional[str] = None,
    ):
        super().__init__(
            encoder=encoder,
            tokenizer=tokenizer,
            output_key=output_key,
            pooling=pooling,
            freeze_encoder=freeze_encoder,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self.model_name_or_path = model_name_or_path
        if family is not None:
            self.family = family

    @classmethod
    def from_transformers(
        cls,
        model_name_or_path: str,
        pooling: str = "mean",
        freeze_encoder: bool = True,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        trust_remote_code: Optional[bool] = None,
        auto_model_class: str = "AutoModel",
        register_module: Optional[str] = None,
        output_key: str = "last_hidden_state",
    ) -> "GenomicLanguageModelAdapter":
        """Load a Hugging Face Transformers genome encoder lazily."""
        tokenizer, model = _load_transformers_model(
            model_name_or_path,
            tokenizer_kwargs=tokenizer_kwargs,
            model_kwargs=model_kwargs,
            trust_remote_code=trust_remote_code,
            auto_model_class=auto_model_class,
            register_module=register_module,
        )
        return cls(
            encoder=model,
            tokenizer=tokenizer,
            output_key=output_key,
            pooling=pooling,
            freeze_encoder=freeze_encoder,
            model_name_or_path=model_name_or_path,
        )


class DNABERTAdapter(GenomicLanguageModelAdapter):
    """Adapter for original DNABERT k-mer checkpoints loaded through Transformers."""

    family = "DNABERT"

    def __init__(
        self,
        encoder: Union[nn.Module, Callable[..., Any]],
        tokenizer: Optional[Callable[..., Any]] = None,
        output_key: str = "last_hidden_state",
        pooling: str = "mean",
        freeze_encoder: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_name_or_path: Optional[str] = None,
        kmer: int = 3,
        pretokenized: bool = False,
    ):
        if int(kmer) < 1:
            raise ValueError("kmer must be a positive integer.")
        super().__init__(
            encoder=encoder,
            tokenizer=tokenizer,
            output_key=output_key,
            pooling=pooling,
            freeze_encoder=freeze_encoder,
            tokenizer_kwargs=tokenizer_kwargs,
            model_name_or_path=model_name_or_path,
        )
        self.kmer = int(kmer)
        self.pretokenized = bool(pretokenized)

    def _tokenize(self, sequences: Union[str, Sequence[str]]) -> Dict[str, torch.Tensor]:
        """Tokenize raw DNA strings after DNABERT v1 k-mer preprocessing."""
        if isinstance(sequences, str):
            sequences = [sequences]
        prepared = list(sequences) if self.pretokenized else [
            _sequence_to_kmer_string(sequence, self.kmer) for sequence in sequences
        ]
        return super()._tokenize(prepared)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Optional[str] = None,
        kmer: int = 3,
        pretokenized: bool = False,
        pooling: str = "mean",
        freeze_encoder: bool = True,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        trust_remote_code: Optional[bool] = None,
        output_key: str = "last_hidden_state",
    ) -> "DNABERTAdapter":
        """Build an original DNABERT adapter using a documented k-mer checkpoint."""
        kmer = int(kmer)
        if model_name_or_path is None:
            model_name_or_path = f"zhihan1996/DNA_bert_{kmer}"
        tokenizer, model = _load_transformers_model(
            model_name_or_path,
            tokenizer_kwargs=tokenizer_kwargs,
            model_kwargs=model_kwargs,
            trust_remote_code=trust_remote_code,
        )
        return cls(
            encoder=model,
            tokenizer=tokenizer,
            output_key=output_key,
            pooling=pooling,
            freeze_encoder=freeze_encoder,
            model_name_or_path=model_name_or_path,
            kmer=kmer,
            pretokenized=pretokenized,
        )


class DNABERT2Adapter(GenomicLanguageModelAdapter):
    """Adapter for DNABERT-2 checkpoints loaded through Transformers."""

    family = "DNABERT2"

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "zhihan1996/DNABERT-2-117M",
        pooling: str = "mean",
        freeze_encoder: bool = True,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        trust_remote_code: bool = True,
        output_key: str = "last_hidden_state",
    ) -> "DNABERT2Adapter":
        """Build a DNABERT-2 adapter using the documented Hugging Face model."""
        return cls.from_transformers(
            model_name_or_path=model_name_or_path,
            pooling=pooling,
            freeze_encoder=freeze_encoder,
            tokenizer_kwargs=tokenizer_kwargs,
            model_kwargs=model_kwargs,
            trust_remote_code=trust_remote_code,
            output_key=output_key,
        )


class NucleotideTransformerAdapter(GenomicLanguageModelAdapter):
    """Adapter for Nucleotide Transformer checkpoints loaded through Transformers."""

    family = "NucleotideTransformer"

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        pooling: str = "mean",
        freeze_encoder: bool = True,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        trust_remote_code: bool = True,
        output_key: str = "last_hidden_state",
    ) -> "NucleotideTransformerAdapter":
        """Build a Nucleotide Transformer adapter using a Hugging Face checkpoint."""
        return cls.from_transformers(
            model_name_or_path=model_name_or_path,
            pooling=pooling,
            freeze_encoder=freeze_encoder,
            tokenizer_kwargs=tokenizer_kwargs,
            model_kwargs=model_kwargs,
            trust_remote_code=trust_remote_code,
            output_key=output_key,
        )


class GPNAdapter(GenomicLanguageModelAdapter):
    """Adapter for GPN-family checkpoints loaded through Transformers."""

    family = "GPN"

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "songlab/gpn-brassicales",
        pooling: str = "none",
        freeze_encoder: bool = True,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        trust_remote_code: Optional[bool] = None,
        output_key: str = "logits",
        auto_model_class: str = "AutoModelForMaskedLM",
        register_module: Optional[str] = "gpn.model",
    ) -> "GPNAdapter":
        """Build a GPN adapter using the documented Hugging Face load pattern."""
        return cls.from_transformers(
            model_name_or_path=model_name_or_path,
            pooling=pooling,
            freeze_encoder=freeze_encoder,
            tokenizer_kwargs=tokenizer_kwargs,
            model_kwargs=model_kwargs,
            trust_remote_code=trust_remote_code,
            auto_model_class=auto_model_class,
            register_module=register_module,
            output_key=output_key,
        )


class Evo2Adapter(nn.Module):
    """Adapter for Evo 2 forward logits or intermediate embeddings.

    Evo 2 uses its own optional package rather than Hugging Face ``AutoModel``.
    The adapter follows the documented ``Evo2(model_name)`` call pattern and
    returns either logits or one requested embedding layer normalized by the DGS
    pooling convention.
    """

    family = "Evo2"

    def __init__(
        self,
        model_name: str = "evo2_7b",
        model: Optional[Any] = None,
        layer_names: Optional[Sequence[str]] = None,
        output_mode: str = "logits",
        pooling: str = "none",
        device: Optional[Union[str, torch.device]] = None,
        pad_token_id: int = 0,
    ):
        super().__init__()
        _validate_pooling(pooling)
        if output_mode not in {"logits", "embeddings"}:
            raise ValueError("output_mode must be 'logits' or 'embeddings'.")
        if output_mode == "embeddings" and not layer_names:
            raise ValueError("Evo2 embedding mode requires at least one layer name.")

        self.model_name = model_name
        self.model = model if model is not None else _load_evo2_model(model_name)
        self.layer_names = tuple(layer_names or ())
        self.output_mode = output_mode
        self.pooling = pooling
        self.device = torch.device(device) if device is not None else None
        self.pad_token_id = int(pad_token_id)

    def _resolve_device(self) -> Optional[torch.device]:
        """Return an explicit or model-derived device if available."""
        if self.device is not None:
            return self.device
        if isinstance(self.model, nn.Module):
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                return None
        return None

    def _tokenize(self, sequences: Union[str, Sequence[str], torch.Tensor]) -> torch.Tensor:
        """Tokenize DNA strings using the Evo 2 tokenizer."""
        if isinstance(sequences, torch.Tensor):
            tokens = sequences
        else:
            if isinstance(sequences, str):
                sequences = [sequences]
            if not (
                isinstance(sequences, SequenceABC)
                and bool(sequences)
                and isinstance(sequences[0], str)
            ):
                raise ValueError("Evo2Adapter expects DNA strings or token tensors.")
            tokenized = [
                torch.as_tensor(
                    self.model.tokenizer.tokenize(sequence),
                    dtype=torch.int,
                )
                for sequence in sequences
            ]
            max_len = max(token.numel() for token in tokenized)
            tokens = torch.full(
                (len(tokenized), max_len),
                fill_value=self.pad_token_id,
                dtype=torch.int,
            )
            for idx, token in enumerate(tokenized):
                tokens[idx, : token.numel()] = token

        device = self._resolve_device()
        return tokens.to(device) if device is not None else tokens

    def forward(self, inputs: Union[str, Sequence[str], torch.Tensor]) -> torch.Tensor:
        """Run Evo 2 and return logits or pooled intermediate embeddings."""
        input_ids = self._tokenize(inputs)
        if self.output_mode == "embeddings":
            _, embeddings = self.model(
                input_ids,
                return_embeddings=True,
                layer_names=list(self.layer_names),
            )
            tensor = embeddings[self.layer_names[0]]
        else:
            outputs, _ = self.model(input_ids)
            tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        return _pool_sequence_tensor(tensor, self.pooling)

    def generate(self, prompt_seqs: Sequence[str], **kwargs: Any) -> Any:
        """Forward generation requests to the wrapped Evo 2 model."""
        if not hasattr(self.model, "generate"):
            raise AttributeError("Wrapped Evo 2 model does not expose generate().")
        return self.model.generate(prompt_seqs=prompt_seqs, **kwargs)


class _CallableEncoder(nn.Module):
    """Small nn.Module wrapper for encoder callables."""

    def __init__(self, encoder: Callable[..., Any]):
        super().__init__()
        self.encoder = encoder

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward all arguments to the wrapped callable."""
        return self.encoder(*args, **kwargs)


def _load_transformers_model(
    model_name_or_path: str,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    trust_remote_code: Optional[bool] = None,
    auto_model_class: str = "AutoModel",
    register_module: Optional[str] = None,
) -> Tuple[Any, nn.Module]:
    """Load tokenizer/model pair from Hugging Face Transformers."""
    if register_module:
        try:
            importlib.import_module(register_module)
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                f"Loading `{model_name_or_path}` requires optional module "
                f"`{register_module}`. Install the model-specific package "
                "following the upstream documentation before constructing this adapter."
            ) from exc

    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Genomic language model adapters require optional dependency "
            "`transformers`; install with `pip install -e \".[llm]\"`."
        ) from exc
    if not hasattr(transformers, auto_model_class):
        raise ValueError(f"transformers does not expose {auto_model_class}.")

    tokenizer_kwargs = dict(tokenizer_kwargs or {})
    model_kwargs = dict(model_kwargs or {})
    if trust_remote_code is not None:
        tokenizer_kwargs.setdefault("trust_remote_code", trust_remote_code)
        model_kwargs.setdefault("trust_remote_code", trust_remote_code)

    model_loader = getattr(transformers, auto_model_class)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    model = model_loader.from_pretrained(model_name_or_path, **model_kwargs)
    return tokenizer, model


def _load_evo2_model(model_name: str) -> Any:
    """Load Evo 2 lazily so base DGS installs do not require CUDA-heavy deps."""
    try:
        from evo2 import Evo2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Evo2Adapter requires the optional `evo2` package and its CUDA "
            "runtime requirements. Install Evo 2 following the upstream "
            "instructions before calling build_evo2_adapter()."
        ) from exc
    return Evo2(model_name)


def build_transformers_sequence_adapter(
    model_name_or_path: str,
    pooling: str = "mean",
    freeze_encoder: bool = True,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    trust_remote_code: Optional[bool] = None,
    output_key: str = "last_hidden_state",
    auto_model_class: str = "AutoModel",
    register_module: Optional[str] = None,
) -> FoundationSequenceAdapter:
    """Build a foundation-model adapter from Hugging Face Transformers.

    This helper imports ``transformers`` lazily so base DGS installations do not
    need LLM dependencies. It may still download model files depending on the
    ``model_name_or_path`` argument and local cache state.
    """
    tokenizer, model = _load_transformers_model(
        model_name_or_path,
        tokenizer_kwargs=tokenizer_kwargs,
        model_kwargs=model_kwargs,
        trust_remote_code=trust_remote_code,
        auto_model_class=auto_model_class,
        register_module=register_module,
    )
    return FoundationSequenceAdapter(
        encoder=model,
        tokenizer=tokenizer,
        output_key=output_key,
        pooling=pooling,
        freeze_encoder=freeze_encoder,
    )


def build_dnabert_adapter(
    model_name_or_path: Optional[str] = None,
    kmer: int = 3,
    pretokenized: bool = False,
    pooling: str = "mean",
    freeze_encoder: bool = True,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    trust_remote_code: Optional[bool] = None,
    output_key: str = "last_hidden_state",
) -> DNABERTAdapter:
    """Build an original DNABERT adapter with k-mer preprocessing."""
    return DNABERTAdapter.from_pretrained(
        model_name_or_path=model_name_or_path,
        kmer=kmer,
        pretokenized=pretokenized,
        pooling=pooling,
        freeze_encoder=freeze_encoder,
        tokenizer_kwargs=tokenizer_kwargs,
        model_kwargs=model_kwargs,
        trust_remote_code=trust_remote_code,
        output_key=output_key,
    )


def build_dnabert2_adapter(
    model_name_or_path: str = "zhihan1996/DNABERT-2-117M",
    pooling: str = "mean",
    freeze_encoder: bool = True,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    trust_remote_code: bool = True,
    output_key: str = "last_hidden_state",
) -> DNABERT2Adapter:
    """Build a DNABERT-2 adapter from the documented Hugging Face checkpoint."""
    return DNABERT2Adapter.from_pretrained(
        model_name_or_path=model_name_or_path,
        pooling=pooling,
        freeze_encoder=freeze_encoder,
        tokenizer_kwargs=tokenizer_kwargs,
        model_kwargs=model_kwargs,
        trust_remote_code=trust_remote_code,
        output_key=output_key,
    )


def build_nucleotide_transformer_adapter(
    model_name_or_path: str = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    pooling: str = "mean",
    freeze_encoder: bool = True,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    trust_remote_code: bool = True,
    output_key: str = "last_hidden_state",
) -> NucleotideTransformerAdapter:
    """Build a Nucleotide Transformer adapter from a Hugging Face checkpoint."""
    return NucleotideTransformerAdapter.from_pretrained(
        model_name_or_path=model_name_or_path,
        pooling=pooling,
        freeze_encoder=freeze_encoder,
        tokenizer_kwargs=tokenizer_kwargs,
        model_kwargs=model_kwargs,
        trust_remote_code=trust_remote_code,
        output_key=output_key,
    )


def build_gpn_adapter(
    model_name_or_path: str = "songlab/gpn-brassicales",
    pooling: str = "none",
    freeze_encoder: bool = True,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    trust_remote_code: Optional[bool] = None,
    output_key: str = "logits",
    auto_model_class: str = "AutoModelForMaskedLM",
    register_module: Optional[str] = "gpn.model",
) -> GPNAdapter:
    """Build a GPN-family adapter from a Hugging Face checkpoint."""
    return GPNAdapter.from_pretrained(
        model_name_or_path=model_name_or_path,
        pooling=pooling,
        freeze_encoder=freeze_encoder,
        tokenizer_kwargs=tokenizer_kwargs,
        model_kwargs=model_kwargs,
        trust_remote_code=trust_remote_code,
        output_key=output_key,
        auto_model_class=auto_model_class,
        register_module=register_module,
    )


def build_evo2_adapter(
    model_name: str = "evo2_7b",
    layer_names: Optional[Sequence[str]] = None,
    output_mode: str = "logits",
    pooling: str = "none",
    device: Optional[Union[str, torch.device]] = None,
    pad_token_id: int = 0,
) -> Evo2Adapter:
    """Build an Evo 2 adapter through the optional upstream ``evo2`` package."""
    return Evo2Adapter(
        model_name=model_name,
        layer_names=layer_names,
        output_mode=output_mode,
        pooling=pooling,
        device=device,
        pad_token_id=pad_token_id,
    )


def build_genomic_language_model_adapter(
    model_family: str,
    **kwargs: Any,
) -> nn.Module:
    """Build a DGS adapter for a supported genomic language-model preset.

    Supported preset aliases include ``dnabert-3mer``, ``dnabert2``,
    ``nucleotide-transformer``, ``nt-v2-50m``, ``gpn``, and ``evo2-7b``.
    Keyword arguments override preset defaults.
    """
    key = _normalize_preset_key(model_family)
    if key not in GLM_MODEL_PRESETS:
        supported = ", ".join(sorted(GLM_MODEL_PRESETS))
        raise ValueError(f"Unknown genomic language model preset '{model_family}'. Supported presets: {supported}.")

    preset = dict(GLM_MODEL_PRESETS[key])
    provider = preset.pop("provider")
    family = preset.pop("family")
    preset.update(kwargs)

    if provider == "transformers":
        model_name_or_path = preset.pop("model_name_or_path")
        common_allowed = {
            "pooling",
            "freeze_encoder",
            "tokenizer_kwargs",
            "model_kwargs",
            "trust_remote_code",
            "output_key",
        }

        def _check_allowed(allowed) -> None:
            if preset.keys() - allowed:
                unknown = ", ".join(sorted(preset.keys() - allowed))
                raise ValueError(f"Unsupported {family} adapter arguments: {unknown}.")

        if family == "DNABERT":
            _check_allowed(common_allowed | {"kmer", "pretokenized"})
            return build_dnabert_adapter(
                model_name_or_path=model_name_or_path,
                kmer=preset.pop("kmer", 3),
                pretokenized=preset.pop("pretokenized", False),
                pooling=preset.pop("pooling", "mean"),
                freeze_encoder=preset.pop("freeze_encoder", True),
                tokenizer_kwargs=preset.pop("tokenizer_kwargs", None),
                model_kwargs=preset.pop("model_kwargs", None),
                trust_remote_code=preset.pop("trust_remote_code", None),
                output_key=preset.pop("output_key", "last_hidden_state"),
            )

        if family == "DNABERT2":
            _check_allowed(common_allowed)
            return build_dnabert2_adapter(
                model_name_or_path=model_name_or_path,
                pooling=preset.pop("pooling", "mean"),
                freeze_encoder=preset.pop("freeze_encoder", True),
                tokenizer_kwargs=preset.pop("tokenizer_kwargs", None),
                model_kwargs=preset.pop("model_kwargs", None),
                trust_remote_code=preset.pop("trust_remote_code", True),
                output_key=preset.pop("output_key", "last_hidden_state"),
            )

        if family == "NucleotideTransformer":
            _check_allowed(common_allowed)
            return build_nucleotide_transformer_adapter(
                model_name_or_path=model_name_or_path,
                pooling=preset.pop("pooling", "mean"),
                freeze_encoder=preset.pop("freeze_encoder", True),
                tokenizer_kwargs=preset.pop("tokenizer_kwargs", None),
                model_kwargs=preset.pop("model_kwargs", None),
                trust_remote_code=preset.pop("trust_remote_code", True),
                output_key=preset.pop("output_key", "last_hidden_state"),
            )

        if family == "GPN":
            _check_allowed(common_allowed | {"auto_model_class", "register_module"})
            return build_gpn_adapter(
                model_name_or_path=model_name_or_path,
                pooling=preset.pop("pooling", "none"),
                freeze_encoder=preset.pop("freeze_encoder", True),
                tokenizer_kwargs=preset.pop("tokenizer_kwargs", None),
                model_kwargs=preset.pop("model_kwargs", None),
                trust_remote_code=preset.pop("trust_remote_code", None),
                output_key=preset.pop("output_key", "logits"),
                auto_model_class=preset.pop("auto_model_class", "AutoModelForMaskedLM"),
                register_module=preset.pop("register_module", "gpn.model"),
            )

        raise ValueError(f"Unsupported genomic language model family: {family}.")

    if provider == "evo2":
        model_name = preset.pop("model_name")
        allowed = {
            "layer_names",
            "output_mode",
            "pooling",
            "device",
            "pad_token_id",
        }
        if preset.keys() - allowed:
            unknown = ", ".join(sorted(preset.keys() - allowed))
            raise ValueError(f"Unsupported Evo2 adapter arguments: {unknown}.")
        return build_evo2_adapter(
            model_name=model_name,
            layer_names=preset.pop("layer_names", None),
            output_mode=preset.pop("output_mode", "logits"),
            pooling=preset.pop("pooling", "none"),
            device=preset.pop("device", None),
            pad_token_id=preset.pop("pad_token_id", 0),
        )

    raise ValueError(f"Unsupported genomic language model provider: {provider}.")


__all__ = [
    "FoundationSequenceAdapter",
    "GenomicLanguageModelAdapter",
    "DNABERTAdapter",
    "DNABERT2Adapter",
    "NucleotideTransformerAdapter",
    "GPNAdapter",
    "Evo2Adapter",
    "GLM_MODEL_PRESETS",
    "build_transformers_sequence_adapter",
    "build_dnabert_adapter",
    "build_dnabert2_adapter",
    "build_nucleotide_transformer_adapter",
    "build_gpn_adapter",
    "build_evo2_adapter",
    "build_genomic_language_model_adapter",
]
