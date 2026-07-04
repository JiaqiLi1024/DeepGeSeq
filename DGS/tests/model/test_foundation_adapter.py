"""Tests for optional foundation-model adapter interfaces."""

import importlib.util

import pytest
import torch

from DGS.Model import (
    DNABERTAdapter,
    DNABERT2Adapter,
    Evo2Adapter,
    FoundationSequenceAdapter,
    GPNAdapter,
    NucleotideTransformerAdapter,
    build_dnabert_adapter,
    build_dnabert2_adapter,
    build_genomic_language_model_adapter,
    build_gpn_adapter,
    build_nucleotide_transformer_adapter,
    build_transformers_sequence_adapter,
    get_model_zoo,
)
import DGS.Model.Foundation as foundation_module


class TensorEncoder(torch.nn.Module):
    """Tiny encoder that returns sequence embeddings from tensor inputs."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 3)

    def forward(self, x):
        if x.shape[1] == 4:
            x = x.transpose(1, 2)
        return {"last_hidden_state": self.proj(x)}


class TokenEncoder(torch.nn.Module):
    """Tiny token encoder that mimics transformer-style outputs."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(5, 2)

    def forward(self, input_ids, attention_mask=None):
        return {"last_hidden_state": self.embedding(input_ids)}


class FakeEvoTokenizer:
    """Tiny Evo-style tokenizer for adapter tests."""

    def tokenize(self, sequence):
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        return [mapping.get(base, 4) for base in sequence]


class FakeEvo2Model:
    """Small callable matching the Evo 2 forward/generate contract."""

    def __init__(self):
        self.tokenizer = FakeEvoTokenizer()

    def __call__(self, input_ids, return_embeddings=False, layer_names=None):
        if return_embeddings:
            layer = layer_names[0]
            values = torch.arange(
                input_ids.shape[0] * input_ids.shape[1] * 3,
                dtype=torch.float32,
                device=input_ids.device,
            ).reshape(input_ids.shape[0], input_ids.shape[1], 3)
            return None, {layer: values}
        logits = torch.nn.functional.one_hot(input_ids.long() % 5, num_classes=5).float()
        return (logits,), None

    def generate(self, prompt_seqs, **kwargs):
        return {"sequences": [seq + "A" for seq in prompt_seqs], "kwargs": kwargs}


def tiny_tokenizer(sequences, return_tensors="pt", padding=True):
    """Tokenize A/C/G/T/N strings into integer IDs for tests."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    max_len = max(len(seq) for seq in sequences)
    encoded = []
    for seq in sequences:
        row = [mapping.get(base, 4) for base in seq]
        row += [4] * (max_len - len(row))
        encoded.append(row)
    return {"input_ids": torch.tensor(encoded, dtype=torch.long)}


def test_foundation_adapter_pools_tensor_encoder_outputs():
    """Tensor encoders can be wrapped without optional LLM dependencies."""
    adapter = FoundationSequenceAdapter(TensorEncoder(), pooling="mean")
    x = torch.zeros(2, 6, 4)
    x[:, :, 0] = 1

    output = adapter(x)

    assert output.shape == (2, 3)


def test_foundation_adapter_tokenizes_dna_strings():
    """String inputs use the caller-provided tokenizer."""
    adapter = FoundationSequenceAdapter(
        TokenEncoder(),
        tokenizer=tiny_tokenizer,
        pooling="cls",
    )

    output = adapter(["ACGT", "TGCA"])

    assert output.shape == (2, 2)


def test_foundation_adapter_can_freeze_encoder_parameters():
    """Adapter freezing should disable encoder gradients."""
    encoder = TensorEncoder()
    adapter = FoundationSequenceAdapter(encoder, freeze_encoder=True)

    assert all(not parameter.requires_grad for parameter in adapter.encoder.parameters())


def test_foundation_adapter_rejects_unknown_pooling():
    """Unsupported pooling modes should fail early."""
    with pytest.raises(ValueError, match="pooling"):
        FoundationSequenceAdapter(TensorEncoder(), pooling="median")


def test_transformers_helper_reports_missing_optional_dependency():
    """The transformers builder should stay isolated behind the llm extra."""
    real_find_spec = importlib.util.find_spec

    if real_find_spec("transformers") is not None:
        pytest.skip("transformers is installed in this environment")

    with pytest.raises(RuntimeError, match="transformers"):
        build_transformers_sequence_adapter("dummy/model")


def test_model_zoo_lists_foundation_adapter_as_adapter_only():
    """Model metadata should not overclaim bundled foundation-model support."""
    adapter_meta = get_model_zoo()["FoundationSequenceAdapter"]

    assert adapter_meta["workflow_status"] == "adapter tested"
    assert "does not bundle" in adapter_meta["notes"]


def test_dnabert2_builder_uses_documented_transformers_defaults(monkeypatch):
    """DNABERT-2 preset should use the documented Hugging Face checkpoint."""
    calls = []

    def fake_loader(
        model_name_or_path,
        tokenizer_kwargs=None,
        model_kwargs=None,
        trust_remote_code=None,
        auto_model_class="AutoModel",
        register_module=None,
    ):
        calls.append(
            {
                "model_name_or_path": model_name_or_path,
                "tokenizer_kwargs": tokenizer_kwargs,
                "model_kwargs": model_kwargs,
                "trust_remote_code": trust_remote_code,
                "auto_model_class": auto_model_class,
                "register_module": register_module,
            }
        )
        return tiny_tokenizer, TokenEncoder()

    monkeypatch.setattr(foundation_module, "_load_transformers_model", fake_loader)

    adapter = build_dnabert2_adapter(pooling="mean")
    output = adapter(["ACGT", "TGCA"])

    assert isinstance(adapter, DNABERT2Adapter)
    assert adapter.family == "DNABERT2"
    assert calls[0]["model_name_or_path"] == "zhihan1996/DNABERT-2-117M"
    assert calls[0]["trust_remote_code"] is True
    assert output.shape == (2, 2)


def test_dnabert_v1_adapter_kmerizes_sequences_and_uses_checkpoint(monkeypatch):
    """Original DNABERT presets should convert raw DNA to k-mer token strings."""
    calls = []
    tokenized_sequences = []

    def recording_tokenizer(sequences, return_tensors="pt", padding=True):
        tokenized_sequences.extend(sequences)
        lengths = [len(sequence.split()) for sequence in sequences]
        return {"input_ids": torch.zeros((len(sequences), max(lengths)), dtype=torch.long)}

    def fake_loader(
        model_name_or_path,
        tokenizer_kwargs=None,
        model_kwargs=None,
        trust_remote_code=None,
        auto_model_class="AutoModel",
        register_module=None,
    ):
        calls.append((model_name_or_path, trust_remote_code))
        return recording_tokenizer, TokenEncoder()

    monkeypatch.setattr(foundation_module, "_load_transformers_model", fake_loader)

    adapter = build_genomic_language_model_adapter("dnabert_4mer", pooling="none")
    output = adapter(["ACGTAC"])

    assert isinstance(adapter, DNABERTAdapter)
    assert adapter.kmer == 4
    assert calls == [("zhihan1996/DNA_bert_4", None)]
    assert tokenized_sequences == ["ACGT CGTA GTAC"]
    assert output.shape == (1, 3, 2)


def test_dnabert_direct_builder_accepts_pretokenized_inputs(monkeypatch):
    """Callers can pass already k-merized DNABERT strings when needed."""
    tokenized_sequences = []

    def recording_tokenizer(sequences, return_tensors="pt", padding=True):
        tokenized_sequences.extend(sequences)
        return {"input_ids": torch.zeros((len(sequences), 2), dtype=torch.long)}

    monkeypatch.setattr(
        foundation_module,
        "_load_transformers_model",
        lambda *args, **kwargs: (recording_tokenizer, TokenEncoder()),
    )

    adapter = build_dnabert_adapter(kmer=3, pretokenized=True, pooling="cls")
    output = adapter(["ACG CGT"])

    assert isinstance(adapter, DNABERTAdapter)
    assert tokenized_sequences == ["ACG CGT"]
    assert output.shape == (1, 2)


def test_nucleotide_transformer_factory_supports_nt_presets(monkeypatch):
    """Nucleotide Transformer aliases should resolve to explicit HF checkpoints."""
    calls = []

    def fake_loader(
        model_name_or_path,
        tokenizer_kwargs=None,
        model_kwargs=None,
        trust_remote_code=None,
        auto_model_class="AutoModel",
        register_module=None,
    ):
        calls.append(model_name_or_path)
        return tiny_tokenizer, TokenEncoder()

    monkeypatch.setattr(foundation_module, "_load_transformers_model", fake_loader)

    adapter = build_genomic_language_model_adapter("nt_v2_100m", pooling="cls")
    output = adapter(["ACGT", "TGCA"])

    assert isinstance(adapter, NucleotideTransformerAdapter)
    assert calls == ["InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"]
    assert output.shape == (2, 2)


def test_genomic_language_model_factory_rejects_unknown_arguments(monkeypatch):
    """Misspelled preset arguments should fail loudly."""
    monkeypatch.setattr(
        foundation_module,
        "_load_transformers_model",
        lambda *args, **kwargs: (tiny_tokenizer, TokenEncoder()),
    )

    with pytest.raises(ValueError, match="Unsupported DNABERT2"):
        build_genomic_language_model_adapter("dnabert2", unknown=True)


def test_direct_nucleotide_transformer_builder_accepts_checkpoint_override(monkeypatch):
    """Callers can use any compatible Nucleotide Transformer HF model path."""
    calls = []

    def fake_loader(
        model_name_or_path,
        tokenizer_kwargs=None,
        model_kwargs=None,
        trust_remote_code=None,
        auto_model_class="AutoModel",
        register_module=None,
    ):
        calls.append((model_name_or_path, trust_remote_code))
        return tiny_tokenizer, TokenEncoder()

    monkeypatch.setattr(foundation_module, "_load_transformers_model", fake_loader)

    adapter = build_nucleotide_transformer_adapter(
        model_name_or_path="InstaDeepAI/custom-nt",
        trust_remote_code=False,
        pooling="max",
    )

    assert isinstance(adapter, NucleotideTransformerAdapter)
    assert calls == [("InstaDeepAI/custom-nt", False)]


def test_gpn_factory_uses_documented_registration_and_model_class(monkeypatch):
    """GPN presets should pass through AutoModelForMaskedLM and registration args."""
    calls = []

    def fake_loader(
        model_name_or_path,
        tokenizer_kwargs=None,
        model_kwargs=None,
        trust_remote_code=None,
        auto_model_class="AutoModel",
        register_module=None,
    ):
        calls.append(
            {
                "model_name_or_path": model_name_or_path,
                "trust_remote_code": trust_remote_code,
                "auto_model_class": auto_model_class,
                "register_module": register_module,
            }
        )
        return tiny_tokenizer, TokenEncoder()

    monkeypatch.setattr(foundation_module, "_load_transformers_model", fake_loader)

    adapter = build_genomic_language_model_adapter("gpn", pooling="none")
    output = adapter(["ACGT"])

    assert isinstance(adapter, GPNAdapter)
    assert adapter.output_key == "logits"
    assert calls == [
        {
            "model_name_or_path": "songlab/gpn-brassicales",
            "trust_remote_code": None,
            "auto_model_class": "AutoModelForMaskedLM",
            "register_module": "gpn.model",
        }
    ]
    assert output.shape == (1, 4, 2)


def test_gpn_direct_builder_supports_phylogpn_style_overrides(monkeypatch):
    """GPN builder should also support AutoModel/trust_remote_code checkpoints."""
    calls = []

    def fake_loader(
        model_name_or_path,
        tokenizer_kwargs=None,
        model_kwargs=None,
        trust_remote_code=None,
        auto_model_class="AutoModel",
        register_module=None,
    ):
        calls.append((model_name_or_path, trust_remote_code, auto_model_class, register_module))
        return tiny_tokenizer, TokenEncoder()

    monkeypatch.setattr(foundation_module, "_load_transformers_model", fake_loader)

    adapter = build_gpn_adapter(
        model_name_or_path="songlab/PhyloGPN",
        auto_model_class="AutoModel",
        register_module=None,
        trust_remote_code=True,
        output_key="last_hidden_state",
        pooling="mean",
    )

    assert isinstance(adapter, GPNAdapter)
    assert calls == [("songlab/PhyloGPN", True, "AutoModel", None)]

    calls.clear()
    adapter = build_genomic_language_model_adapter("phylogpn")

    assert isinstance(adapter, GPNAdapter)
    assert calls == [("songlab/PhyloGPN", True, "AutoModel", None)]


def test_evo2_adapter_returns_logits_and_generation_results():
    """Evo2Adapter should normalize Evo 2 tokenization and logits output."""
    adapter = Evo2Adapter(model_name="mock", model=FakeEvo2Model(), output_mode="logits")

    logits = adapter(["AC", "T"])
    generated = adapter.generate(prompt_seqs=["AC"], n_tokens=2)

    assert logits.shape == (2, 2, 5)
    assert generated["sequences"] == ["ACA"]
    assert generated["kwargs"] == {"n_tokens": 2}


def test_evo2_factory_supports_embedding_mode(monkeypatch):
    """Factory presets should expose Evo 2 intermediate embedding extraction."""
    monkeypatch.setattr(
        foundation_module,
        "_load_evo2_model",
        lambda model_name: FakeEvo2Model(),
    )

    adapter = build_genomic_language_model_adapter(
        "evo2-1b-base",
        output_mode="embeddings",
        layer_names=["blocks.1.mlp"],
        pooling="mean",
    )
    embeddings = adapter(["AC", "TG"])

    assert isinstance(adapter, Evo2Adapter)
    assert adapter.model_name == "evo2_1b_base"
    assert embeddings.shape == (2, 3)


def test_evo2_factory_includes_documented_large_checkpoints(monkeypatch):
    """Evo 2 presets should include documented 20B/40B checkpoint names."""
    calls = []

    def fake_loader(model_name):
        calls.append(model_name)
        return FakeEvo2Model()

    monkeypatch.setattr(foundation_module, "_load_evo2_model", fake_loader)

    adapter = build_genomic_language_model_adapter("evo2-20b")

    assert isinstance(adapter, Evo2Adapter)
    assert adapter.model_name == "evo2_20b"
    assert calls == ["evo2_20b"]
