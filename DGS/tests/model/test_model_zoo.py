"""Tests for public model support metadata."""

from DGS import Model


def test_model_zoo_lists_exported_public_models():
    """Model zoo metadata should match public model exports."""
    zoo = Model.get_model_zoo()

    for model_name in [
        "CNN",
        "CAN",
        "DeepSEA",
        "Beluga",
        "DanQ",
        "Basset",
        "BPNet",
        "ChromBPNet",
        "Enformer",
        "Borzoi",
        "FoundationSequenceAdapter",
        "DNABERTAdapter",
        "DNABERT2Adapter",
        "NucleotideTransformerAdapter",
        "GPNAdapter",
        "Evo2Adapter",
        "scBasset",
    ]:
        assert model_name in zoo
        assert hasattr(Model, model_name)
        assert zoo[model_name]["class_name"] == model_name
        assert "workflow_status" in zoo[model_name]
        assert "output_shape" in zoo[model_name]


def test_model_zoo_is_a_copy():
    """Callers should not mutate the registry through helper return values."""
    zoo = Model.get_model_zoo()
    zoo["CNN"]["workflow_status"] = "changed"

    assert Model.get_model_zoo()["CNN"]["workflow_status"] == "tested"


def test_profile_model_support_lists_native_and_external_paths():
    """Profile metadata should advertise native and external-model support paths."""
    bpnet = Model.get_model_zoo()["BPNet"]
    chrombpnet = Model.get_model_zoo()["ChromBPNet"]
    borzoi = Model.get_model_zoo()["Borzoi"]

    assert bpnet["task_type"] == "profile/count prediction"
    assert bpnet["workflow_status"] == "minimal profile workflow tested"
    assert "Keras checkpoint inference" in chrombpnet["workflow_status"]
    assert "Keras checkpoint inference" in borzoi["workflow_status"]


def test_genomic_language_model_support_lists_external_scope():
    """Genome-LM metadata should advertise tested adapters without bundling weights."""
    zoo = Model.get_model_zoo()

    assert "does not bundle" in zoo["FoundationSequenceAdapter"]["notes"]
    for model_name in [
        "DNABERTAdapter",
        "DNABERT2Adapter",
        "NucleotideTransformerAdapter",
        "GPNAdapter",
        "Evo2Adapter",
    ]:
        assert "external" in zoo[model_name]["workflow_status"].lower()
        assert zoo[model_name]["tested_in"] == "DGS/tests/model/test_foundation_adapter.py"
