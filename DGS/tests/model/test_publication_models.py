"""Shape checks for publication model exports."""

import pytest
import torch

from DGS.Model.Publications import Basset, DeepSEA


@pytest.mark.parametrize("model_cls", [Basset, DeepSEA])
def test_publication_models_accept_sciatac_sequence_shape(model_cls):
    """Publication models should run on sciATAC-style 600 bp one-hot input."""
    model = model_cls(sequence_length=600, n_genomic_features=85)
    model.eval()

    with torch.no_grad():
        output = model(torch.zeros(2, 4, 600))

    assert tuple(output.shape) == (2, 85)


def test_basset_reports_dynamic_flatten_dim_for_sciatac_length():
    """Basset should infer the dense input width instead of hard-coding it."""
    model = Basset(sequence_length=600, n_genomic_features=85)

    assert model.flatten_dim == 2000
    assert model.architecture()["flatten_dim"] == 2000
