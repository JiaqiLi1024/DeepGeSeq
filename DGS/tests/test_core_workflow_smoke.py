"""CPU smoke test for the core DGS library workflow."""

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

try:
    import pysam
except ImportError:  # pragma: no cover - optional dependency
    pysam = None

from DGS.Data import build_supervised_dataloaders
from DGS.DL import Explain as explain_module
from DGS.DL.Predict import VariantDataset, read_vcf, vep_centred_on_ds
from DGS.DL.Trainer import Trainer
from DGS.Data.Sequence import Genome


pytestmark = pytest.mark.skipif(pysam is None, reason="pysam is required for smoke test")


class TinySequenceModel(torch.nn.Module):
    """Small deterministic model for CPU smoke workflows."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(4, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(4, 1),
        )

    def forward(self, x):
        if x.ndim == 3 and x.shape[-1] == 4:
            x = x.transpose(1, 2)
        return self.net(x)


def _write_smoke_files(tmp_path: Path):
    fasta_path = tmp_path / "mini.fa"
    chr1 = list("ACGT" * 16)
    chr2 = list("TGCA" * 16)
    chr3 = list("GATTACAAGTCC" * 6)
    chr1[9] = "C"  # VCF chr1:10 reference allele.
    fasta_path.write_text(
        ">chr1\n" + "".join(chr1) + "\n"
        ">chr2\n" + "".join(chr2) + "\n"
        ">chr3\n" + "".join(chr3) + "\n"
    )
    pysam.faidx(str(fasta_path))

    intervals = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2", "chr2", "chr3", "chr3"],
            "start": [0, 4, 0, 4, 0, 8],
            "end": [12, 16, 12, 16, 12, 20],
            "name": [f"region_{i}" for i in range(6)],
            "score": [0] * 6,
            "strand": ["+", "+", "+", "+", "+", "+"],
        }
    )
    intervals_path = tmp_path / "regions.bed"
    intervals.to_csv(intervals_path, sep="\t", header=False, index=False)

    targets = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2", "chr3"],
            "start": [0, 0, 0],
            "end": [24, 24, 24],
            "name": [1, 1, 1],
        }
    )
    targets_path = tmp_path / "targets.bed"
    targets.to_csv(targets_path, sep="\t", header=False, index=False)

    vcf_path = tmp_path / "variants.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\n"
        "chr1\t10\trs1\tC\tT\t100\tPASS\t.\tGT\n"
    )

    tasks = [
        {
            "task_name": "binding",
            "file_path": str(targets_path),
            "file_type": "bed",
            "task_type": "binary",
            "target_column": "name",
        }
    ]
    return fasta_path, intervals_path, tasks, vcf_path


def test_core_fasta_bed_train_explain_predict_smoke(tmp_path):
    """Run tiny FASTA/BED -> train/evaluate/explain -> VCF prediction on CPU."""
    fasta_path, intervals_path, tasks, vcf_path = _write_smoke_files(tmp_path)

    train_loader, val_loader, test_loader = build_supervised_dataloaders(
        fasta_path=fasta_path,
        intervals_path=intervals_path,
        target_tasks=tasks,
        batch_size=2,
        split="chromosome",
        val_chroms=["chr2"],
        test_chroms=["chr1"],
        train_shuffle=False,
        num_workers=0,
    )

    model = TinySequenceModel()
    trainer = Trainer(
        model=model,
        criterion=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
        use_tensorboard=False,
        patience=1,
    )
    trainer.train(train_loader, val_loader, epochs=1, early_stopping=False, verbose=False)
    _, _, predictions, targets = trainer.validate(test_loader, return_predictions=True)
    assert predictions.shape[0] == targets.shape[0]

    with mock.patch.object(explain_module, "_TANGERMEME_IMPORT_ERROR", None), \
         mock.patch.object(explain_module, "deep_lift_shap", side_effect=lambda _m, x, target=0: x * 0.1):
        seqs, attrs = explain_module.calculate_attributions_on_ds(
            trainer.model,
            test_loader.dataset,
            target=0,
            device=torch.device("cpu"),
            batch_size=2,
            method="deeplift_shap",
        )
    assert seqs.shape == attrs.shape
    assert attrs.shape[-2:] == (4, 12)

    genome = Genome(fasta_path)
    try:
        variants = read_vcf(vcf_path)
        variant_ds = VariantDataset(genome, variants, target_len=12)
        scores = vep_centred_on_ds(
            trainer.model,
            variant_ds,
            metric_func="diff",
            mean_by_tasks=True,
            device=torch.device("cpu"),
            batch_size=2,
        )
    finally:
        genome.close()

    assert scores.shape == (1,)
    assert np.isfinite(scores).all()
