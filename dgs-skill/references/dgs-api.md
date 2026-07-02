# DGS API Reference

Use this only when the DGS skill needs concrete package details beyond the quick workflow.

## CLI

Console entry point:

```bash
dgs [--verbose 0|1|2] [--gpu <id|-1>] [--seed <int>] [--benchmark|--no-benchmark] <command>
```

Commands:

- `dgs config --example minimal --output config.json`
- `dgs config --example full --output config.json`
- `dgs run --config config.json`
- `dgs train --config config.json`
- `dgs evaluate --config config.json`
- `dgs explain --config config.json`
- `dgs predict --config config.json`

Global flags must appear before the subcommand. `--gpu -1` forces CPU. `--no-benchmark` improves reproducibility.

If the console command is unavailable, run from the DeepGeSeq repository:

```bash
python -m DGS.Main --gpu -1 run --config config.json
```

## Config Schema

Top-level:

- `modes`: subset of `["train", "evaluate", "explain", "predict"]`.
- `device`: `"cuda"` or `"cpu"`.
- `output_dir`: main output directory.
- `data`: FASTA, intervals, target tasks, split and dataloader settings.
- `model`: model type and constructor args.
- `train`: optimizer, criterion, epochs, checkpoints, tensorboard, AMP.
- `evaluate`: checkpoint and output settings.
- `explain`: attribution/motif settings.
- `predict`: VCF and variant-effect settings.

Data section:

```json
{
  "genome_path": "reference.fa",
  "intervals_path": "regions.bed",
  "target_tasks": [
    {"task_name": "signal", "file_path": "signal.bw", "file_type": "bigwig"}
  ],
  "train_test_split": "random_split",
  "test_size": 0.2,
  "val_size": 0.2,
  "test_chroms": ["chr8"],
  "val_chroms": ["chr7"],
  "strand_aware": true,
  "batch_size": 4,
  "loader_mode": "streaming",
  "random_state": 42,
  "num_workers": 0,
  "pin_memory": false,
  "persistent_workers": false,
  "prefetch_factor": null
}
```

Target tasks:

- BigWig task: `{"task_name": "...", "file_path": "...bw", "file_type": "bigwig", "bin_size": null, "aggfunc": "mean"}`
- BED task: `{"task_name": "...", "file_path": "...bed", "file_type": "bed", "task_type": "binary", "target_column": "name"}`

Training:

```json
{
  "optimizer": {"type": "Adam", "params": {"lr": 0.001}},
  "criterion": {"type": "MSELoss", "params": {}},
  "patience": 10,
  "max_epochs": 500,
  "checkpoint_dir": "checkpoints",
  "use_tensorboard": false,
  "tensorboard_dir": "tensorboard",
  "use_amp": false,
  "amp_dtype": "float16",
  "non_blocking": false
}
```

Evaluation-only requires `evaluate.checkpoint_path` unless evaluation follows training in the same `run`.

Prediction:

```json
{
  "vcf_path": "variants.vcf",
  "sequence_length": 1000,
  "metric_func": "diff",
  "mean_by_tasks": true,
  "batch_size": null
}
```

Explain:

```json
{
  "target": 0,
  "output_dir": "motif_results",
  "max_seqlets": 2000,
  "method": "deeplift_shap",
  "baseline": "zero",
  "n_steps": 50,
  "internal_batch_size": null
}
```

Supported explain methods include `deeplift_shap`, `deeplift`, `integrated_gradients`, and `ig`.

## Common Python APIs

Data loaders:

```python
from DGS.Data import (
    build_supervised_dataloaders,
    build_supervised_dataloader,
    build_sequence_dataloader,
    build_profile_dataloaders,
    build_profile_dataloader,
)
```

Training:

```python
from DGS.DL.Trainer import Trainer
```

Evaluation metrics:

```python
from DGS.DL.Evaluator import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    calculate_sequence_classification_metrics,
    calculate_sequence_regression_metrics,
)
```

Explanation:

```python
from DGS.DL.Explain import calculate_attributions, calculate_attributions_on_ds, motif_enrich
```

Variant effect prediction:

```python
from DGS.DL.Predict import VariantDataset, vep_centred_on_ds, vep_centred_from_files
```

Sequence design:

```python
from DGS.DL.Design import gradient_ascent_sequence_design, greedy_ism_sequence_design
```

Profile models/losses:

```python
from DGS.Model import ChromBPNet, Borzoi, KerasProfileAdapter, load_keras_profile_model
from DGS.DL.Profile import ProfileCountLoss, calculate_profile_metrics, save_profile_predictions_npz
```

Sequence helpers:

```python
from DGS.Data.Sequence import one_hot_encode, one_hot_decode, reverse_complement, calculate_gc_content
```

Interval helpers:

```python
from DGS.Data.Interval import Interval, find_overlaps, merge_intervals, find_closest
```

## Model Names

Use `python <DeepGeSeq>/DGS/skills/dgs/scripts/dgs_helper.py inspect --repo <repo>` for the current exported list.

Common exported models include:

- `CNN`
- `CAN`
- `DeepSEA`
- `Beluga`
- `DanQ`
- `Basset`
- `BPNet`
- `scBasset`
- `ResidualNet`
- `Enformer`
- `ChromBPNet`
- `Borzoi`
- foundation adapters such as `DNABERTAdapter`, `DNABERT2Adapter`, `NucleotideTransformerAdapter`, `GPNAdapter`, `Evo2Adapter`

## Outputs

Typical files:

- Main log: `<output_dir>/DGS.log`.
- Checkpoints: configured `train.checkpoint_dir`, usually `checkpoints/best_model.pt`.
- Evaluation metrics: `<output_dir>/metrics.csv`.
- Variant predictions: `<output_dir>/variant_predictions.csv`.
- Explain/motif artifacts: `explain.output_dir`, usually `motif_results`.
- TensorBoard logs: configured `train.tensorboard_dir` when enabled.

## Troubleshooting

- Missing console command: use `python -m DGS.Main` from the repo or install with `pip install -e .`.
- Missing explain dependency: install `pip install -e ".[explain]"`; ensure `modisco` is in `PATH` for motif report generation.
- Unknown model type: inspect `DGS.Model.__all__` and `DGS.Model.get_model_zoo()`, then ensure `model.type` is exported by `DGS/Model/__init__.py`.
- Shape mismatch: verify `model.args.output_size` equals the number of scalar target tasks, and check whether the chosen model expects sequence length or task-count constructor args.
- File validation failure: run helper validation with `--check-files` and fix paths before starting long jobs.
