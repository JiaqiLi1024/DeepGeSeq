---
name: dgs
description: Natural-language control layer for DeepGeSeq (DGS), a deep learning toolkit for genomic sequence analysis. Use when the user asks to call DGS/DeepGeSeq from conversation, create or validate DGS JSON configs, run DGS CLI modes (`config`, `run`, `train`, `evaluate`, `explain`, `predict`), train/evaluate sequence models from FASTA/BED/BigWig data, interpret models with motif/attribution workflows, predict variant effects from VCF, use DGS Python APIs, inspect supported DGS models/data loaders, or handle Chinese requests such as DGS包调用、自然语言运行DGS、训练模型、评估模型、变异效应预测、解释模型、生成配置文件、检查DGS配置.
---

# DGS

Translate a user's natural-language genomics task into a concrete DeepGeSeq action. Prefer the DGS CLI for complete workflows and the Python API for notebooks, exploratory analysis, custom model objects, profile models, sequence design, or low-level data operations.

## Quick Workflow

1. Identify the intent: `config`, `run`, `train`, `evaluate`, `explain`, `predict`, Python API, or inspection.
2. Extract required inputs from the request: reference FASTA, intervals BED, target BED/BigWig tasks, VCF, checkpoint, model type, output directory, device, batch size, epochs, and target index.
3. Inspect the local DGS package when the exact API/model/config surface matters:

```bash
python /path/to/DeepGeSeq/DGS/skills/dgs/scripts/dgs_helper.py inspect --repo /path/to/DeepGeSeq
```

4. Create or edit a DGS config instead of assembling long CLI arguments. For new configs, use:

```bash
python /path/to/DeepGeSeq/DGS/skills/dgs/scripts/dgs_helper.py new-config --repo /path/to/DeepGeSeq --example full --output config.json
```

5. Validate before expensive work:

```bash
python /path/to/DeepGeSeq/DGS/skills/dgs/scripts/dgs_helper.py validate-config --repo /path/to/DeepGeSeq --config config.json --check-files
```

6. Run with global flags before the subcommand:

```bash
dgs --gpu -1 --seed 42 --no-benchmark train --config config.json
```

If DGS is not installed as a console command, run from the repository with:

```bash
python -m DGS.Main --gpu -1 train --config config.json
```

## Intent Mapping

- "生成配置", "create config", "template": run `dgs config --example minimal|full --output <file>` or use `dgs_helper.py new-config` when paths/modes must be filled in.
- "完整流程", "run pipeline", "train then evaluate/predict": create config with `modes`, validate, then run `dgs run --config <file>`.
- "训练", "train model": set `modes: ["train"]` or use `dgs train --config <file>`.
- "评估", "evaluate/test metrics": require target data and either prior training in the same run or `evaluate.checkpoint_path`; run `dgs evaluate --config <file>`.
- "解释", "motif", "attribution", "DeepLIFT", "integrated gradients": require FASTA, intervals, model config/checkpoint context, and optional explain dependencies; run `dgs explain --config <file>`.
- "变异效应", "variant effect", "VCF", "predict variants": require FASTA and VCF; set `predict.vcf_path`; run `dgs predict --config <file>`.
- "Python API", "notebook", "custom model": import DGS modules directly and build loaders/trainers in code.
- "支持哪些模型/API": inspect DGS with the helper and read `references/dgs-api.md`.

## Config Rules

Use JSON config as the handoff between natural language and DGS. Preserve user-provided paths exactly unless converting to absolute paths is useful for reproducibility.

Required by most modes:

- `data.genome_path`: reference FASTA.
- `model.type`: exported DGS model class, commonly `CNN`, `DeepSEA`, `DanQ`, `Basset`, `BPNet`, `scBasset`, `ResidualNet`, `Enformer`, `ChromBPNet`, or `Borzoi`.
- `model.args`: constructor kwargs; keep `output_size` aligned to target-task count for supervised scalar outputs.

Required for train/evaluate:

- `data.intervals_path`: BED-like intervals.
- `data.target_tasks`: list of `{task_name, file_path, file_type}` where `file_type` is `bed` or `bigwig`.
- `train.optimizer` and `train.criterion` for training. Use nested `params`.

Required for predict:

- `predict.vcf_path`.
- `predict.sequence_length`.

Useful runtime choices:

- Use `--gpu -1` or `device: "cpu"` for CPU runs and smoke tests.
- Use low `train.max_epochs`, low `data.batch_size`, and `data.num_workers: 0` for quick validation.
- Use `data.train_test_split: "chromosome_split"` with `test_chroms`/`val_chroms` when the user asks for chromosome holdout.

## Python API

Use Python API when the user asks for code, notebooks, custom torch models, in-memory data, profile predictions, sequence design, or diagnostics that the CLI does not expose.

Common imports:

```python
from DGS.Data import build_supervised_dataloaders, build_sequence_dataloader, build_profile_dataloaders
from DGS.DL.Trainer import Trainer
from DGS.DL.Predict import vep_centred_from_files, vep_centred_on_ds
from DGS.DL.Explain import calculate_attributions, motif_enrich
from DGS.DL.Design import gradient_ascent_sequence_design, greedy_ism_sequence_design
from DGS.Model import CNN, DeepSEA, DanQ, Basset, BPNet, ChromBPNet, Borzoi
```

Read `references/dgs-api.md` when using direct APIs, profile models, sequence design, foundation adapters, data loaders, or output-file conventions.

## Execution Discipline

- Confirm missing biological inputs before running expensive workflows; do not invent FASTA/BED/BigWig/VCF paths.
- Prefer config validation with `--check-files` before training, explaining, or predicting.
- For large training or motif-discovery jobs, show the exact command and expected outputs before running unless the user already asked to execute.
- After running, report created artifacts such as `DGS.log`, `metrics.csv`, `variant_predictions.csv`, checkpoints, motif output directories, and any validation errors.
- If optional dependencies are missing, explain the missing extra: `pip install -e ".[explain]"` for Captum/tangermeme Python dependencies; `modisco` must still be available in `PATH` for motif reports.
