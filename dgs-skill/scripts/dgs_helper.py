#!/usr/bin/env python3
"""Small utilities for natural-language DGS workflows.

The helper intentionally keeps expensive DGS execution outside this script.
It creates/validates configs and inspects the installed or source-checkout API.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _add_repo_to_path(repo: Optional[str]) -> Optional[Path]:
    if not repo:
        return None
    repo_path = Path(repo).expanduser().resolve()
    if not repo_path.exists():
        raise SystemExit(f"DGS repo path does not exist: {repo_path}")
    sys.path.insert(0, str(repo_path))
    return repo_path


def _load_config_module(repo: Optional[str]):
    _add_repo_to_path(repo)
    try:
        from DGS.Config import (  # type: ignore
            ConfigManager,
            complete_configs,
            get_config_schema_reference,
            minimal_config,
            validate_config,
        )
    except Exception as exc:  # pragma: no cover - user environment dependent
        raise SystemExit(f"Unable to import DGS.Config. Install DGS or pass --repo. Error: {exc}") from exc
    return ConfigManager, complete_configs, get_config_schema_reference, minimal_config, validate_config


def _deep_update(target: Dict[str, Any], dotted_path: str, value: Any) -> None:
    current = target
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _parse_target_task(raw: str) -> Dict[str, Any]:
    """Parse task_name:file_type:file_path plus optional key=value pairs.

    Example:
        binding:bed:targets.bed:task_type=binary:target_column=name
    """
    parts = raw.split(":")
    if len(parts) < 3:
        raise argparse.ArgumentTypeError(
            "--target must look like task_name:file_type:file_path[:key=value...]"
        )
    task = {
        "task_name": parts[0],
        "file_type": parts[1].lower(),
        "file_path": parts[2],
    }
    for extra in parts[3:]:
        if "=" not in extra:
            raise argparse.ArgumentTypeError(f"Target option lacks '=': {extra}")
        key, value = extra.split("=", 1)
        task[key] = _coerce_scalar(value)
    return task


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _json_dump(data: Any, path: Optional[str]) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if path:
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"Wrote {output_path}")
    else:
        print(text, end="")


def _static_model_info(package_file: str) -> Dict[str, Any]:
    """Read model exports from source without importing torch-backed modules."""
    model_init = Path(package_file).resolve().parent / "Model" / "__init__.py"
    if not model_init.exists():
        return {}

    try:
        tree = ast.parse(model_init.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"static_model_parse_error": str(exc)}

    info: Dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                try:
                    value = ast.literal_eval(node.value)
                except Exception:
                    value = None
                if isinstance(value, list):
                    excluded_exports = {"GLM_MODEL_PRESETS", "MODEL_ZOO"}
                    info["exported_model_classes"] = sorted(
                        item for item in value
                        if isinstance(item, str)
                        and item[:1].isupper()
                        and item not in excluded_exports
                    )
            elif isinstance(target, ast.Name) and target.id == "MODEL_ZOO":
                try:
                    value = ast.literal_eval(node.value)
                except Exception:
                    value = None
                if isinstance(value, dict):
                    info["model_zoo"] = sorted(str(key) for key in value.keys())
    return info


def cmd_inspect(args: argparse.Namespace) -> None:
    repo_path = _add_repo_to_path(args.repo)
    try:
        import DGS  # type: ignore
    except Exception as exc:  # pragma: no cover - user environment dependent
        raise SystemExit(f"Unable to import DGS. Install DGS or pass --repo. Error: {exc}") from exc

    info: Dict[str, Any] = {
        "repo": str(repo_path) if repo_path else None,
        "package_file": str(Path(DGS.__file__).resolve()),
        "version": getattr(DGS, "__version__", None),
        "console_command": shutil.which("dgs"),
        "cli_commands": ["config", "run", "train", "evaluate", "explain", "predict"],
    }

    try:
        from DGS import Model  # type: ignore

        model_names = sorted(
            name for name in dir(Model)
            if not name.startswith("_") and isinstance(getattr(Model, name), type)
        )
        info["exported_model_classes"] = model_names
        if hasattr(Model, "get_model_zoo"):
            zoo = Model.get_model_zoo()
            if isinstance(zoo, dict):
                info["model_zoo"] = sorted(zoo)
    except Exception as exc:
        info["model_inspection_error"] = str(exc)
        static_info = _static_model_info(info["package_file"])
        info.update(static_info)

    try:
        _, _, get_schema, _, _ = _load_config_module(None)
        info["schema_modes"] = get_schema()["modes"]["allowed"]
    except Exception as exc:
        info["schema_error"] = str(exc)

    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        print(f"DGS version: {info.get('version')}")
        print(f"Package file: {info.get('package_file')}")
        print(f"Console command: {info.get('console_command') or 'not found'}")
        print(f"CLI commands: {', '.join(info['cli_commands'])}")
        models = info.get("model_zoo") or info.get("exported_model_classes") or []
        print(f"Models ({len(models)}): {', '.join(models[:80])}")


def cmd_new_config(args: argparse.Namespace) -> None:
    _, complete_configs, _, minimal_config, validate_config = _load_config_module(args.repo)
    config = copy.deepcopy(complete_configs if args.example == "full" else minimal_config)

    if args.mode:
        config["modes"] = args.mode
    if args.device:
        config["device"] = args.device
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.genome:
        _deep_update(config, "data.genome_path", args.genome)
    if args.intervals:
        _deep_update(config, "data.intervals_path", args.intervals)
    if args.target:
        _deep_update(config, "data.target_tasks", args.target)
        if not args.model_output_size:
            _deep_update(config, "model.args.output_size", len(args.target))
    if args.vcf:
        _deep_update(config, "predict.vcf_path", args.vcf)
    if args.model:
        _deep_update(config, "model.type", args.model)
    if args.model_output_size is not None:
        _deep_update(config, "model.args.output_size", args.model_output_size)
    if args.batch_size is not None:
        _deep_update(config, "data.batch_size", args.batch_size)
    if args.max_epochs is not None:
        _deep_update(config, "train.max_epochs", args.max_epochs)
    if args.checkpoint:
        _deep_update(config, "evaluate.checkpoint_path", args.checkpoint)
    if args.cpu:
        config["device"] = "cpu"

    if args.validate:
        validate_config(config, check_files=args.check_files)
        print("Config validation OK", file=sys.stderr)

    _json_dump(config, args.output)


def cmd_validate_config(args: argparse.Namespace) -> None:
    ConfigManager, _, _, _, validate_config = _load_config_module(args.repo)
    manager = ConfigManager()
    config = manager.load_config(args.config)
    validate_config(config, check_files=args.check_files)
    print("Config validation OK")
    notes = manager.get_compat_notes()
    if notes:
        print("Compatibility notes:")
        for note in notes:
            print(f"- {note}")


def cmd_command(args: argparse.Namespace) -> None:
    exe = "dgs" if shutil.which("dgs") else "python -m DGS.Main"
    parts: List[str] = [exe]
    if args.gpu is not None:
        parts.extend(["--gpu", str(args.gpu)])
    if args.seed is not None:
        parts.extend(["--seed", str(args.seed)])
    if args.verbose is not None:
        parts.extend(["--verbose", str(args.verbose)])
    if args.no_benchmark:
        parts.append("--no-benchmark")
    parts.extend([args.command, "--config", str(Path(args.config).expanduser())])
    print(" ".join(parts))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DGS skill helper")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect local DGS package")
    inspect_parser.add_argument("--repo", help="Path to a DeepGeSeq source checkout")
    inspect_parser.add_argument("--json", action="store_true", help="Emit JSON")
    inspect_parser.set_defaults(func=cmd_inspect)

    config_parser = subparsers.add_parser("new-config", help="Create a DGS JSON config")
    config_parser.add_argument("--repo", help="Path to a DeepGeSeq source checkout")
    config_parser.add_argument("--example", choices=["minimal", "full"], default="full")
    config_parser.add_argument("--output", required=True, help="Output JSON path")
    config_parser.add_argument("--mode", action="append", choices=["train", "evaluate", "explain", "predict"])
    config_parser.add_argument("--device", choices=["cpu", "cuda"], help="DGS device field")
    config_parser.add_argument("--cpu", action="store_true", help="Set device to CPU")
    config_parser.add_argument("--output-dir", help="DGS output_dir")
    config_parser.add_argument("--genome", help="Reference FASTA path")
    config_parser.add_argument("--intervals", help="Intervals BED path")
    config_parser.add_argument("--target", action="append", type=_parse_target_task, help="task_name:file_type:file_path[:key=value...]")
    config_parser.add_argument("--vcf", help="VCF path for predict mode")
    config_parser.add_argument("--model", help="DGS model type")
    config_parser.add_argument("--model-output-size", type=int)
    config_parser.add_argument("--batch-size", type=int)
    config_parser.add_argument("--max-epochs", type=int)
    config_parser.add_argument("--checkpoint", help="evaluate.checkpoint_path")
    config_parser.add_argument("--validate", action="store_true")
    config_parser.add_argument("--check-files", action="store_true")
    config_parser.set_defaults(func=cmd_new_config)

    validate_parser = subparsers.add_parser("validate-config", help="Validate a DGS JSON config")
    validate_parser.add_argument("--repo", help="Path to a DeepGeSeq source checkout")
    validate_parser.add_argument("--config", required=True)
    validate_parser.add_argument("--check-files", action="store_true")
    validate_parser.set_defaults(func=cmd_validate_config)

    command_parser = subparsers.add_parser("command", help="Print a DGS run command")
    command_parser.add_argument("--command", choices=["run", "train", "evaluate", "explain", "predict"], required=True)
    command_parser.add_argument("--config", required=True)
    command_parser.add_argument("--gpu", type=int)
    command_parser.add_argument("--seed", type=int)
    command_parser.add_argument("--verbose", type=int, choices=[0, 1, 2])
    command_parser.add_argument("--no-benchmark", action="store_true")
    command_parser.set_defaults(func=cmd_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
