"""Configuration schema and compatibility helpers for DGS.

This module defines dataclass-backed configuration structures and a
`ConfigManager` helper for loading, normalizing, and exporting runtime config.

Compatibility behavior:
    - Legacy optimizer/loss keys such as `train.optimizer.lr` are normalized to
      `train.optimizer.params.lr`.
    - Normalization is non-destructive for caller input because updates are
      applied to a deep-copied dictionary.
"""

import json
import logging
import copy
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
try:
    from typing import Literal  # Python >= 3.8
except ImportError:
    try:
        from typing_extensions import Literal  # Python 3.7 fallback
    except ImportError:
        class _Literal:
            def __getitem__(self, _):
                return Any
        Literal = _Literal()  # type: ignore
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("dgs.config")

@dataclass
class DataConfig:
    """
    Configuration for genomic data processing and dataset management.

    This class defines settings for:
    - Genome and interval data sources
    - Target task specifications
    - Data splitting strategies
    - Dataset parameters

    Used by:
    - preprocess_data_for_train: Training data preparation
    - preprocess_data_for_evaluate: Evaluation data preparation
    - preprocess_data_for_explain: Model interpretation data
    - preprocess_data_for_predict: Prediction data preparation

    Attributes:
        genome_path (str): Path to reference genome FASTA file
        intervals_path (str): Path to genomic intervals BED file
        target_tasks (List[Dict]): List of target task specifications
        train_test_split (str): Data split strategy ('random_split' or 'chromosome_split')
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of data for validation
        test_chroms (List[str]): Chromosomes to use for testing
        val_chroms (List[str]): Chromosomes to use for validation
        batch_size (int): Batch size for data loading
        strand_aware (bool): Whether to consider DNA strand information
        sequence_length (int): Length of input sequences
        loader_mode (str): Sequence loading mode, either 'streaming' or 'cached'
        random_state (Optional[int]): Optional seed for random data splitting
        num_workers (int): DataLoader worker count
        pin_memory (bool): DataLoader pinned-memory flag
        persistent_workers (bool): Keep workers alive between epochs
        prefetch_factor (Optional[int]): DataLoader prefetch factor for workers
    """
    # Dataset settings
    genome_path: str = ""  # Path to genome fasta file
    intervals_path: str = ""  # Path to intervals bed file
    
    # Target settings
    target_tasks: List[Dict[str, Any]] = field(default_factory=lambda: [{
        'task_name': 'default',
        'file_path': '',
        'file_type': 'bigwig',  # or 'bed'
        'bin_size': None,
        'aggfunc': 'mean'
    }])
    
   # Data splitting settings
    train_test_split: Literal["random_split", "chromosome_split"] = "random_split"
    test_size: float = 0.2
    val_size: float = 0.2
    test_chroms: List[str] = field(default_factory=lambda: ["chr8"])
    val_chroms: List[str] = field(default_factory=lambda: ["chr7"])
    
    # Dataset settings
    batch_size: int = 4
    strand_aware: bool = True
    sequence_length: int = 1000
    loader_mode: Literal["streaming", "cached"] = "streaming"
    random_state: Optional[int] = None
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None


@dataclass
class ModelConfig:
    """
    Configuration for neural network model architecture.

    This class defines the structure and parameters of the deep learning model
    used for genomic sequence analysis.

    Attributes:
        type (str): Model architecture type (e.g., 'CNN', 'Transformer')
        input_size (int): Length of input sequences
        output_size (int): Number of output features/tasks
        hidden_size (int): Size of hidden layers
        num_layers (int): Number of layers in the model
        dropout (float): Dropout rate for regularization
        activation (str): Activation function type
        normalization (str): Normalization layer type
    """
    type: str = "CNN"  # Model type (e.g., CNN, Transformer)
    input_size: int = 1000  # Input sequence length
    output_size: int = 1  # Number of output features
    hidden_size: int = 128  # Hidden layer size
    num_layers: int = 3  # Number of layers
    dropout: float = 0.1  # Dropout rate
    activation: str = "ReLU"  # Activation function
    normalization: str = "BatchNorm1d"  # Normalization layer

@dataclass
class TrainerConfig:
    """
    Configuration for model training and optimization.

    This class defines all parameters related to model training, including:
    - Optimizer settings and hyperparameters
    - Loss function configuration
    - Training control parameters
    - Checkpoint and logging settings

    Used by:
        execute_dgs_train: Main training execution function

    Attributes:
        optimizer (Dict): Optimizer type and parameters
        criterion (Dict): Loss function type and parameters
        patience (int): Early stopping patience
        max_epochs (int): Maximum training epochs
        clip_grad_norm (bool): Whether to clip gradients
        max_norm (float): Maximum gradient norm
        checkpoint_dir (str): Directory for saving checkpoints
        best_model_path (str): Path for saving best model
        resume (bool): Whether to resume training
        resume_model_name (str): Name of checkpoint to resume from
        use_tensorboard (bool): Whether to use TensorBoard
        tensorboard_dir (str): Directory for TensorBoard logs
        use_amp (bool): Whether to enable CUDA automatic mixed precision
        amp_dtype (str): AMP autocast dtype name
        non_blocking (bool): Whether to use non-blocking device transfers
    """

    # Optimizer settings
    optimizer: Dict[str, Any] = field(default_factory=lambda: {
        "type": "Adam",
        "params": {
            "lr": 1e-3,
            "weight_decay": 0
        }
    })

    # Loss function settings
    criterion: Dict[str, Any] = field(default_factory=lambda: {
        "type": "MSELoss",
        "params": {}
    })

    # early stopping settings
    patience: int = 10
    max_epochs: int = 500
    
    # Gradient settings
    clip_grad_norm: bool = False
    max_norm: float = 10.0
    
    # Output settings
    checkpoint_dir: str = "checkpoints"
    best_model_path: str = "checkpoints/best_model.pt"

    # resume settings
    resume: bool = False
    resume_model_name: str = "best_model.pt"

    # Tensorboard settings
    use_tensorboard: bool = True
    tensorboard_dir: str = "tensorboard"
    use_amp: bool = False
    amp_dtype: str = "float16"
    non_blocking: bool = False
    
    # # Evaluation settings
    # evaluate_training: bool = True
    # metric_sample: int = 100
    # item_sample: int = 5000
    
@dataclass
class EvaluateConfig:
    """
    Configuration for model evaluation and performance assessment.

    This class defines settings for evaluating trained models on test data
    and computing various performance metrics.

    Used by:
        execute_dgs_evaluate: Model evaluation function

    Attributes:
        split (str): Data split to evaluate on ('train', 'val', 'test')
        metrics (List[str]): List of metrics to compute
        output_dir (str): Directory for evaluation results
        save_predictions (bool): Whether to save model predictions
        checkpoint_path (Optional[str]): Optional checkpoint file for evaluate-only runs
    """
    split: Literal["train", "val", "test"] = "test"
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "auc"
    ])
    output_dir: str = "evaluation_results"
    save_predictions: bool = True
    checkpoint_path: Optional[str] = None

@dataclass
class ExplainConfig:
    """
    Configuration for model interpretation and visualization.

    This class defines parameters for generating explanations of model
    predictions and identifying important sequence patterns.

    Attributes:
        target (int): Target task index for interpretation
        output_dir (str): Directory for saving interpretation results
        max_seqlets (int): Maximum number of sequence elements to analyze
        method (str): Attribution method for contribution scores
        baseline (str): Baseline strategy for Captum methods
        n_steps (int): Number of steps for Integrated Gradients
        internal_batch_size (Optional[int]): Captum internal batch size
    """
    target: int = 0
    output_dir: str = "motif_results"
    max_seqlets: int = 2000
    method: str = "deeplift_shap"
    baseline: str = "zero"
    n_steps: int = 50
    internal_batch_size: Optional[int] = None

@dataclass
class PredictConfig:
    """
    Configuration for variant effect prediction.

    This class defines settings for predicting the effects of genetic
    variants on sequence function.

    Attributes:
        vcf_path (str): Path to VCF file containing variants
        sequence_length (int): Length of sequence context around variants
        metric_func (str): Function for computing variant effects
        mean_by_tasks (bool): Whether to average predictions across tasks
        batch_size (Optional[int]): Batch size for batched prediction; None or
            <=1 uses the legacy per-variant path
        num_workers (int): Optional dataloader workers for batched prediction
        pin_memory (bool): Optional pinned memory for batched prediction
        persistent_workers (bool): Keep prediction workers alive between batches
        prefetch_factor (Optional[int]): DataLoader prefetch factor when workers are used
    """
    vcf_path: str = ""
    sequence_length: int = 1000
    metric_func: str = "diff"
    mean_by_tasks: bool = True
    batch_size: Optional[int] = None
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None

@dataclass
class DgsConfig:
    """
    Main configuration class for DGS toolkit.

    This class serves as the central configuration container, combining
    all component-specific configurations and global settings.

    Attributes:
        modes (List[str]): List of analysis modes to run
        device (str): Computation device ('cuda' or 'cpu')
        output_dir (str): Main output directory
        data (DataConfig): Data processing configuration
        model (Dict): Model architecture configuration
        train (TrainerConfig): Training configuration
        evaluate (EvaluateConfig): Evaluation configuration
        explain (ExplainConfig): Interpretation configuration
        predict (PredictConfig): Prediction configuration
    """
    # Required configurations
    modes: List[Literal["train", "evaluate", "explain", "predict"]] = field(
        default_factory=lambda: ["train", "evaluate"]
    )
    device: str = "cuda"
    output_dir: str = "output"
    
    # Component configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: Dict[str, Any] = field(default_factory=lambda: {
        "type": "CNN",
        "args": {"output_size": 1}
    })
    train: TrainerConfig = field(default_factory=TrainerConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)

class ConfigError(Exception):
    """
    Exception class for configuration-related errors.

    This class is used to raise errors when:
    - Required configuration parameters are missing
    - Parameter values are invalid
    - Configuration file cannot be loaded
    """


SUPPORTED_MODES = ("train", "evaluate", "explain", "predict")

CONFIG_SCHEMA_REFERENCE: Dict[str, Any] = {
    "modes": {
        "type": "list[str]",
        "required": False,
        "default": ["train", "evaluate"],
        "allowed": list(SUPPORTED_MODES),
    },
    "device": {
        "type": "str",
        "required": False,
        "default": "cuda",
        "allowed": ["cuda", "cpu", "cuda:<index>"],
    },
    "output_dir": {"type": "str", "required": False, "default": "output"},
    "data": {
        "required_for": ["train", "evaluate", "explain", "predict"],
        "fields": {
            "genome_path": {"type": "str", "required": True, "description": "Reference FASTA path."},
            "intervals_path": {"type": "str", "required_for": ["train", "evaluate", "explain"]},
            "target_tasks": {
                "type": "list[dict]",
                "required_for": ["train", "evaluate"],
                "item_required": ["task_name", "file_path", "file_type"],
                "file_type": ["bed", "bigwig"],
            },
            "loader_mode": {
                "type": "str",
                "required": False,
                "default": "streaming",
                "allowed": ["streaming", "cached"],
            },
            "train_test_split": {
                "type": "str",
                "required": False,
                "default": "random_split",
                "allowed": ["random_split", "chromosome_split"],
            },
            "batch_size": {"type": "int", "required": False, "default": 4},
            "num_workers": {"type": "int", "required": False, "default": 0},
            "pin_memory": {"type": "bool", "required": False, "default": False},
            "persistent_workers": {"type": "bool", "required": False, "default": False},
            "prefetch_factor": {"type": "int | null", "required": False, "default": None},
        },
    },
    "model": {
        "required_for": ["train", "evaluate", "explain", "predict"],
        "fields": {
            "type": {"type": "str", "required": True, "example": "CNN"},
            "args": {"type": "dict", "required": False, "default": {}},
        },
    },
    "train": {
        "required_for": ["train"],
        "fields": {
            "optimizer": {
                "type": "dict",
                "required": True,
                "fields": {"type": "str", "params": "dict"},
            },
            "criterion": {
                "type": "dict",
                "required": True,
                "fields": {"type": "str", "params": "dict"},
            },
            "use_amp": {"type": "bool", "required": False, "default": False},
            "amp_dtype": {
                "type": "str",
                "required": False,
                "default": "float16",
                "allowed": ["float16", "fp16", "bfloat16", "bf16"],
            },
            "non_blocking": {"type": "bool", "required": False, "default": False},
            "resume": {"type": "bool", "required": False, "default": False},
            "resume_model_name": {"type": "str", "required": False, "default": "best_model.pt"},
        },
    },
    "evaluate": {
        "required_for": ["evaluate"],
        "fields": {
            "checkpoint_path": {"type": "str | null", "required": False, "default": None},
            "split": {"type": "str", "required": False, "default": "test"},
        },
    },
    "explain": {
        "required_for": ["explain"],
        "fields": {
            "target": {"type": "int", "required": False, "default": 0},
            "method": {
                "type": "str",
                "required": False,
                "default": "deeplift_shap",
                "allowed": ["deeplift_shap", "deeplift", "integrated_gradients", "ig"],
            },
        },
    },
    "predict": {
        "required_for": ["predict"],
        "fields": {
            "vcf_path": {"type": "str", "required": True},
            "sequence_length": {"type": "int", "required": False, "default": 1000},
            "batch_size": {"type": "int | null", "required": False, "default": None},
        },
    },
}


def get_config_schema_reference() -> Dict[str, Any]:
    """Return a copy of the documented DGS configuration schema reference."""
    return copy.deepcopy(CONFIG_SCHEMA_REFERENCE)


def _as_mapping(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return a nested mapping or raise a user-facing config error."""
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Config section '{key}' must be a dictionary.")
    return value


def _require_value(config: Dict[str, Any], path: str) -> Any:
    """Fetch a required dotted config path and ensure it is not empty."""
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(f"Missing required config field: '{path}'.")
        current = current[part]
    if current is None or current == "":
        raise ConfigError(f"Missing required config field: '{path}'.")
    return current


def _validate_existing_file(path_value: Any, path_name: str) -> None:
    """Validate an optional file path when strict file checks are requested."""
    path = Path(path_value)
    if not path.exists():
        raise ConfigError(f"Config field '{path_name}' points to a missing file: {path}")
    if not path.is_file():
        raise ConfigError(f"Config field '{path_name}' must point to a file: {path}")


def _validate_component(component: Dict[str, Any], path: str) -> None:
    """Validate optimizer or criterion component dictionaries."""
    if not isinstance(component, dict):
        raise ConfigError(f"Config section '{path}' must be a dictionary.")
    if not component.get("type"):
        raise ConfigError(f"Missing required config field: '{path}.type'.")
    params = component.get("params", {})
    if not isinstance(params, dict):
        raise ConfigError(f"Config field '{path}.params' must be a dictionary.")


def validate_config(config: Dict[str, Any], check_files: bool = False) -> None:
    """Validate the DGS runtime config before expensive work starts.

    Args:
        config: Normalized configuration dictionary.
        check_files: Whether referenced input files must already exist.

    Raises:
        ConfigError: If required sections, fields, values, or files are invalid.
    """
    if not isinstance(config, dict):
        raise ConfigError("Config must be a dictionary.")

    modes = config.get("modes", ["train", "evaluate"])
    if not isinstance(modes, list) or not modes:
        raise ConfigError("Config field 'modes' must be a non-empty list.")
    unknown_modes = sorted(set(modes) - set(SUPPORTED_MODES))
    if unknown_modes:
        raise ConfigError(
            f"Unsupported mode(s): {unknown_modes}. Supported modes are: {list(SUPPORTED_MODES)}."
        )

    data_cfg = _as_mapping(config, "data")
    model_cfg = _as_mapping(config, "model")

    if not model_cfg.get("type"):
        raise ConfigError("Missing required config field: 'model.type'.")
    if "args" in model_cfg and not isinstance(model_cfg["args"], dict):
        raise ConfigError("Config field 'model.args' must be a dictionary if provided.")

    if any(mode in modes for mode in ("train", "evaluate", "explain", "predict")):
        genome_path = _require_value(config, "data.genome_path")
        if check_files:
            _validate_existing_file(genome_path, "data.genome_path")

    if any(mode in modes for mode in ("train", "evaluate", "explain")):
        intervals_path = _require_value(config, "data.intervals_path")
        if check_files:
            _validate_existing_file(intervals_path, "data.intervals_path")

    loader_mode = data_cfg.get("loader_mode", "streaming")
    if loader_mode not in {"streaming", "cached"}:
        raise ConfigError("Config field 'data.loader_mode' must be 'streaming' or 'cached'.")

    split = data_cfg.get("train_test_split", "random_split")
    if split not in {"random_split", "chromosome_split"}:
        raise ConfigError(
            "Config field 'data.train_test_split' must be 'random_split' or 'chromosome_split'."
        )

    if any(mode in modes for mode in ("train", "evaluate")):
        target_tasks = data_cfg.get("target_tasks")
        if not isinstance(target_tasks, list) or not target_tasks:
            raise ConfigError("Config field 'data.target_tasks' must be a non-empty list.")
        for idx, task in enumerate(target_tasks):
            if not isinstance(task, dict):
                raise ConfigError(f"Config field 'data.target_tasks[{idx}]' must be a dictionary.")
            missing = sorted({"task_name", "file_path", "file_type"} - set(task))
            if missing:
                raise ConfigError(
                    f"Config field 'data.target_tasks[{idx}]' is missing: {missing}."
                )
            file_type = str(task["file_type"]).lower()
            if file_type not in {"bed", "bigwig"}:
                raise ConfigError(
                    f"Config field 'data.target_tasks[{idx}].file_type' must be 'bed' or 'bigwig'."
                )
            if check_files:
                _validate_existing_file(task["file_path"], f"data.target_tasks[{idx}].file_path")

    if "train" in modes:
        train_cfg = _as_mapping(config, "train")
        _validate_component(train_cfg.get("optimizer"), "train.optimizer")
        _validate_component(train_cfg.get("criterion"), "train.criterion")
        if train_cfg.get("resume"):
            checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
            resume_name = train_cfg.get("resume_model_name", "best_model.pt")
            resume_path = checkpoint_dir / resume_name
            if check_files and not resume_path.exists():
                raise ConfigError(f"Training resume checkpoint not found: {resume_path}")

    if "predict" in modes:
        predict_cfg = _as_mapping(config, "predict")
        vcf_path = _require_value(config, "predict.vcf_path")
        if check_files:
            _validate_existing_file(vcf_path, "predict.vcf_path")
        sequence_length = int(predict_cfg.get("sequence_length", 1000))
        if sequence_length <= 0:
            raise ConfigError("Config field 'predict.sequence_length' must be a positive integer.")

    if "evaluate" in modes:
        evaluate_cfg = config.get("evaluate", {})
        if evaluate_cfg is not None and not isinstance(evaluate_cfg, dict):
            raise ConfigError("Config section 'evaluate' must be a dictionary.")
        checkpoint_path = (evaluate_cfg or {}).get("checkpoint_path")
        if check_files and checkpoint_path:
            _validate_existing_file(checkpoint_path, "evaluate.checkpoint_path")


def _normalize_legacy_component_params(component_cfg: Dict[str, Any], name: str) -> List[str]:
    """Normalize legacy component keys into a nested `params` object.

    Args:
        component_cfg: Component config dictionary (for example optimizer config).
        name: Human-readable component path used in compatibility notes.

    Returns:
        List of compatibility notes describing applied conversions.

    Raises:
        ConfigError: If `params` exists but is not a dictionary.
    """
    notes: List[str] = []
    params = component_cfg.get("params")
    if params is None:
        params = {}
        component_cfg["params"] = params
    elif not isinstance(params, dict):
        raise ConfigError(f"{name}.params must be a dictionary if provided")

    reserved = {"type", "params"}
    legacy_keys = [key for key in component_cfg.keys() if key not in reserved]
    if legacy_keys:
        for key in legacy_keys:
            params.setdefault(key, component_cfg.pop(key))
        notes.append(
            f"Normalized legacy config keys for '{name}' into '{name}.params': {legacy_keys}"
        )

    return notes


def normalize_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Normalize legacy config layouts into the current schema.

    Args:
        config: Raw user configuration.

    Returns:
        Tuple `(normalized_config, notes)` where `notes` records applied
        compatibility transformations.
    """
    normalized = copy.deepcopy(config)
    notes: List[str] = []

    train_cfg = normalized.get("train")
    if isinstance(train_cfg, dict):
        optimizer_cfg = train_cfg.get("optimizer")
        if isinstance(optimizer_cfg, dict):
            notes.extend(_normalize_legacy_component_params(optimizer_cfg, "train.optimizer"))

        criterion_cfg = train_cfg.get("criterion")
        if isinstance(criterion_cfg, dict):
            notes.extend(_normalize_legacy_component_params(criterion_cfg, "train.criterion"))

    return normalized, notes

class ConfigManager:
    """
    Configuration management system for DGS.

    This class provides methods for:
    - Loading configurations from files or dictionaries
    - Saving configurations to files
    - Generating example configurations
    - Updating existing configurations
    - Validating configuration parameters
    """
    
    def __init__(self):
        """Initialize an empty configuration manager state."""
        self._config: Dict[str, Any] = {}
        self._compat_notes: List[str] = []
        
    def load_config(self, config: Union[str, Dict[str, Any], Path]) -> Dict[str, Any]:
        """
        Load configuration from a file or dictionary.

        Args:
            config: Configuration source (file path or dictionary)

        Returns:
            Dict[str, Any]: Loaded configuration dictionary

        Raises:
            FileNotFoundError: If config file not found
            ConfigError: If configuration is invalid
        """
        if isinstance(config, (str, Path)):
            config = self._load_from_file(config)

        normalized_config, compat_notes = normalize_config(config)
        self._config = normalized_config
        self._compat_notes = compat_notes
        return normalized_config
    
    def _load_from_file(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from a JSON file.

        Args:
            path: JSON file path.

        Returns:
            Parsed configuration dictionary.

        Raises:
            FileNotFoundError: If `path` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path) as f:
            return json.load(f)
    
    def save_config(self, path: Union[str, Path]):
        """
        Save current configuration to a file.

        Args:
            path: Path to save configuration file

        Side effects:
            Writes JSON content to `path`.
        """
        with open(path, 'w') as f:
            json.dump(self._config, f, indent=4)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config

    def get_compat_notes(self) -> List[str]:
        """Get compatibility notes generated during config normalization."""
        return self._compat_notes
    
    def update_config(self, updates: Dict[str, Any]):
        """Update current configuration with a shallow dictionary merge.

        Args:
            updates: Key-value pairs to merge into current configuration.
        """
        self._config.update(updates)

    def generate_example_config(self, example: Literal["minimal", "full"], output: str):
        """
        Generate an example configuration file.

        Args:
            example: Type of example configuration ('minimal' or 'full')
                - minimal: concise demonstration config for quick orientation
                - full: schema-aligned reference config for production use
            output: Path to save the example configuration

        Raises:
            ValueError: If invalid example type specified
        """
        if example == "minimal":
            config = minimal_config
        elif example == "full":
            config = complete_configs
        else:
            raise ValueError(f"Invalid example: {example}")
        self._config, self._compat_notes = normalize_config(config)
        self.save_config(output)

# Minimal demonstration config.
# Note:
#   This example intentionally keeps fields concise and may omit nested
#   structure used by strict dataclass schema in some sections.
#   For schema-aligned usage, prefer `complete_configs` below.
minimal_config = {
    "modes" : ["train", "evaluate", "explain", "predict"],
    "device": "cuda",
    "output_dir": "Test",
    "data": {
        "genome_path": "Test/reference_grch38p13/GRCh38.p13.genome.fa.gz",
        "intervals_path": "Test/random_regions.bed",
        "target_tasks": [
            {
                "task_name": "gc_content",
                "file_path": "Test/hg38.gc5Base.bw",
                "file_type": "bigwig",                
            },
            {
                "task_name": "recomb",
                "file_path": "Test/recombAvg.bw",
                "file_type": "bigwig",                
            }
        ],
        "train_test_split": "random_split",
        "batch_size": 4,
        "loader_mode": "streaming",
        "num_workers": 0,
        "pin_memory": False
    },
    "train": {
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 0.001}
        },
        "criterion": {
            "type": "MSELoss",
            "params": {}
        },
        "use_amp": False,
        "non_blocking": False
    },
    "model": {
        "type": "CNN",
        "args":{"output_size": 2}
    },
    "explain":{"target":0},
    "predict":{"vcf_path":"Test/test.vcf",
              "sequence_length":1000,
              "batch_size": None
              }
}

# Full reference config aligned with dataclass-style schema.
complete_configs = {
    "modes": ["train", "evaluate", "predict"],
    "device": "cuda",
    "output_dir": "Test",
    
    "data": {
        "genome_path": "Test/reference_grch38p13/GRCh38.p13.genome.fa.gz",
        "intervals_path": "Test/random_regions.bed",
        "target_tasks": [
            {
                "task_name": "gc_content",
                "file_path": "Test/hg38.gc5Base.bw",
                "file_type": "bigwig"
            },
            {
                "task_name": "recomb",
                "file_path": "Test/recombAvg.bw",
                "file_type": "bigwig"
            }
        ],
        "train_test_split": "random_split",
        "test_size": 0.2,
        "val_size": 0.2,
        "test_chroms": ["chr8"],
        "val_chroms": ["chr7"],
        "strand_aware": True,
        "batch_size": 4,
        "loader_mode": "streaming",
        "random_state": 42,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None
    },
    
    "model": {
        "type": "CNN",
        "args": {"output_size": 2}
    },
    
    "train": {
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 1e-3}
        },
        "criterion": {
            "type": "MSELoss",
            "params": {}
        },
        "patience": 10,
        "max_epochs": 500,
        "checkpoint_dir": "checkpoints",
        "use_tensorboard": False,
        "tensorboard_dir": "tensorboard",
        "use_amp": False,
        "amp_dtype": "float16",
        "non_blocking": False
    },

    "evaluate": {
        "checkpoint_path": None
    },
    
    "explain": {
        "target": 0,
        "output_dir": "motif_results",
        "max_seqlets": 2000,
        "method": "deeplift_shap",
        "baseline": "zero",
        "n_steps": 50,
        "internal_batch_size": None
    },
    
    "predict": {
        "vcf_path": "Test/test.vcf",
        "sequence_length": 1000,
        "metric_func": "diff",
        "mean_by_tasks": True,
        "batch_size": None,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None
    }
}
