"""DGS CLI orchestration utilities.

Purpose:
    Bridge normalized configuration dictionaries to executable DGS workflows.

Main Responsibilities:
    - Build datasets and dataloaders for train/evaluate/explain/predict modes.
    - Initialize runtime components (model, optimizer, criterion, trainer).
    - Execute mode-specific pipelines through the `DgsCLI` coordinator.

Key Runtime Notes:
    - Compatibility keys are preserved where possible to avoid breaking configs.
    - Evaluation can run in checkpoint-only mode via `evaluate.checkpoint_path`.
    - Prediction supports optional batched inference dataloader settings.
"""

from typing import Optional, Dict, Any

import os
import logging
import numpy as np

import torch
from torch import nn
from torch import optim

from .Config import ConfigError, validate_config

logger = logging.getLogger(__name__)

# TODO: add a public HPO command entrypoint.
# Why: the architecture module already contains tuning helpers, but CLI users
# cannot invoke them directly.
# Done criteria: expose a stable `execute_dgs_hpo` path with config validation.
# def execute_dgs_hpo(search_space=None):
#     from .Architecture import hyperparameter_tune
#     hyperparameter_tune(search_space=search_space, num_samples=10, max_num_epochs=10, gpus_per_trial=1)


def _get_dataloader_kwargs(dataloader_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Filter dataloader runtime options while keeping defaults backward compatible.

    Args:
        dataloader_config: Optional dictionary from user config.

    Returns:
        Dictionary containing only supported `torch.utils.data.DataLoader`
        runtime keys with non-`None` values.
    """
    if not dataloader_config:
        return {}
    allowed_keys = {
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
    }
    return {k: v for k, v in dataloader_config.items() if k in allowed_keys and v is not None}

def preprocess_data_for_train(genome_path, target_tasks, intervals_path, 
                              train_test_split="random_split",
                              test_size=0.2, val_size=0.2,
                              test_chroms=["chr8"], val_chroms=["chr7"],
                              strand_aware=True, batch_size=4,
                              loader_mode="streaming",
                              random_state=None,
                              dataloader_config: Optional[Dict[str, Any]] = None):
    """Build train/validation/test dataloaders for supervised training.

    Args:
        genome_path (str): Path to the reference genome FASTA.
        target_tasks (list): Task definitions consumed by `Target`.
        intervals_path (str): Path to interval file used as base samples.
        train_test_split (str): Split strategy (`random_split` or `chromosome_split`).
        test_size (float): Test fraction when using random split.
        val_size (float): Validation fraction when using random split.
        test_chroms (list): Held-out chromosomes for chromosome split.
        val_chroms (list): Validation chromosomes for chromosome split.
        strand_aware (bool): Whether reverse-complement follows strand annotations.
        batch_size (int): Dataloader batch size.
        loader_mode (str): `"streaming"` for lazy FASTA reads or `"cached"`.
        random_state (int, optional): Random seed for random splitting.
        dataloader_config (dict, optional): Optional dataloader runtime overrides.

    Returns:
        tuple: `(train_loader, val_loader, test_loader)`.

    Raises:
        ValueError: If split strategy or inputs are invalid.

    Notes:
        Reads genome/interval/target files and emits progress logs.
    """
    from .Data.Loader import build_supervised_dataloaders

    dataloader_kwargs = _get_dataloader_kwargs(dataloader_config)
    train_loader, val_loader, test_loader = build_supervised_dataloaders(
        fasta_path=genome_path,
        intervals_path=intervals_path,
        target_tasks=target_tasks,
        batch_size=batch_size,
        mode=loader_mode,
        split=train_test_split,
        test_size=test_size,
        val_size=val_size,
        test_chroms=test_chroms,
        val_chroms=val_chroms,
        random_state=random_state,
        strand_aware=strand_aware,
        train_shuffle=True,
        **dataloader_kwargs,
    )
    logger.info("Created supervised dataloaders with loader_mode=%s", loader_mode)
    
    return train_loader, val_loader, test_loader

def execute_dgs_train(train_loader, validate_loader, 
                       model, optimizer, criterion, device, 
                       patience=10, max_epochs=500,
                       resume=False, resume_model_name="best_model.pt",
                       checkpoint_dir="checkpoints", 
                       use_tensorboard=True, tensorboard_dir="tensorboard", 
                       **trainer_args):
    """Run model training and return the best-checkpoint model/trainer pair.

    Args:
        train_loader (DataLoader): DataLoader for training data
        validate_loader (DataLoader): DataLoader for validation data
        model (nn.Module): Neural network model to train
        optimizer (optim.Optimizer): Optimizer for model training
        criterion (callable): Loss function
        device (str): Device to use for training ('cuda' or 'cpu')
        patience (int): Number of epochs to wait for improvement before early stopping
        max_epochs (int): Maximum number of training epochs
        resume (bool): Whether to resume training from a checkpoint
        resume_model_name (str): Name of the checkpoint file to resume from
        checkpoint_dir (str): Directory for saving checkpoints
        use_tensorboard (bool): Whether to use TensorBoard for logging
        tensorboard_dir (str): Directory for TensorBoard logs
        **trainer_args: Extra keyword arguments passed to `Trainer`.

    Returns:
        Tuple of `(model, trainer)`.

    Notes:
        - Writes checkpoints under `checkpoint_dir`.
        - Optionally writes TensorBoard logs under `tensorboard_dir`.
        - Reloads `best_model.pt` from `checkpoint_dir` before returning.
    """
    from .DL.Trainer import Trainer

    # initialize trainer
    trainer = Trainer(model, criterion, optimizer, device, 
                      checkpoint_dir=checkpoint_dir, 
                      use_tensorboard=use_tensorboard, tensorboard_dir=tensorboard_dir,
                      patience=patience, **trainer_args)
    logger.info("Initialized trainer")

    # resume from checkpoint
    if resume:
        # check if checkpoint exists
        if not os.path.exists(os.path.join(checkpoint_dir, resume_model_name)):
            raise FileNotFoundError(f"Checkpoint file {resume_model_name} not found in {checkpoint_dir}")
        trainer.load_checkpoint(os.path.join(checkpoint_dir, resume_model_name))
        logger.info("Resumed from checkpoint")
    
    # train
    trainer.train(train_loader, validate_loader, epochs=max_epochs, early_stopping=True)
    logger.info("Training completed")

    # reload best model
    trainer.load_checkpoint(os.path.join(checkpoint_dir, "best_model.pt"))
    logger.info("Reloaded best model")

    return trainer.model, trainer


def preprocess_data_for_evaluate(genome_path, target_task, intervals_path,
                                  train_test_split="random_split", test_size=0.2, test_chroms=["chr8"],
                                  strand_aware=True, batch_size=4,
                                  dataloader_config: Optional[Dict[str, Any]] = None):
    """Prepare an evaluation dataloader from genomic intervals and targets.

    Args:
        genome_path: Path to the reference genome FASTA.
        target_task: Target-task definitions used to construct labels.
        intervals_path: Path to BED-like interval file.
        train_test_split: Split strategy (`"random_split"` or `"chromosome_split"`).
        test_size: Fraction assigned to test split for random strategy.
        test_chroms: Chromosomes assigned to test split for chromosome strategy.
        strand_aware: Whether reverse-complement logic should use strand column.
        batch_size: Evaluation batch size.
        dataloader_config: Optional dataloader runtime overrides.

    Returns:
        Evaluation `DataLoader`.

    Raises:
        ValueError: If split configuration or input data is invalid.

    Side effects:
        Reads input files and emits progress logs.
    """
    from .Data.Sequence import Genome
    from .Data.Target import Target
    from .Data.Interval import Interval
    from .Data.Dataset import GenomicDataset
    from .Data.Dataset import create_dataloader
    from .Data.Sampler import chromosome_split, random_split

    # load genome and intervals
    genome = Genome(genome_path)
    intervals = Interval(intervals_path)
    logger.info("Loaded genome and intervals")

    # train test split
    if train_test_split == "chromosome_split":
        _, test_intervals = chromosome_split(intervals, test_chroms=test_chroms)
    elif train_test_split == "random_split":
        _, test_intervals = random_split(intervals, test_size=test_size)
    logger.info("Split intervals")

    # load target
    target = Target(test_intervals.data, target_task)
    logger.info("Loaded target")

    # create dataset
    test_ds = GenomicDataset(test_intervals, genome, target, strand_aware=strand_aware)
    logger.info("Created dataset")

    # create dataloader
    test_loader = create_dataloader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **_get_dataloader_kwargs(dataloader_config),
    )
    logger.info("Created dataloader")

    return test_loader

def execute_dgs_evaluate(test_loader, trainer, checkpoint_path: Optional[str] = None):
    """Evaluate a model checkpoint and compute task metrics.

    Args:
        test_loader (DataLoader): Dataloader containing evaluation samples.
        trainer (Trainer): Trainer instance used for validation.
        checkpoint_path (str, optional): Explicit checkpoint path. When omitted,
            defaults to `best_model.pt` under `trainer.checkpoint_dir`.

    Returns:
        Metrics object returned by evaluator utilities.
        For current implementation this is typically a pandas DataFrame.

    Notes:
        Reloads the selected checkpoint before running validation.
    """
    from .DL.Evaluator import calculate_classification_metrics, calculate_regression_metrics
    from .DL.Evaluator import calculate_sequence_classification_metrics, calculate_sequence_regression_metrics

    # reload model checkpoint
    checkpoint_to_load = checkpoint_path or os.path.join(trainer.checkpoint_dir, "best_model.pt")
    trainer.load_checkpoint(checkpoint_to_load, load_optimizer=False)
    logger.info("Reloaded checkpoint for evaluation: %s", checkpoint_to_load)

    # evaluate
    avg_loss, metric, predictions, targets = trainer.validate(test_loader, return_predictions=True)
    logger.info("Evaluated model")

    # calculate metrics
    if np.all(np.isin(targets, [0, 1])):
        target_task = 'classification'
    else:
        target_task = 'regression'

    if target_task == 'classification':
        if predictions.ndim == 3:
            metrics = calculate_sequence_classification_metrics(targets, predictions)
        else:
            metrics = calculate_classification_metrics(targets, predictions)
    elif target_task == 'regression':
        if predictions.ndim == 3:
            metrics = calculate_sequence_regression_metrics(targets, predictions)
        else:
            metrics = calculate_regression_metrics(targets, predictions)
    logger.info("Calculated metrics")

    return metrics


def preprocess_data_for_explain(genome_path, intervals_path,
                                train_test_split="random_split", test_size=0.2, test_chroms=["chr8"],
                                strand_aware=True, batch_size=4,
                                dataloader_config: Optional[Dict[str, Any]] = None):
    """Build sequence dataloader used by attribution/explain workflows.

    Args:
        genome_path (str): Path to the reference genome FASTA.
        intervals_path (str): Path to interval file used for explain inputs.
        train_test_split (str): Split strategy (`random_split` or `chromosome_split`).
        test_size (float): Test fraction when using random split.
        test_chroms (list): Held-out chromosomes for chromosome split.
        strand_aware (bool): Whether reverse-complement follows strand annotations.
        batch_size (int): Dataloader batch size.
        dataloader_config (dict, optional): Optional dataloader runtime overrides.

    Returns:
        DataLoader containing sequences for interpretation.

    Notes:
        Reads genome and interval files and emits progress logs.
    """
    from .Data.Sequence import Genome
    from .Data.Interval import Interval
    from .Data.Dataset import SeqDataset
    from .Data.Dataset import create_dataloader
    from .Data.Sampler import chromosome_split, random_split

    # load genome and intervals
    genome = Genome(genome_path)
    intervals = Interval(intervals_path)
    logger.info("Loaded genome and intervals")

    # split intervals
    if train_test_split == "chromosome_split":
        _, test_intervals = chromosome_split(intervals, test_chroms=test_chroms)
    elif train_test_split == "random_split":
        _, test_intervals = random_split(intervals, test_size=test_size)
    logger.info("Split intervals")

    # create dataset
    test_ds = SeqDataset(test_intervals, genome, strand_aware=strand_aware)
    logger.info("Created dataset")

    # create dataloader
    test_loader = create_dataloader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **_get_dataloader_kwargs(dataloader_config),
    )
    logger.info("Created dataloader")

    return test_loader

def execute_dgs_model_predict(model, test_loader, target, device, output_dir="motif_results", max_seqlets=2000):
    """
    Execute model prediction on test sequences.

    Args:
        model (nn.Module): Trained neural network model
        test_loader (DataLoader): DataLoader containing test sequences
        target (str): Target task for prediction
        device (str): Device to use for prediction ('cuda' or 'cpu')
        output_dir (str): Directory to save prediction results
        max_seqlets (int): Maximum number of sequence elements to process

    Side effects:
        Writes explanation-related artifacts through the prediction backend.
    """
    from .DL.Predict import predict
    predict(model, test_loader, target=target, device=device,
            output_dir=output_dir, max_seqlets=max_seqlets)
    logger.info("Model prediction completed")

def execute_dgs_explain(
    model,
    test_loader,
    target,
    device,
    output_dir="motif_results",
    max_seqlets=2000,
    batch_size=None,
    method="deeplift_shap",
    baseline="zero",
    n_steps=50,
    internal_batch_size=None,
):
    """
    Generate model explanations and motif enrichment analysis.

    Args:
        model (nn.Module): Trained neural network model
        test_loader (DataLoader): DataLoader containing test sequences
        target (str): Target task for explanation
        device (str): Device to use for computation ('cuda' or 'cpu')
        output_dir (str): Directory to save explanation results
        max_seqlets (int): Maximum number of sequence elements to analyze
        batch_size (int, optional): Batch size for attribution inference.
        method (str): Attribution method for contribution scores.
        baseline: Baseline for Captum methods. Only zero is supported.
        n_steps (int): Number of integration steps for Integrated Gradients.
        internal_batch_size (int, optional): Captum internal batch size for
            Integrated Gradients.

    Side effects:
        Motif enrichment results are written under `output_dir`.

    Runtime dependencies:
        - Legacy DeepLIFT/SHAP requires `tangermeme`.
        - Captum DeepLift/Integrated Gradients requires `captum`.
        - Motif discovery/report steps require `modisco` CLI in PATH.
    """
    from .DL.Explain import motif_enrich
    motif_enrich(model, test_loader.dataset, target=target, device=device,
                 output_dir=output_dir, max_seqlets=max_seqlets,
                 batch_size=batch_size, method=method, baseline=baseline,
                 n_steps=n_steps, internal_batch_size=internal_batch_size)
    logger.info("Motif enrichment completed")

def preprocess_data_for_predict(genome_path, vcf_filename, target_len=1000):
    """
    Prepare genomic data for variant effect prediction.

    Args:
        genome_path (str): Path to the reference genome file
        vcf_filename (str): Path to the VCF file containing variants
        target_len (int): Length of sequence context around variants

    Returns:
        Tuple `(variant_dataset, variant_df)`.

    Side effects:
        Loads FASTA and VCF data and emits progress logs.
    """
    from .Data.Sequence import Genome
    from .DL.Predict import read_vcf
    from .DL.Predict import VariantDataset
    
    # load genome
    genome = Genome(genome_path)
    logger.info("Loaded genome")
    
    # load vcf
    variant_df = read_vcf(vcf_filename)
    logger.info("Loaded vcf")
    
    # create vds
    vds = VariantDataset(genome, variant_df, target_len=target_len)
    logger.info("Created vds")

    return vds, variant_df

def execute_dgs_predict(
    model,
    vds,
    variant_df,
    metric_func,
    mean_by_tasks,
    device,
    batch_size=None,
    dataloader_config: Optional[Dict[str, Any]] = None,
):
    """
    Execute variant effect prediction.

    Args:
        model (nn.Module): Trained neural network model
        vds (VariantDataset): Dataset containing variant sequences
        variant_df (DataFrame): DataFrame containing variant information
        metric_func (callable): Function to calculate prediction metrics
        mean_by_tasks (bool): Whether to average predictions across tasks
        device (str): Device to use for prediction ('cuda' or 'cpu')
        batch_size (int, optional): Batch size for batched variant inference.
        dataloader_config (dict, optional): DataLoader runtime options for
            prediction batching.

    Returns:
        DataFrame with prediction results for each variant.

    Side effects:
        - Adds a `P_diff` column to `variant_df` in place.
    """
    from .DL.Predict import vep_centred_on_ds
    
    # calculate P_diff
    P_diff = vep_centred_on_ds(
        model,
        vds,
        metric_func=metric_func,
        mean_by_tasks=mean_by_tasks,
        device=device,
        batch_size=batch_size,
        dataloader_config=dataloader_config,
    )
    logger.info("Calculated P_diff")

    # add P_diff to variant_df
    variant_df['P_diff'] = P_diff
    logger.info("Added P_diff to variant_df")

    return variant_df


# client class
class DgsCLI:
    """
    Main client class for executing DGS commands.

    This class handles:
    - Configuration validation and management
    - Model and component initialization
    - Execution of different analysis modes (train, evaluate, explain, predict)
    - Resource management and cleanup

    Attributes:
        config (dict): Configuration dictionary containing all parameters
        device (str): Device to use for computation ('cuda' or 'cpu')
        model (nn.Module): Neural network model instance
        optimizer (optim.Optimizer): Optimizer for model training
        criterion (callable): Loss function for training
    """
    
    def __init__(self, config, device=None):
        """
        Initialize DgsCLI instance.

        Args:
            config (dict): Configuration dictionary containing all parameters
            device (str, optional): Device to use for computation. Defaults to None.
        """
        self.logger = logging.getLogger("dgs")
        self.optimizer = None
        self.criterion = None
        self.trainer = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.explain_data_loader = None
        self.predict_data_loader = None
        self.vds = None
        self.variant_df = None

        # Validate essential config keys
        self.config = config
        self._validate_minimal_config()

        if device is None:
            requested_device = self.config.get("device", "cuda")
            if requested_device == "cuda" and not torch.cuda.is_available():
                requested_device = "cpu"
            self.device = torch.device(requested_device)
        else:
            self.device = torch.device(device)

        # Initialize components
        self._initialize_components()

    def _build_dataloader_config(self) -> Dict[str, Any]:
        """Build dataloader runtime options from config, keeping defaults unchanged."""
        data_cfg = self.config.get("data", {})
        dataloader_cfg = {
            "num_workers": data_cfg.get("num_workers", 0),
            "pin_memory": data_cfg.get("pin_memory", False),
            "persistent_workers": data_cfg.get("persistent_workers", False),
            "prefetch_factor": data_cfg.get("prefetch_factor"),
        }
        self.logger.info(
            "DataLoader options: num_workers=%s, pin_memory=%s, persistent_workers=%s, prefetch_factor=%s",
            dataloader_cfg["num_workers"],
            dataloader_cfg["pin_memory"],
            dataloader_cfg["persistent_workers"],
            dataloader_cfg["prefetch_factor"],
        )
        return dataloader_cfg

    def _build_predict_dataloader_config(self) -> Dict[str, Any]:
        """Build prediction dataloader options with backward-compatible defaults."""
        predict_cfg = self.config.get("predict", {})
        dataloader_cfg = {
            "num_workers": predict_cfg.get("num_workers", 0),
            "pin_memory": predict_cfg.get("pin_memory", False),
            "persistent_workers": predict_cfg.get("persistent_workers", False),
            "prefetch_factor": predict_cfg.get("prefetch_factor"),
        }
        self.logger.info(
            "Predict DataLoader options: num_workers=%s, pin_memory=%s, persistent_workers=%s, prefetch_factor=%s",
            dataloader_cfg["num_workers"],
            dataloader_cfg["pin_memory"],
            dataloader_cfg["persistent_workers"],
            dataloader_cfg["prefetch_factor"],
        )
        return dataloader_cfg

    def _ensure_evaluation_trainer(self) -> None:
        """Create a minimal trainer for evaluate-only mode if not already available."""
        if self.trainer is not None:
            return

        if self.criterion is None:
            self.logger.warning(
                "No criterion configured for evaluation; falling back to torch.nn.MSELoss()."
            )
            self.criterion = nn.MSELoss()

        if self.optimizer is None:
            self.logger.warning(
                "No optimizer configured for evaluation; creating a temporary Adam optimizer."
            )
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        from .DL.Trainer import Trainer
        self.trainer = Trainer(
            self.model,
            self.criterion,
            self.optimizer,
            self.device,
            checkpoint_dir=self.config.get("train", {}).get("checkpoint_dir", "checkpoints"),
        )
    
    def _validate_minimal_config(self):
        """
        Validate the minimal required configuration parameters.

        Raises:
            ConfigError: If required parameters are missing or invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")
        
        if "data" not in self.config:
            raise ValueError("Config must contain 'data' key")
        
        if "model" not in self.config:
            self.config["model"] = {"type": "CNN"}
            self.config["model"]["args"] = {"output_size": len(self.config["data"]["target_tasks"])}
            self.logger.warning("No model specified in config, using default model: %s", self.config["model"])
        
        if "modes" not in self.config:
            self.config["modes"] = ["train", "evaluate"]
            self.logger.warning("No modes specified in config, using default modes: %s, \n \
                                if you want to use explain mode, please specify the target in config, \n \
                                if you want to use prediction mode, please specify the VCF path in config", 
                                self.config["modes"])

        if "device" not in self.config:
            self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.warning("No device specified in config, using default device: %s", self.config["device"])

        if "output_dir" not in self.config:
            self.config["output_dir"] = os.path.join(os.getcwd(), "output")
            self.logger.warning("No output_dir specified in config, using default output_dir: %s", self.config["output_dir"])

        try:
            validate_config(self.config, check_files=True)
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(f"Invalid DGS config: {exc}") from exc

    def _initialize_components(self):
        """
        Initialize all required components based on configuration.

        This includes:
        - Model architecture
        - Optimizer
        - Loss function
        - Data preprocessing components
        """
        self.logger.info("Initializing components...")

        # Initialize model
        self.logger.info("Initializing model...")
        try:
            # from .Config import get_model_from_config
            # self.model = get_model_from_config(self.config["model"])
            from . import Model
            model_config = self.config["model"]
            model_type = model_config["type"]
            available_models = sorted(
                name for name in dir(Model)
                if not name.startswith("_") and isinstance(getattr(Model, name), type)
            )
            if not hasattr(Model, model_type):
                raise ValueError(
                    f"Unknown model type '{model_type}'. Available exported models: {available_models}. "
                    "If you expect this model to exist, ensure it is exported in DGS/Model/__init__.py."
                )
            model = getattr(Model, model_type)
            self.model = model(**model_config.get("args", {}))
            self.model = self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}")
            
        # Initialize train components
        if "train" in self.config:
            self.logger.info("Initializing optimizer...")
            try:
                from torch import optim
                optimizer_config = self.config["train"]["optimizer"]
                optimizer_class = getattr(optim, optimizer_config["type"])
                self.optimizer = optimizer_class(
                    self.model.parameters(),
                    **optimizer_config.get("params", {})
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize optimizer: {e}")
            
            self.logger.info("Initializing criterion...")
            try:
                criterion_config = self.config["train"]["criterion"]
                criterion_name = criterion_config["type"]
                if hasattr(nn, criterion_name):
                    criterion_class = getattr(nn, criterion_name)
                else:
                    from . import DL
                    if not hasattr(DL, criterion_name):
                        raise AttributeError(
                            f"Unknown criterion '{criterion_name}'. "
                            "Use torch.nn losses or exported DGS.DL losses."
                        )
                    criterion_class = getattr(DL, criterion_name)
                self.criterion = criterion_class(**criterion_config.get("params", {}))
            except Exception as e:
                raise RuntimeError(f"Failed to initialize criterion: {e}")

        # initialize trainer
        if self.optimizer is not None and self.criterion is not None:
            self.logger.info("Initializing trainer...")
            try:
                from .DL.Trainer import Trainer
                self.trainer = Trainer(
                    self.model, 
                    self.criterion, 
                    self.optimizer, 
                    self.device, 
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize trainer: {e}")
    
    def execute(self):
        """
        Execute the requested command based on configuration.

        This method dispatches to the appropriate execution method based on
        the command specified in the configuration (train, evaluate, explain, predict).

        Raises:
            ValueError: If an invalid command is specified

        Side effects:
            Depending on selected modes, this call may train models, write
            checkpoints, save evaluation metrics, produce motif outputs, and
            write variant prediction CSV files.
        """
    
        # Training mode
        if "train" in self.config.get("modes", []):
            self.logger.info("Starting training mode...")
            self._execute_train()
        
        # Evaluation mode
        if "evaluate" in self.config.get("modes", []):
            self.logger.info("Starting evaluation mode...")
            self._execute_evaluate()
        
        # Explanation mode
        if "explain" in self.config.get("modes", []):
            self.logger.info("Starting explanation mode...")
            self._execute_explain()
            
        # Prediction mode
        if "predict" in self.config.get("modes", []):
            self.logger.info("Starting prediction mode...")
            self._execute_predict()
    
    def _execute_train(self):
        """Execute training mode."""
        if "train" not in self.config:
            raise RuntimeError("Training mode requires 'train' configuration")

        dataloader_config = self._build_dataloader_config()

        # Prepare data
        self.logger.info("Preparing data for training...")
        train_loader, val_loader, test_loader = preprocess_data_for_train(
            self.config["data"]["genome_path"],
            self.config["data"]["target_tasks"],
            self.config["data"]["intervals_path"],
            train_test_split=self.config["data"].get("train_test_split", "random_split"),
            test_size=self.config["data"].get("test_size", 0.2),
            val_size=self.config["data"].get("val_size", 0.2),
            test_chroms=self.config["data"].get("test_chroms", ["chr8"]),
            val_chroms=self.config["data"].get("val_chroms", ["chr7"]),
            strand_aware=self.config["data"].get("strand_aware", True),
            batch_size=self.config["data"].get("batch_size", 4),
            loader_mode=self.config["data"].get("loader_mode", "streaming"),
            random_state=self.config["data"].get("random_state"),
            dataloader_config=dataloader_config,
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Train model
        self.logger.info("Training model...")
        trainer_args = dict(self.config["train"].get("trainer_args", {}))
        trainer_args.setdefault("use_amp", self.config["train"].get("use_amp", False))
        trainer_args.setdefault("amp_dtype", self.config["train"].get("amp_dtype", "float16"))
        trainer_args.setdefault(
            "non_blocking",
            self.config["train"].get("non_blocking", self.config["data"].get("pin_memory", False)),
        )
        self.logger.info(
            "Trainer acceleration options: use_amp=%s, amp_dtype=%s, non_blocking=%s",
            trainer_args.get("use_amp"),
            trainer_args.get("amp_dtype"),
            trainer_args.get("non_blocking"),
        )
        self.model, self.trainer = execute_dgs_train(
            self.train_loader,
            self.val_loader,
            self.model,
            self.optimizer,
            self.criterion,
            self.device,
            patience=self.config["train"].get("patience", 10),
            max_epochs=self.config["train"].get("max_epochs", 500),
            resume=self.config["train"].get("resume", False),
            resume_model_name=self.config["train"].get("resume_model_name", "best_model.pt"),
            checkpoint_dir=self.config["train"].get("checkpoint_dir", "checkpoints"), 
            use_tensorboard=self.config["train"].get("use_tensorboard", False),
            tensorboard_dir=self.config["train"].get("tensorboard_dir", "tensorboard"), 
            **trainer_args,
        )
        self.logger.info("Training completed.")

    def _execute_evaluate(self):
        """Execute evaluation mode."""
        evaluate_config = self.config.get("evaluate", {})
        checkpoint_path = evaluate_config.get("checkpoint_path")

        if self.trainer is None and checkpoint_path is None:
            raise RuntimeError(
                "No trained model available for evaluation. "
                "Please train first or set evaluate.checkpoint_path."
            )

        # Evaluate-only mode can build a minimal trainer when checkpoint is provided.
        self._ensure_evaluation_trainer()
        
        self.logger.info("Preparing data for evaluation...")
        if self.test_loader is None:
            self.test_loader = preprocess_data_for_evaluate(
                self.config["data"]["genome_path"],
                self.config["data"]["target_tasks"],
                self.config["data"]["intervals_path"],
                train_test_split=self.config["data"].get("train_test_split", "random_split"),
                test_size=self.config["data"].get("test_size", 0.2),
                test_chroms=self.config["data"].get("test_chroms", ["chr8"]),
                strand_aware=self.config["data"].get("strand_aware", True),
                batch_size=self.config["data"].get("batch_size", 4),
                dataloader_config=self._build_dataloader_config(),
            )
        
        self.logger.info("Evaluating model...")
        self.metrics = execute_dgs_evaluate(
            self.test_loader, 
            self.trainer,
            checkpoint_path=checkpoint_path,
        )

        # save metrics
        self.metrics.to_csv(os.path.join(self.config["output_dir"], "metrics.csv"))
        self.logger.info("Evaluation completed.")
    
    def _execute_explain(self):
        """Execute explanation mode."""
        if self.model is None:
            raise RuntimeError("No model available for explanation")
        explain_config = self.config.get("explain", {})
        
        self.logger.info("Preparing data for explanation...")
        if self.explain_data_loader is None:
            self.explain_data_loader = preprocess_data_for_explain(
                self.config["data"]["genome_path"],
                self.config["data"]["intervals_path"],
                train_test_split=self.config["data"].get("train_test_split", "random_split"),
                test_size=self.config["data"].get("test_size", 0.2),
                test_chroms=self.config["data"].get("test_chroms", ["chr8"]),
                strand_aware=self.config["data"].get("strand_aware", True),
                batch_size=self.config["data"].get("batch_size", 4),
                dataloader_config=self._build_dataloader_config(),
            )
        
        self.logger.info("Executing explanation...")
        execute_dgs_explain(
            self.model,
            self.explain_data_loader,
            target=explain_config.get("target", 0),
            device=self.device,
            output_dir=explain_config.get("output_dir", explain_config.get("motif_results", "motif_results")), 
            max_seqlets=explain_config.get("max_seqlets", 2000),
            batch_size=explain_config.get("batch_size"),
            method=explain_config.get("method", "deeplift_shap"),
            baseline=explain_config.get("baseline", "zero"),
            n_steps=explain_config.get("n_steps", 50),
            internal_batch_size=explain_config.get("internal_batch_size"),
        )
        self.logger.info("Explanation completed.")
    
    def _execute_predict(self):
        """Execute prediction mode."""
        if self.model is None:
            raise RuntimeError("No model available for prediction")

        self.logger.info("Preparing data for prediction...")
        if self.vds is None or self.variant_df is None:
            self.vds, self.variant_df  = preprocess_data_for_predict(
                self.config["data"]["genome_path"],
                self.config["predict"]["vcf_path"],
                target_len=self.config["predict"]["sequence_length"]
            )
        
        self.logger.info("Predicting variants...")
        predict_dataloader_config = self._build_predict_dataloader_config()
        self.result_df = execute_dgs_predict(
            self.model,
            self.vds,
            self.variant_df,
            metric_func=self.config["predict"].get("metric_func", "diff"),
            mean_by_tasks=self.config["predict"].get("mean_by_tasks", True),
            device=self.device,
            batch_size=self.config["predict"].get("batch_size"),
            dataloader_config=predict_dataloader_config,
        )

        save_path = os.path.join(self.config["output_dir"], "variant_predictions.csv")
        self.result_df.to_csv(save_path)
        self.logger.info("Variant predictions saved to %s", save_path)
        self.logger.info("Prediction completed.")
