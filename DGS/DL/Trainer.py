"""Model training loops and checkpoint management.

Purpose:
    Provide a reusable trainer for DGS model optimization and validation.

Main Responsibilities:
    - Run train/validate/predict loops with optional AMP and gradient clipping.
    - Save and load checkpoints including model, optimizer, and scaler states.
    - Track losses/metrics and expose progress reporting hooks.

Key Runtime Notes:
    - AMP is only enabled on CUDA when `use_amp=True`.
    - Batch-level failures are logged; all-failed epochs raise explicit errors.
    - Checkpoint loading supports model-only restore for evaluation workflows.
"""

import inspect
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

logger = logging.getLogger("dgs")


def _metrics_to_dict(metrics: "TrainerMetrics") -> Dict[str, Any]:
    """Serialize trainer metrics into plain Python containers."""
    return {
        "train_losses": list(metrics.train_losses),
        "val_losses": list(metrics.val_losses),
        "train_metrics": list(metrics.train_metrics),
        "val_metrics": list(metrics.val_metrics),
        "best_val_loss": float(metrics.best_val_loss),
        "best_val_metric": float(metrics.best_val_metric),
        "best_epoch": int(metrics.best_epoch),
    }


def _move_tensors_to_cpu(value: Any) -> Any:
    """Recursively move tensors inside checkpoint structures to CPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _move_tensors_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_tensors_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_cpu(v) for v in value)
    return value


def _detach_to_cpu(value: Any) -> Any:
    """Recursively detach tensors and move them to CPU for prediction storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_detach_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(v) for v in value)
    if isinstance(value, dict):
        return {k: _detach_to_cpu(v) for k, v in value.items()}
    return torch.as_tensor(value).detach().cpu()


def _cat_batch_outputs(values: List[Any]) -> Any:
    """Concatenate stored batch outputs while preserving nested structures."""
    if not values:
        return torch.empty(0)
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values)
    if isinstance(first, tuple):
        return tuple(_cat_batch_outputs([value[idx] for value in values]) for idx in range(len(first)))
    if isinstance(first, list):
        return [_cat_batch_outputs([value[idx] for value in values]) for idx in range(len(first))]
    if isinstance(first, dict):
        return {key: _cat_batch_outputs([value[key] for value in values]) for key in first}
    return torch.cat([torch.as_tensor(value) for value in values])


def _first_tensor(value: Any) -> torch.Tensor:
    """Return the primary tensor from common model output/target structures."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            return torch.empty(0)
        return _first_tensor(value[0])
    if isinstance(value, dict):
        for item in value.values():
            return _first_tensor(item)
    return torch.as_tensor(value)


def _metrics_from_dict(data: Optional[Dict[str, Any]]) -> "TrainerMetrics":
    """Restore trainer metrics from a checkpoint dictionary."""
    metrics = TrainerMetrics()
    if not data:
        return metrics
    for key, value in data.items():
        if hasattr(metrics, key):
            setattr(metrics, key, value)
    return metrics


def _state_to_dict(state: "TrainerState") -> Dict[str, Any]:
    """Serialize trainer state without requiring dataclass unpickling."""
    return {
        "format_version": 2,
        "epoch": int(state.epoch),
        "global_step": int(state.global_step),
        "best_val_loss": float(state.best_val_loss),
        "model_state": state.model_state,
        "optimizer_state": state.optimizer_state,
        "scaler_state": state.scaler_state,
        "metrics": _metrics_to_dict(state.metrics),
    }


def _state_from_checkpoint(checkpoint: Any) -> "TrainerState":
    """Convert legacy or dict checkpoints into a `TrainerState` instance."""
    if isinstance(checkpoint, TrainerState):
        return checkpoint

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unsupported checkpoint format. Expected a TrainerState or a dictionary."
        )

    if "model_state" not in checkpoint:
        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return TrainerState(model_state=checkpoint)
        raise ValueError("Checkpoint is missing required field: 'model_state'")

    metrics = checkpoint.get("metrics")
    if isinstance(metrics, TrainerMetrics):
        restored_metrics = metrics
    elif isinstance(metrics, dict):
        restored_metrics = _metrics_from_dict(metrics)
    else:
        restored_metrics = TrainerMetrics()

    return TrainerState(
        epoch=int(checkpoint.get("epoch", 0)),
        global_step=int(checkpoint.get("global_step", 0)),
        best_val_loss=float(checkpoint.get("best_val_loss", restored_metrics.best_val_loss)),
        model_state=checkpoint["model_state"],
        optimizer_state=checkpoint.get("optimizer_state", {}),
        scaler_state=checkpoint.get("scaler_state", {}),
        metrics=restored_metrics,
    )

@dataclass
class TrainerMetrics:
    """
    Training metrics tracker for monitoring model performance.

    This class maintains lists of training and validation metrics throughout
    the training process, as well as tracking the best model performance.

    Attributes:
        train_losses (List[float]): History of training losses
        val_losses (List[float]): History of validation losses
        train_metrics (List[float]): History of training metrics
        val_metrics (List[float]): History of validation metrics
        best_val_loss (float): Best validation loss achieved
        best_val_metric (float): Best validation metric achieved
        best_epoch (int): Epoch number where best performance was achieved
    """
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_metrics: List[float] = field(default_factory=list)
    val_metrics: List[float] = field(default_factory=list)
    best_val_loss: float = float('inf')
    best_val_metric: float = 0.0
    best_epoch: int = 0
    
@dataclass
class TrainerState:
    """
    Trainer state container for checkpointing and resuming training.

    This class encapsulates all necessary information to save and restore
    the training state, including model and optimizer states.

    Attributes:
        epoch (int): Current epoch number
        global_step (int): Total number of training steps
        best_val_loss (float): Best validation loss achieved
        model_state (Dict): Model's state dictionary
        optimizer_state (Dict): Optimizer's state dictionary
        metrics (TrainerMetrics): Training metrics history
    """
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    model_state: Dict = field(default_factory=dict)
    optimizer_state: Dict = field(default_factory=dict)
    scaler_state: Dict = field(default_factory=dict)
    metrics: TrainerMetrics = field(default_factory=TrainerMetrics)

class Trainer:
    """
    Comprehensive model trainer for deep learning genomic sequence analysis.

    This class provides a complete training framework with the following features:
    - Flexible training and validation loops
    - Automatic device management (CPU/GPU)
    - Checkpoint saving and loading
    - Early stopping and learning rate scheduling
    - TensorBoard integration for monitoring
    - Custom metric computation and tracking
    - Progress visualization

    The trainer handles all aspects of the training process, including:
    - Batch preparation and device placement
    - Forward and backward passes
    - Gradient clipping and optimization
    - Metrics computation and logging
    - Model state management
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        scheduler: Optional[Any] = None,
        clip_grad_norm: bool = False,
        max_grad_norm: float = 1.0,
        evaluate_training: bool = False,
        metric_sample: int = 100,
        patience: int = 10,
        use_tensorboard: bool = False,
        tensorboard_dir: Optional[Union[str, Path]] = None,
        use_amp: bool = False,
        amp_dtype: Union[str, torch.dtype] = "float16",
        non_blocking: bool = False,
    ):
        """Initialize trainer.
        
        Args:
            model: Neural network model
            criterion: Loss criterion
            optimizer: Optimizer
            device: Computation device
            checkpoint_dir: Directory for saving checkpoints
            scheduler: Learning rate scheduler
            clip_grad_norm: Whether to clip gradients
            max_grad_norm: Maximum gradient norm
            evaluate_training: Whether to evaluate during training
            metric_sample: Number of samples for metric calculation
            patience: Early stopping patience
            use_tensorboard: Whether to use tensorboard
            tensorboard_dir: Tensorboard log directory
            use_amp: Whether to enable mixed precision on CUDA
            amp_dtype: autocast dtype, e.g. "float16" or "bfloat16"
            non_blocking: Use non_blocking host->device transfers when possible
        """
        self.device = device
        self.non_blocking = bool(non_blocking)
        
        # Move model and criterion to device
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        
        # Initialize optimizer
        self.optimizer = optimizer
        
        # Ensure optimizer's parameters are on the correct device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        self.scheduler = scheduler
        
        # Training settings
        self.clip_grad_norm = clip_grad_norm
        self.max_grad_norm = max_grad_norm
        self.evaluate_training = evaluate_training
        self.metric_sample = metric_sample
        self.patience = patience

        # AMP settings (kept opt-in for backward compatibility)
        self.amp_dtype = self._resolve_amp_dtype(amp_dtype)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        if use_amp and self.device.type != "cuda":
            logger.info("AMP requested but disabled because device is not CUDA")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
            except TypeError:
                self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        logger.info(
            "Trainer runtime options: use_amp=%s, amp_dtype=%s, non_blocking=%s",
            self.use_amp,
            self.amp_dtype,
            self.non_blocking,
        )
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir or "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard = SummaryWriter(tensorboard_dir or "runs")
            
        # Initialize state
        self.state = TrainerState()
        self.metrics = TrainerMetrics()

    @staticmethod
    def _resolve_amp_dtype(amp_dtype: Union[str, torch.dtype]) -> torch.dtype:
        """Map user-provided AMP dtype into torch dtype."""
        if isinstance(amp_dtype, torch.dtype):
            return amp_dtype

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if isinstance(amp_dtype, str):
            key = amp_dtype.lower()
            if key in dtype_map:
                return dtype_map[key]
        raise ValueError(
            "Invalid amp_dtype. Supported values are 'float16', 'fp16', 'bfloat16', 'bf16', or torch.dtype."
        )

    def _autocast_context(self):
        """Return an autocast context only when AMP is enabled."""
        if self.use_amp:
            return torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=True,
            )
        return nullcontext()
        
    def _prepare_batch(self, data, target):
        """Prepare batch data by ensuring all inputs are tensors on the correct device.
        
        Args:
            data: Input data, can be:
                - single tensor/array/list
                - list of tensors/arrays/lists
            target: Target data, same format options as data
                
        Returns:
            (tensor or list[tensor], tensor or list[tensor]): 
                Processed data and target tensors on the correct device
        """
        def _to_tensor(x, dtype=None):
            if isinstance(x, torch.Tensor):
                x = x.to(self.device, non_blocking=self.non_blocking)
                return x if dtype is None else x.to(dtype)
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).to(self.device, non_blocking=self.non_blocking)
                return x if dtype is None else x.to(dtype)
            x = torch.tensor(x, device=self.device)
            return x if dtype is None else x.to(dtype)
        
        def _process_input(x, dtype=None):
            if isinstance(x, (list, tuple)):
                return [_to_tensor(item, dtype) for item in x]
            return _to_tensor(x, dtype)
        
        # Process data with float32 dtype for model input
        processed_data = _process_input(data, dtype=torch.float32)
        
        # Process target with float32 dtype by default
        # Only try to get criterion dtype if it has parameters and is not None
        target_dtype = torch.float32
        if self.criterion is not None and hasattr(self.criterion, 'parameters'):
            try:
                param = next(self.criterion.parameters())
                if param is not None:
                    target_dtype = param.dtype
            except StopIteration:
                pass
                
        processed_target = _process_input(target, dtype=target_dtype)
            
        return processed_data, processed_target
        
    def save_checkpoint(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save training checkpoint.
        
        Args:
            path: Path to save checkpoint. If None, uses default path.
        """
        path = Path(path or self.checkpoint_dir / f"checkpoint_{self.state.epoch}.pt")
        
        # Move model to CPU before saving
        model_state = _move_tensors_to_cpu(self.model.state_dict())
        
        # Move optimizer state to CPU
        optimizer_state = _move_tensors_to_cpu(self.optimizer.state_dict())
        
        # Update state
        self.state.best_val_loss = self.metrics.best_val_loss
        self.state.model_state = model_state
        self.state.optimizer_state = optimizer_state
        self.state.scaler_state = (
            _move_tensors_to_cpu(self.scaler.state_dict()) if self.use_amp else {}
        )
        self.state.metrics = self.metrics
        
        # Save checkpoint as a plain dictionary. This is friendlier to modern
        # PyTorch safe-loading defaults than pickling the TrainerState dataclass.
        torch.save(_state_to_dict(self.state), path)
        logger.info(f"Saved checkpoint to {path}")

    @staticmethod
    def _torch_load_checkpoint(path: Path, device: torch.device):
        """Load checkpoints across PyTorch versions and safe-loading defaults."""
        load_kwargs = {"map_location": device}
        try:
            if "weights_only" in inspect.signature(torch.load).parameters:
                load_kwargs["weights_only"] = False
        except (TypeError, ValueError):
            # Some patched torch builds may not expose an inspectable signature.
            pass
        return torch.load(path, **load_kwargs)
        
    def load_checkpoint(self, path: Union[str, Path], load_optimizer: bool = True) -> None:
        """Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
            load_optimizer: Whether to restore optimizer/scaler states.
            
        Raises:
            FileNotFoundError: If checkpoint file not found
            RuntimeError: If checkpoint loading fails
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
            
        try:
            # Load checkpoint. `weights_only=False` is explicit when supported
            # because legacy DGS checkpoints stored a TrainerState dataclass.
            checkpoint = self._torch_load_checkpoint(path, self.device)
            checkpoint_state = _state_from_checkpoint(checkpoint)
            
            # Restore state
            self.state = checkpoint_state
            self.metrics = checkpoint_state.metrics
            
            # Restore model and optimizer
            self.model.load_state_dict(checkpoint_state.model_state)
            if load_optimizer and checkpoint_state.optimizer_state:
                self.optimizer.load_state_dict(checkpoint_state.optimizer_state)
                if self.use_amp and checkpoint_state.scaler_state:
                    self.scaler.load_state_dict(checkpoint_state.scaler_state)
                
                # Ensure optimizer state is on correct device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            elif load_optimizer:
                logger.warning(
                    "Checkpoint does not contain optimizer state; restored model weights only."
                )
            
            logger.info(f"Loaded checkpoint from {path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
            
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        validate_fn: Optional[Callable] = None
    ) -> float:
        """
        Train the model for one epoch.

        This method:
        1. Sets the model to training mode
        2. Iterates over all batches in the training loader
        3. Performs forward and backward passes
        4. Updates model parameters
        5. Tracks metrics and losses
        6. Optionally performs validation

        Args:
            train_loader: DataLoader containing training data
            epoch: Current epoch number
            validate_fn: Optional function for validation during training

        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        success_batches = 0
        failed_batches = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                try:
                    # Prepare and convert batch data
                    data, target = self._prepare_batch(data, target)
                    
                    # Forward pass (handle list of inputs if needed)
                    self.optimizer.zero_grad()
                    with self._autocast_context():
                        output = self.model(data)
                        # Handle loss computation with multiple targets if needed
                        loss = self.criterion(output, target)

                    # Backward pass
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        if self.clip_grad_norm:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.max_grad_norm
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        if self.clip_grad_norm:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.max_grad_norm
                            )
                        self.optimizer.step()
                    
                    # Update metrics
                    total_loss += loss.item()
                    success_batches += 1
                    avg_loss = total_loss / success_batches
                    
                    # Update progress bar
                    pbar.set_postfix(loss=f"{avg_loss:.4f}")
                    
                    # Log to tensorboard
                    if self.use_tensorboard:
                        self.tensorboard.add_scalar(
                            "train/loss",
                            loss.item(),
                            self.state.global_step
                        )
                        
                    self.state.global_step += 1
                    
                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {e}")
                    failed_batches += 1
                    continue

        if success_batches == 0:
            raise RuntimeError(
                f"All training batches failed at epoch {epoch} (failed={failed_batches})."
            )
        if failed_batches > 0:
            logger.warning(
                "Epoch %s completed with partial failures: success_batches=%s, failed_batches=%s",
                epoch,
                success_batches,
                failed_batches,
            )

        avg_loss = total_loss / success_batches
        self.metrics.train_losses.append(avg_loss)
        
        return avg_loss
        
    def validate(
        self,
        val_loader: DataLoader,
        return_predictions: bool = False
    ) -> Union[Tuple[float, float], Tuple[float, float, torch.Tensor, torch.Tensor]]:
        """
        Validate the model on a validation dataset.

        This method:
        1. Sets the model to evaluation mode
        2. Performs forward passes without gradient computation
        3. Computes validation loss and metrics
        4. Optionally returns predictions and targets

        Args:
            val_loader: DataLoader containing validation data
            return_predictions: Whether to return model predictions and targets

        Returns:
            If return_predictions is False:
                tuple: (average validation loss, average validation metric)
            If return_predictions is True:
                tuple: (average validation loss, average validation metric,
                       all predictions tensor, all targets tensor)
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        success_batches = 0
        failed_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                try:
                    # Prepare and convert batch data
                    data, target = self._prepare_batch(data, target)
                    
                    # Forward pass
                    with self._autocast_context():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                    
                    # Store results
                    total_loss += loss.item()
                    success_batches += 1
                    if return_predictions:
                        predictions.append(_detach_to_cpu(output))
                        targets.append(_detach_to_cpu(target))
                        
                except Exception as e:
                    logger.error(f"Error in validation: {e}")
                    failed_batches += 1
                    continue

        if success_batches == 0:
            raise RuntimeError(
                f"All validation batches failed (failed={failed_batches})."
            )
        if failed_batches > 0:
            logger.warning(
                "Validation completed with partial failures: success_batches=%s, failed_batches=%s",
                success_batches,
                failed_batches,
            )

        # Calculate metrics
        avg_loss = total_loss / success_batches
        self.metrics.val_losses.append(avg_loss)
        
        if return_predictions:
            predictions = _cat_batch_outputs(predictions)
            targets = _cat_batch_outputs(targets)
            metric = self.compute_metric(_first_tensor(predictions), _first_tensor(targets))
            self.metrics.val_metrics.append(metric)
            return avg_loss, metric, predictions, targets
            
        self.metrics.val_metrics.append(0.0)
        return avg_loss, 0.0
            
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping: bool = True,
        verbose: bool = True
    ) -> TrainerMetrics:
        """
        Train the model for multiple epochs with validation.

        This method implements the complete training loop with:
        - Epoch-wise training and validation
        - Early stopping based on validation performance
        - Learning rate scheduling
        - Progress tracking and visualization
        - Checkpoint management

        Args:
            train_loader: DataLoader containing training data
            val_loader: DataLoader containing validation data
            epochs: Maximum number of epochs to train
            early_stopping: Whether to use early stopping
            verbose: Whether to print progress information

        Returns:
            TrainerMetrics: Complete training history and metrics

        Note:
            The best model state is automatically saved when validation
            performance improves.
        """
        start_time = time.time()
        no_improve = 0
        
        try:
            for epoch in range(self.state.epoch, epochs):
                epoch_start_time = time.time()
                self.state.epoch = epoch
                
                # Training
                train_loss = self.train_epoch(train_loader, epoch)
                
                # Validation
                val_loss, val_metric = self.validate(val_loader)
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                        current_lr = self.optimizer.param_groups[0]['lr']
                    else:
                        self.scheduler.step()
                        current_lr = self.scheduler.get_last_lr()[0]
                    
                # Save best model
                if val_loss < self.metrics.best_val_loss:
                    self.metrics.best_val_loss = val_loss
                    self.metrics.best_val_metric = val_metric
                    self.metrics.best_epoch = epoch
                    self.save_checkpoint(self.checkpoint_dir / "best_model.pt")
                    no_improve = 0
                else:
                    no_improve += 1
                    
                # Show training log
                epoch_time = time.time() - epoch_start_time
                current_metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_metric': val_metric
                }
                
                if self.scheduler is not None:
                    current_metrics['learning_rate'] = current_lr
                
                show_train_log(
                    epoch=epoch,
                    metrics=self.metrics,
                    current_metrics=current_metrics,
                    time_used=epoch_time,
                    save_dir=self.checkpoint_dir,
                    verbose=verbose,
                    plot=True
                )
                
                # Early stopping
                if early_stopping and no_improve >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
            
        finally:
            # Save final checkpoint
            self.save_checkpoint(self.checkpoint_dir / "final_model.pt")
            
            # Training summary
            duration = time.time() - start_time
            logger.info(
                f"\nTraining completed in {duration:.2f}s\n"
                f"Best validation loss: {self.metrics.best_val_loss:.4f}\n"
                f"Best validation metric: {self.metrics.best_val_metric:.4f}\n"
                f"Best epoch: {self.metrics.best_epoch}"
            )
            
        return self.metrics
        
    def predict(
        self,
        test_loader: DataLoader,
        return_probs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Make predictions.
        
        Args:
            test_loader: Test data loader
            return_probs: Whether to return probabilities
            
        Returns:
            Model predictions
            If return_probs=True, returns (predictions, probabilities)
        """
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for data in test_loader:
                try:
                    # Handle both (data) and (data, target) formats
                    if isinstance(data, (tuple, list)):
                        data = data[0]
                    
                    # Move data to device
                    if isinstance(data, np.ndarray):
                        data = torch.from_numpy(data)
                    data = data.to(self.device, non_blocking=self.non_blocking)
                    
                    # Forward pass
                    with self._autocast_context():
                        output = self.model(data)
                    
                    # Store results
                    predictions.append(_detach_to_cpu(output))
                    if return_probs:
                        probabilities.append(torch.sigmoid(_first_tensor(output)).cpu())
                        
                except Exception as e:
                    logger.error(f"Error in prediction: {e}")
                    continue
                    
        predictions = _cat_batch_outputs(predictions)
        
        if return_probs:
            probabilities = torch.cat(probabilities)
            return predictions, probabilities
        return predictions
        
    @staticmethod
    def compute_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute evaluation metric.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Metric value
        """
        # Default to binary accuracy
        predictions = (torch.sigmoid(predictions) > 0.5).float()
        targets = (torch.sigmoid(targets) > 0.5).float()
        return (predictions == targets).float().mean().item()


def show_train_log(
    epoch: int,
    metrics: TrainerMetrics,
    current_metrics: Dict[str, float],
    time_used: float,
    verbose: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
    plot: bool = True
) -> None:
    """Display and visualize training log information.
    
    Args:
        epoch: Current epoch number
        metrics: TrainerMetrics instance containing training history
        current_metrics: Current epoch metrics dictionary containing:
            - train_loss: Training loss
            - val_loss: Validation loss
            - val_metric: Validation metric
            - learning_rate: Optional current learning rate
        time_used: Time used in seconds
        save_dir: Directory to save plots (optional)
        verbose: Whether to print log
        plot: Whether to plot training curves
    """
    # Text logging
    log_str = (
        f"\n{'='*50}\n"
        f"Epoch: {epoch}\n"
        f"Time: {time_used:.2f}s\n"
        f"Training Loss: {current_metrics['train_loss']:.4f}\n"
        f"Validation Loss: {current_metrics['val_loss']:.4f}\n"
        f"Validation Metric: {current_metrics['val_metric']:.4f}\n"
        f"Best Validation Loss: {metrics.best_val_loss:.4f}\n"
        f"Best Validation Metric: {metrics.best_val_metric:.4f}\n"
        f"Best Epoch: {metrics.best_epoch}\n"
    )
    
    if 'learning_rate' in current_metrics:
        log_str += f"Learning Rate: {current_metrics['learning_rate']:.6f}\n"
        
    log_str += f"{'='*50}\n"
    
    # Log to logger and print
    if verbose:
        logger.info(log_str)

    # Plot training curves
    if plot and save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot losses
        plot_curves(
            y_values=[metrics.train_losses, metrics.val_losses],
            labels=['Training', 'Validation'],
            title='Loss Curves',
            ylabel='Loss',
            save_path=save_dir / 'loss_curves.png'
        )
        
        # Plot metrics
        if metrics.val_metrics:
            plot_curves(
                y_values=[metrics.val_metrics],
                labels=['Validation'],
                title='Metric Curves',
                ylabel='Metric',
                save_path=save_dir / 'metric_curves.png'
            )

def plot_curves(
    y_values: List[List[float]],
    labels: List[str],
    title: str,
    ylabel: str,
    save_path: Union[str, Path],
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """Plot training curves.
    
    Args:
        y_values: List of y-values to plot
        labels: Labels for each curve
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save plot
        figsize: Figure size (width, height)
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    epochs = range(1, len(y_values[0]) + 1)
    
    for y, label in zip(y_values, labels):
        plt.plot(epochs, y, label=label)
        
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()

def plot_learning_rate(
    lr_history: List[float],
    save_path: Union[str, Path]
) -> None:
    """Plot learning rate curve.
    
    Args:
        lr_history: List of learning rates
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(lr_history) + 1)
    
    plt.plot(epochs, lr_history)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()
