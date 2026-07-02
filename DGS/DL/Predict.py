"""Variant effect prediction helpers for trained DGS models.

Purpose:
    Build variant-centered sequence pairs and compute model-based effect scores.

Main Responsibilities:
    - Parse VCF inputs and construct interval windows around variants.
    - Generate reference/alternate one-hot pairs through `VariantDataset`.
    - Run per-variant or batched inference and aggregate effect metrics.

Key Runtime Notes:
    - Batched inference supports optional dataloader worker/pin-memory settings.
    - Reference/alternate predictions are compared via configurable effect metrics.
    - Input tensors are normalized to model shape `(batch, channels, length)`.
"""

import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Union, Optional, Dict, Any
import os
import logging
from torch.utils.data import DataLoader

logger = logging.getLogger("dgs.predict")

from ..IO.vcf import read_vcf


from ..Data.Interval import Interval
def variants_to_intervals(variants, seq_len=1000) -> Interval:
    """
    Convert variant positions to sequence intervals for analysis.

    This function creates fixed-length sequence windows centered on each variant
    for model input preparation.

    Args:
        variants (pd.DataFrame): DataFrame containing variant information
        seq_len (int): Length of sequence context around variants

    Returns:
        Interval: Interval object containing sequence windows
            Each interval includes:
            - Chromosome name
            - Start position (0-based)
            - End position (exclusive)

    Note:
        The function automatically handles coordinate conversion
        between 1-based VCF and 0-based interval formats.
    """
    if int(seq_len) <= 0:
        raise ValueError(f"seq_len must be a positive integer; got {seq_len}.")

    intervals = []
    for _, mutation in variants.iterrows():
        chrom = mutation['CHROM']
        pos = int(mutation['POS']) - 1 #VCF is 1-based
        mid = seq_len // 2
        start = pos - mid
        end = seq_len + start
        if start < 0:
            raise ValueError(
                f"Variant window for {chrom}:{mutation['POS']} extends before chromosome start "
                f"with sequence_length={seq_len}. Use a shorter sequence_length or "
                "filter variants too close to chromosome starts."
            )
        intervals.append({'chrom': chrom, 'start': start, 'end': end})
    intervals = pd.DataFrame(intervals)
    intervals = Interval(intervals)
    return intervals

from ..Data.Dataset import SeqDataset
from ..Data.Sequence import DNASeq, mutate_sequence

# Backward-compatible alias (historical public symbol in this module).
mutate = mutate_sequence

class VariantDataset(SeqDataset):
    """
    Dataset class for variant effect prediction.

    This class manages sequence data for variant analysis:
    - Loads reference and variant sequences
    - Validates variant positions
    - Generates paired sequences for comparison

    Attributes:
        genome: Reference genome object
        variants (pd.DataFrame): Variant information
        var_ref (List[str]): Reference alleles
        var_alt (List[str]): Alternative alleles

    Methods:
        check_reference: Validate reference sequence matches
        __getitem__: Get reference-variant sequence pairs
    """
    
    def __init__(self, genome, variants, target_len=1000, **kwargs):
        """
        Initialize variant dataset.

        Args:
            genome: Reference genome object
            variants (pd.DataFrame): Variant information
            target_len (int): Sequence context length
            **kwargs: Additional arguments for parent class
        """
        intervals = variants_to_intervals(variants, seq_len=target_len)
        super().__init__(intervals, genome, **kwargs)
        self.variants = variants
        self.var_ref = variants['REF'].tolist()
        self.var_alt = variants['ALT'].tolist()
    
    def check_reference(self, seq, ref):
        """
        Validate that extracted sequence matches reference allele.

        Args:
            seq (str): Extracted sequence
            ref (str): Reference allele from VCF

        Returns:
            bool: True if reference matches, False otherwise
        """
        start_idx = len(seq) // 2 # variant-centred
        stop_idx = start_idx + len(ref)
        return seq[start_idx:stop_idx] == ref

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return one sample for the given flattened index."""
        seq_ref = self.seqs[idx]
        seq_alt = self.seqs[idx]

        var_ref = self.var_ref[idx]
        var_alt = self.var_alt[idx]

        if self.check_reference(seq_ref.sequence, var_ref):
            seq_alt = mutate_sequence(seq_alt.sequence, var_ref, var_alt)
            seq_alt = DNASeq(seq_alt)
        else:
            variant = self.variants.iloc[idx]
            start_idx = len(seq_ref.sequence) // 2
            observed = seq_ref.sequence[start_idx:start_idx + len(var_ref)]
            raise ValueError(
                "Reference allele mismatch for variant "
                f"{variant['CHROM']}:{variant['POS']} {var_ref}>{var_alt}: "
                f"expected REF='{var_ref}' at the center of the extracted window, "
                f"found '{observed}'. Check that the FASTA and VCF use the same "
                "genome assembly and chromosome naming."
            )

        return seq_ref.to_onehot(), seq_alt.to_onehot()


from ..Data.Sequence import one_hot_encode
def variant_effect_prediction(model, X_ref, X_alt, device=torch.device('cpu')):
    """
    Predict the effect of genetic variants using a trained model.

    This function:
    1. Processes reference and alternative sequences
    2. Runs model predictions
    3. Returns prediction scores for comparison

    Args:
        model (nn.Module): Trained neural network model
        X_ref: Reference sequences (str, array, or tensor)
        X_alt: Alternative sequences (str, array, or tensor)
        device (str): Computation device ('cuda' or 'cpu')

    Returns:
        tuple: (reference_predictions, alternate_predictions)
            Both in numpy array format with shape (N, num_tasks)

    Note:
        The function automatically handles different input formats
        and ensures consistent tensor shapes.
    """
    if type(X_ref) == str:
        X_ref = one_hot_encode(X_ref)
    if type(X_alt) == str:
        X_alt = one_hot_encode(X_alt)

    if X_ref.ndim == 2:
        X_ref = X_ref[None, :, :]
    if X_alt.ndim == 2:
        X_alt = X_alt[None, :, :]

    if X_ref.shape[-1] != 4:
        X_ref = X_ref.swapaxes(1,-1)
    if X_alt.shape[-1] != 4:
        X_alt = X_alt.swapaxes(1,-1)

    device = torch.device(device)

    # Set model to evaluation mode
    model.eval()
    model.to(device)

    p_ref = None
    with torch.inference_mode():
        d = torch.from_numpy(X_ref.astype(np.float32)).to(device)
        e = _predict_with_layout_fallback(model, d)
        p_ref = e.data.cpu().numpy()
    #print(p_ref.shape)

    p_alt = None
    with torch.inference_mode():
        d_alt = torch.from_numpy(X_alt.astype(np.float32)).to(device)
        e_alt = _predict_with_layout_fallback(model, d_alt)
        p_alt = e_alt.data.cpu().numpy()
    #print(p_alt.shape)

    return p_ref, p_alt


def metric_predicted_effect(p_ref, p_alt, metric_func='diff', mean_by_tasks=True):
    """
    Calculate variant effect scores using specified metrics.

    Supported metrics:
    - diff: Absolute difference between predictions
    - ratio: Ratio of predicted effects
    - log_ratio: Log ratio of predicted effects
    - max: Maximum predicted effect
    - min: Minimum predicted effect
    - custom: User-provided callable function

    Args:
        p_ref (np.ndarray): Reference sequence predictions
        p_alt (np.ndarray): Alternative sequence predictions
        metric_func (str or callable): Metric calculation method
        mean_by_tasks (bool): Whether to average across tasks

    Returns:
        np.ndarray: Variant effect scores

    Note:
        For multi-task models, scores can be averaged or kept separate
        based on mean_by_tasks parameter.
    """
    allowed_metrics = ('diff', 'ratio', 'log_ratio', 'max', 'min')

    if metric_func == 'diff':
        p_eff = p_alt - p_ref
        p_eff = np.abs(p_eff)

    elif metric_func == 'ratio':
        p_eff = p_alt / p_ref

    elif metric_func == 'log_ratio':
        p_eff = np.log(p_alt / p_ref)

    elif metric_func == 'max':
        p_eff = np.max(p_alt, axis=1) - np.max(p_ref, axis=1)

    elif metric_func == 'min':
        p_eff = np.min(p_alt, axis=1) - np.min(p_ref, axis=1)

    elif callable(metric_func):
        p_eff = metric_func(p_ref, p_alt)

    else:
        raise ValueError(
            f"Invalid metric function: {metric_func}. "
            f"Must be one of: {allowed_metrics}, or a callable."
        )

    if mean_by_tasks:
        p_eff = p_eff.mean(axis=1)

    return p_eff


def _to_model_input(batch: Union[np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
    """Convert batches to model input shape (N, 4, L) on the target device."""
    if isinstance(batch, np.ndarray):
        batch = torch.from_numpy(batch)
    if batch.ndim == 2:
        batch = batch.unsqueeze(0)
    # Keep parity with legacy `variant_effect_prediction` layout handling.
    if batch.shape[-1] != 4:
        batch = batch.transpose(1, 2)
    return batch.to(device, dtype=torch.float32)


def _predict_with_layout_fallback(model, batch: torch.Tensor) -> torch.Tensor:
    """Run a model, retrying NLC input as NCL for channel-first-only models."""
    try:
        return model(batch)
    except RuntimeError as exc:
        if batch.ndim == 3 and batch.shape[1] != 4 and batch.shape[-1] == 4:
            try:
                return model(batch.transpose(1, 2).contiguous())
            except RuntimeError:
                pass
        raise exc


def vep_centred_on_ds(model, ds,
                metric_func='diff', mean_by_tasks=True, 
                device=torch.device('cpu'),
                batch_size: Optional[int] = None,
                dataloader_config: Optional[Dict[str, Any]] = None):
    """
    Perform variant effect prediction on a complete dataset.

    This function:
    1. Processes all variants in the dataset
    2. Calculates effect scores using specified metric
    3. Aggregates results across all variants

    Args:
        model (nn.Module): Trained neural network model
        ds (VariantDataset): Dataset containing variants
        metric_func (str or callable): Effect score calculation method
        mean_by_tasks (bool): Whether to average across tasks
        device (str): Computation device ('cuda' or 'cpu')
        batch_size (int, optional): Batch size for batched inference.
            If None or <=1, uses the legacy per-variant path.
        dataloader_config (dict, optional): Optional DataLoader runtime args
            such as `num_workers` and `pin_memory`.

    Returns:
        np.ndarray: Effect scores for all variants
    """
    device = torch.device(device)
    model.eval()
    model.to(device)

    if not batch_size or batch_size <= 1:
        P_diff = []
        for i in range(len(ds)):
            seq_ref, seq_alt = ds[i]
            p_ref, p_alt = variant_effect_prediction(model, seq_ref, seq_alt, device=device)
            p_eff = metric_predicted_effect(p_ref, p_alt, metric_func, mean_by_tasks)
            P_diff.append(p_eff)

        if not P_diff:
            return np.array([])
        return np.concatenate(P_diff, axis=0)

    logger.info("Using batched variant effect prediction with batch_size=%s", batch_size)
    dataloader_kwargs: Dict[str, Any] = {}
    if dataloader_config:
        num_workers = int(dataloader_config.get("num_workers", 0))
        dataloader_kwargs["num_workers"] = num_workers
        dataloader_kwargs["pin_memory"] = bool(dataloader_config.get("pin_memory", False))
        if num_workers > 0:
            if "persistent_workers" in dataloader_config:
                dataloader_kwargs["persistent_workers"] = bool(dataloader_config["persistent_workers"])
            if dataloader_config.get("prefetch_factor") is not None:
                dataloader_kwargs["prefetch_factor"] = int(dataloader_config["prefetch_factor"])
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, **dataloader_kwargs)
    batch_scores = []
    with torch.inference_mode():
        for seq_ref, seq_alt in dataloader:
            if isinstance(seq_ref, np.ndarray):
                seq_ref = torch.from_numpy(seq_ref)
            if isinstance(seq_alt, np.ndarray):
                seq_alt = torch.from_numpy(seq_alt)
            split_idx = seq_ref.shape[0]
            joint_input = _to_model_input(torch.cat([seq_ref, seq_alt], dim=0), device)
            joint_output = _predict_with_layout_fallback(model, joint_input).detach().cpu().numpy()
            p_ref = joint_output[:split_idx]
            p_alt = joint_output[split_idx:]
            p_eff = metric_predicted_effect(p_ref, p_alt, metric_func, mean_by_tasks)
            batch_scores.append(p_eff)

    if not batch_scores:
        return np.array([])
    return np.concatenate(batch_scores, axis=0)


def vep_centred_from_files(model, genome_filename, bcf_filename, 
                           target_len=1000, device=torch.device('cpu'), 
                           metric_func='diff', mean_by_tasks=True, 
                           return_df=True, save_path=None,
                           batch_size: Optional[int] = None,
                           dataloader_config: Optional[Dict[str, Any]] = None):
    """
    Complete variant effect prediction pipeline from input files.

    This function provides a high-level interface for:
    1. Loading genome and variant data
    2. Setting up prediction pipeline
    3. Calculating variant effects
    4. Saving and returning results

    Args:
        model (nn.Module): Trained neural network model
        genome_filename (str): Path to reference genome
        bcf_filename (str): Path to VCF/BCF file
        target_len (int): Sequence context length
        device (str): Computation device
        metric_func (str or callable): Effect score method
        mean_by_tasks (bool): Whether to average across tasks
        return_df (bool): Whether to return DataFrame
        save_path (str, optional): Path to save results
        batch_size (int, optional): Batch size for batched variant inference.
            If omitted or <=1, uses the per-variant path.
        dataloader_config (dict, optional): Optional DataLoader runtime args
            such as `num_workers`, `pin_memory`, and `prefetch_factor`.

    Returns:
        Union[pd.DataFrame, Tuple[np.ndarray, pd.DataFrame]]:
            - If return_df=True: DataFrame with predictions
            - If return_df=False: (predictions, variant_info)

    Note:
        Results can be saved to disk and/or returned for
        further analysis.
    """
    variant_df = read_vcf(bcf_filename)
    from ..Data.Sequence import Genome
    ds = VariantDataset(Genome(genome_filename), variant_df, target_len=target_len)
    
    P_diff = vep_centred_on_ds(
        model,
        ds,
        metric_func,
        mean_by_tasks,
        device,
        batch_size=batch_size,
        dataloader_config=dataloader_config,
    )

    if save_path:
        variant_df.to_csv(os.path.join(save_path, 'variant_df.csv'), index=False)
        np.save(os.path.join(save_path, 'P_diff.npy'), P_diff)

    if return_df:
        variant_df['P_diff'] = P_diff
        return variant_df
    else:
        return P_diff, variant_df
