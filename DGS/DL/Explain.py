"""Attribution and motif analysis helpers for trained models.

Purpose:
    Generate sequence attributions and downstream motif/seqlet summaries.

Main Responsibilities:
    - Compute DeepLIFT/SHAP-style and Captum attributions for tensors and datasets.
    - Export attribution artifacts for TF-MoDISco-lite workflows.
    - Run optional seqlet calling and motif annotation pipelines.

Key Runtime Notes:
    - Requires `tangermeme` for legacy DeepLIFT/SHAP and seqlet operations.
    - Requires `captum` for Captum DeepLift and Integrated Gradients.
    - Motif enrichment/report generation additionally requires `modisco` CLI.
    - Batched attribution mode is available through `batch_size` parameters.
"""

import os
import logging
import subprocess
import shutil
from typing import Optional

import torch
import numpy as np
from torch.utils.data import DataLoader

try:
    from tangermeme.deep_lift_shap import deep_lift_shap
    from tangermeme.seqlet import recursive_seqlets
    from tangermeme.annotate import annotate_seqlets
    from tangermeme.io import read_meme
    _TANGERMEME_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised in dependency-missing envs
    deep_lift_shap = None
    recursive_seqlets = None
    annotate_seqlets = None
    read_meme = None
    _TANGERMEME_IMPORT_ERROR = exc

try:
    from captum.attr import DeepLift, IntegratedGradients
    _CAPTUM_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised in dependency-missing envs
    DeepLift = None
    IntegratedGradients = None
    _CAPTUM_IMPORT_ERROR = exc

logger = logging.getLogger("dgs.explain")

DEFAULT_ATTRIBUTION_METHOD = "deeplift_shap"
_CAPTUM_METHODS = {"deeplift", "integrated_gradients"}


def _normalize_attribution_method(method: str) -> str:
    """Normalize public attribution method aliases."""
    if method is None:
        return DEFAULT_ATTRIBUTION_METHOD

    normalized = method.lower().replace("-", "_")
    aliases = {
        "deep_lift_shap": "deeplift_shap",
        "deep_lift": "deeplift",
        "integrated_gradient": "integrated_gradients",
        "ig": "integrated_gradients",
    }
    normalized = aliases.get(normalized, normalized)
    supported = {DEFAULT_ATTRIBUTION_METHOD, "deeplift", "integrated_gradients"}
    if normalized not in supported:
        raise ValueError(
            "Unsupported attribution method "
            f"'{method}'. Supported methods are: "
            "'deeplift_shap', 'deeplift', 'integrated_gradients'/'ig'."
        )
    return normalized


def _ensure_explain_dependencies(
    require_modisco: bool = False,
    method: str = DEFAULT_ATTRIBUTION_METHOD,
    require_tangermeme: bool = False,
) -> None:
    """Raise clear runtime errors for optional explain dependencies."""
    method = _normalize_attribution_method(method)
    if (require_tangermeme or method == DEFAULT_ATTRIBUTION_METHOD) and _TANGERMEME_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Explain mode requires optional dependency 'tangermeme'. "
            "Install it with `pip install -e \".[explain]\"` or `pip install tangermeme`."
        ) from _TANGERMEME_IMPORT_ERROR
    if method in _CAPTUM_METHODS and _CAPTUM_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Captum attribution methods require optional dependency 'captum'. "
            "Install it with `pip install -e \".[explain]\"` or `pip install captum`."
        ) from _CAPTUM_IMPORT_ERROR
    if require_modisco and shutil.which("modisco") is None:
        raise RuntimeError(
            "Explain mode requires the `modisco` CLI in PATH for motif workflows."
        )


def _to_batch_input(data) -> torch.Tensor:
    """Convert sample/batch data to (N, 4, L) float tensor."""
    if isinstance(data, (tuple, list)):
        data = data[0]

    if isinstance(data, np.ndarray):
        x = torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        x = data
    else:
        x = torch.tensor(data)

    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.shape[1] != 4:
        x = x.transpose(1, 2)
    return x.float()


def _resolve_baseline(X: torch.Tensor, baseline):
    """Return a Captum baseline tensor. Currently only zero baseline is supported."""
    if baseline is None:
        return torch.zeros_like(X)
    if isinstance(baseline, str) and baseline.lower() in {"zero", "zeros"}:
        return torch.zeros_like(X)
    if isinstance(baseline, (int, float)) and baseline == 0:
        return torch.zeros_like(X)
    raise ValueError("Only zero baselines are currently supported for Captum attributions.")


def _ensure_ncl_array(values, name: str) -> np.ndarray:
    """Return a float array in DGS attribution convention: (N, 4, L)."""
    array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D array with shape (N, 4, L) or (N, L, 4).")
    if array.shape[1] == 4:
        return array.astype(np.float32, copy=False)
    if array.shape[2] == 4:
        return np.swapaxes(array, 1, 2).astype(np.float32, copy=False)
    raise ValueError(f"{name} must include exactly four nucleotide channels.")


def save_attribution_artifacts(
    output_path,
    sequences,
    attributions,
    method: str = DEFAULT_ATTRIBUTION_METHOD,
    target: Optional[int] = None,
):
    """Save sequence and attribution arrays as ``.npz`` or ``.h5`` artifacts.

    Args:
        output_path: Output path ending in ``.npz``, ``.h5``, or ``.hdf5``.
        sequences: Nucleotide sequence array.
        attributions: Attribution array matching ``sequences`` shape.
        method: Attribution method name recorded in output metadata.
        target: Optional target index recorded as metadata.

    Returns:
        Path to the written artifact.

    Arrays are stored using the DGS attribution convention ``(N, 4, L)`` with
    dataset keys ``sequences`` and ``attributions``. ``.npz`` works with the
    base installation; ``.h5`` additionally requires ``h5py``.
    """
    sequences = _ensure_ncl_array(sequences, "sequences")
    attributions = _ensure_ncl_array(attributions, "attributions")
    if sequences.shape != attributions.shape:
        raise ValueError(
            "sequences and attributions must have the same shape; "
            f"got {sequences.shape} and {attributions.shape}."
        )

    output_path = os.fspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    method = _normalize_attribution_method(method)
    metadata = {
        "method": method,
        "shape_convention": "NCL",
        "shape": np.asarray(sequences.shape, dtype=np.int64),
    }
    if target is not None:
        metadata["target"] = int(target)

    suffix = os.path.splitext(output_path)[1].lower()
    if suffix == ".npz":
        np.savez_compressed(output_path, sequences=sequences, attributions=attributions, **metadata)
    elif suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError("Saving attribution artifacts as HDF5 requires optional dependency 'h5py'.") from exc
        with h5py.File(output_path, "w") as handle:
            handle.create_dataset("sequences", data=sequences, compression="gzip")
            handle.create_dataset("attributions", data=attributions, compression="gzip")
            handle.attrs["method"] = method
            handle.attrs["shape_convention"] = "NCL"
            if target is not None:
                handle.attrs["target"] = int(target)
    else:
        raise ValueError("output_path must end with .npz, .h5, or .hdf5.")

    return output_path


def calculate_attributions(
    model,
    X,
    target,
    device,
    method: str = DEFAULT_ATTRIBUTION_METHOD,
    baseline="zero",
    n_steps: int = 50,
    internal_batch_size: Optional[int] = None,
):
    """
    Calculate nucleotide-resolution attribution scores for model predictions.

    Args:
        model (nn.Module): Trained neural network model.
        X (torch.Tensor or np.ndarray): Input sequences in one-hot format,
            either (N, 4, L) or (N, L, 4).
        target (int): Target task index for multi-task models.
        device (str or torch.device): Computation device.
        method (str): Attribution method. Supported values are
            ``"deeplift_shap"`` (legacy tangermeme), ``"deeplift"``
            (Captum), and ``"integrated_gradients"``/``"ig"`` (Captum).
        baseline: Baseline for Captum methods. Only zero baselines are
            currently supported.
        n_steps (int): Number of integration steps for Integrated Gradients.
        internal_batch_size (int, optional): Captum internal batch size for
            Integrated Gradients.

    Returns:
        np.ndarray: Attribution scores with shape (N, 4, L).
    """
    method = _normalize_attribution_method(method)
    _ensure_explain_dependencies(method=method)

    model.eval()
    model.to(device)
    X = _to_batch_input(X).to(device)

    if method == DEFAULT_ATTRIBUTION_METHOD:
        X_attr = deep_lift_shap(model, X, target=target)
    elif method == "deeplift":
        attribution = DeepLift(model)
        baselines = _resolve_baseline(X, baseline)
        with torch.enable_grad():
            X_attr = attribution.attribute(X, baselines=baselines, target=target)
    else:
        attribution = IntegratedGradients(model)
        baselines = _resolve_baseline(X, baseline)
        with torch.enable_grad():
            X_attr = attribution.attribute(
                X,
                baselines=baselines,
                target=target,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
            )

    return _ensure_ncl_array(X_attr, "attributions")


def calculate_shap(model, X, target, device):
    """
    Calculate SHAP (SHapley Additive exPlanations) attributions for model predictions.

    This function computes importance scores for each position in the input sequences
    using the DeepLIFT algorithm adapted for SHAP values.

    Args:
        model (nn.Module): Trained neural network model
        X (torch.Tensor): Input sequences in one-hot encoded format (N, 4, L)
        target (int): Target task index for multi-task models
        device (str): Computation device ('cuda' or 'cpu')

    Returns:
        np.ndarray: Attribution scores with shape matching input (N, 4, L)

    Note:
        The function automatically handles device placement and fallback behavior.
        If attribution computation fails, an error is printed and zero-valued
        attributions are returned with the same shape as input.
    """

    _ensure_explain_dependencies(method=DEFAULT_ATTRIBUTION_METHOD)

    X = _to_batch_input(X).to(device)

    try:
        X_attr = calculate_attributions(
            model,
            X,
            target,
            device,
            method=DEFAULT_ATTRIBUTION_METHOD,
        )
    except Exception as e:
        logger.error("Error calculating SHAP attributions: %s", e)
        X_attr = np.zeros_like(X.cpu().numpy())
    
    return X_attr


def calculate_attributions_on_ds(
    model,
    ds,
    target,
    device,
    batch_size: Optional[int] = None,
    method: str = DEFAULT_ATTRIBUTION_METHOD,
    baseline="zero",
    n_steps: int = 50,
    internal_batch_size: Optional[int] = None,
):
    """
    Calculate attribution scores for an entire dataset.

    Args:
        model (nn.Module): Trained neural network model.
        ds (Dataset): Dataset containing sequences.
        target (int): Target task index for multi-task models.
        device (str or torch.device): Computation device.
        batch_size (int, optional): Batch size for attribution inference.
            If omitted or <=1, samples are processed one-by-one.
        method (str): Attribution method. See :func:`calculate_attributions`.
        baseline: Baseline for Captum methods. Only zero baselines are
            currently supported.
        n_steps (int): Number of integration steps for Integrated Gradients.
        internal_batch_size (int, optional): Captum internal batch size for
            Integrated Gradients.

    Returns:
        tuple: (sequences, attributions)
            - sequences: Original sequences in one-hot format (N, 4, L)
            - attributions: Attribution values for each sequence (N, 4, L)
    """
    method = _normalize_attribution_method(method)
    _ensure_explain_dependencies(method=method)

    X, X_attr = [], []
    if not batch_size or batch_size <= 1:
        for i in range(len(ds)):
            x = _to_batch_input(ds[i])
            x_attr = calculate_attributions(
                model,
                x,
                target,
                device,
                method=method,
                baseline=baseline,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
            )
            X.append(_ensure_ncl_array(x, "sequences"))
            X_attr.append(x_attr)
    else:
        logger.info(
            "Using batched %s attribution calculation with batch_size=%s",
            method,
            batch_size,
        )
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        for batch in dataloader:
            x = _to_batch_input(batch)
            x_attr = calculate_attributions(
                model,
                x,
                target,
                device,
                method=method,
                baseline=baseline,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
            )
            X.append(_ensure_ncl_array(x, "sequences"))
            X_attr.append(x_attr)

    X = np.concatenate(X, axis=0)
    X_attr = np.concatenate(X_attr, axis=0)

    return X, X_attr


def calculate_shap_on_ds(model, ds, target, device, batch_size: Optional[int] = None):
    """
    Calculate SHAP attributions for an entire dataset.

    This function processes a dataset in batches, handling various input formats
    and ensuring consistent tensor shapes.

    Args:
        model (nn.Module): Trained neural network model
        ds (Dataset): Dataset containing sequences
        target (int): Target task index for multi-task models
        device (str): Computation device ('cuda' or 'cpu')
        batch_size (int, optional): Batch size for attribution inference.
            If omitted or <=1, samples are processed one-by-one.

    Returns:
        tuple: (sequences, attributions)
            - sequences: Original sequences in one-hot format (N, 4, L)
            - attributions: SHAP values for each sequence (N, 4, L)
    """
    return calculate_attributions_on_ds(
        model,
        ds,
        target,
        device,
        batch_size=batch_size,
        method=DEFAULT_ATTRIBUTION_METHOD,
    )


def motif_enrich(
    model,
    ds,
    target,
    output_dir="motif_results",
    max_seqlets=2000,
    device=torch.device("cpu"),
    batch_size: Optional[int] = None,
    method: str = DEFAULT_ATTRIBUTION_METHOD,
    baseline="zero",
    n_steps: int = 50,
    internal_batch_size: Optional[int] = None,
):
    """
    Perform comprehensive motif enrichment analysis using model interpretations.

    This function:
    1. Calculates SHAP attributions for input sequences
    2. Identifies important sequence patterns
    3. Runs TF-MoDISco-lite for motif discovery
    4. Generates visualization reports

    Args:
        model (nn.Module): Trained neural network model
        ds (Dataset): Dataset containing sequences
        target (int): Target task index for multi-task models
        output_dir (str): Directory to save analysis results
        max_seqlets (int): Maximum number of sequence elements to analyze
        device (str): Computation device ('cuda' or 'cpu')
        batch_size (int, optional): Batch size for attribution inference.
        method (str): Attribution method. Supported values are
            ``"deeplift_shap"``, ``"deeplift"``, and
            ``"integrated_gradients"``/``"ig"``.
        baseline: Baseline for Captum methods. Only zero baselines are
            currently supported.
        n_steps (int): Number of integration steps for Integrated Gradients.
        internal_batch_size (int, optional): Captum internal batch size for
            Integrated Gradients.

    Returns:
        str: Path to generated motifs file

    Raises:
        subprocess.CalledProcessError:
            If `modisco motifs` or `modisco report` command fails.

    Note:
        Runtime requirements:
        - `tangermeme` must be importable for `method="deeplift_shap"`.
        - `captum` must be importable for Captum methods.
        - `modisco` command must be available in PATH.

        Results include:
        - Sequence attributions (NPZ format)
        - Discovered motifs (MEME format)
        - Visualization reports (HTML/PDF)
        - Raw data for further analysis
    """
    
    method = _normalize_attribution_method(method)
    _ensure_explain_dependencies(require_modisco=True, method=method)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
        
    # Calculate attribution scores
    logger.info("Calculating %s attributions...", method)
    X, X_attr = calculate_attributions_on_ds(
        model,
        ds,
        target=target,
        device=device,
        batch_size=batch_size,
        method=method,
        baseline=baseline,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )
    
    # Save one-hot encoded sequences and attributions
    logger.info("Saving sequences and attributions...")
    ohe_path = os.path.join(output_dir, "ohe.npz")
    shap_path = os.path.join(output_dir, "shap.npz")
    
    np.savez_compressed(ohe_path, X, sequences=X)
    np.savez_compressed(shap_path, X_attr, attributions=X_attr)
    save_attribution_artifacts(
        os.path.join(output_dir, "attributions.npz"),
        X,
        X_attr,
        method=method,
        target=target,
    )
    
    # Run TF-MoDISco-lite
    logger.info("Running TF-MoDISco-lite...")
    modisco_output = os.path.join(output_dir, "modisco_results.h5")
    motifs_output = os.path.join(output_dir, "motifs.txt")
    
    # Run modisco motifs command
    cmd = f"modisco motifs -s {ohe_path} -a {shap_path} -n {max_seqlets} -o {modisco_output}"
    subprocess.run(cmd, shell=True, check=True)
    
    # Generate report and motifs.txt
    cmd = f"modisco report -i {modisco_output} -o {output_dir} -s {output_dir}"
    subprocess.run(cmd, shell=True, check=True)
    
    logger.info(f"Motif analysis complete. Results saved in {output_dir}")
    return motifs_output

def Seqlet_Calling(model, ds, target, output_dir="seqlet_results", motif_db=None, device=torch.device("cpu")):
    """
    Identify and annotate regulatory elements (seqlets) in sequences.

    This function performs:
    1. Attribution calculation for sequences
    2. Seqlet identification using recursive algorithm
    3. Motif annotation if database provided
    4. Statistical significance assessment

    Args:
        model (nn.Module): Trained neural network model
        ds (Dataset): Dataset containing sequences
        target (int): Target task index for multi-task models
        output_dir (str): Directory to save results
        motif_db (str, optional): Path to MEME format motif database
        device (str): Computation device ('cuda' or 'cpu')

    Returns:
        DataFrame: Identified seqlets with annotations
            Columns include:
            - Sequence coordinates
            - Importance scores
            - Motif matches (if database provided)
            - Statistical significance

    Raises:
        FileNotFoundError:
            Potentially raised by downstream motif readers if motif files are missing.

    Note:
        Runtime requirements:
        - `tangermeme` must be importable.
        - If motif annotation is requested, `motif_db` should point to a valid
          MEME-format motif file.

        Results are saved in BED format for compatibility with
        genome browsers and downstream analysis tools.
    """
    
    _ensure_explain_dependencies()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate DeepLIFT/SHAP attributions
    logger.info("Calculating DeepLIFT/SHAP attributions...")
    X, X_attr = calculate_shap_on_ds(model, ds, target=target, device=device)
    
    # Call seqlets using recursive algorithm
    logger.info("Calling seqlets...")
    seqlets = recursive_seqlets(np.sum(X_attr, axis=1))  # Sum across channels for overall importance
    
    # Save seqlets information
    seqlets.to_csv(os.path.join(output_dir, "seqlets.bed"), sep="\t", header=False, index=False)
    
    # Annotate seqlets if motif database is provided
    if motif_db is not None and os.path.exists(motif_db):
        logger.info("Annotating seqlets with motif database...")
        motifs = read_meme(motif_db)
        motif_idxs, motif_pvalues = annotate_seqlets(X, seqlets, motifs)
        
        # Save annotation results
        seqlets['motif_indices'] = motif_idxs
        seqlets['motif_pvalues'] = motif_pvalues
        seqlets.to_csv(os.path.join(output_dir, "seqlets.bed"), sep="\t", header=False, index=False)
    
    logger.info(f"Seqlet analysis complete. Results saved in {output_dir}")
    return seqlets
