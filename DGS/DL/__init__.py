"""
DGS Deep Learning Module

This module provides deep learning components for genomic sequence analysis:

Core Components:
1. Model Training (Trainer.py):
   - Flexible training framework
   - Checkpoint management
   - Progress monitoring
   - Early stopping

2. Model Evaluation (Evaluator.py):
   - Classification metrics
   - Regression metrics
   - Sequence-level metrics
   - Performance visualization

3. Model Interpretation (Explain.py):
   - SHAP value calculation
   - Motif enrichment analysis
   - Sequence importance visualization
   - Feature attribution

4. Variant Effect Prediction (Predict.py):
   - Variant impact scoring
   - Sequence mutation analysis
   - Batch prediction utilities
   - VCF file processing

5. Model Architecture (Architecture.py):
   - CNN architectures
   - Transformer models
   - Custom layers
   - Model configuration

6. Marginalization Tools (marginlize.py):
   - Feature importance analysis
   - Input perturbation
   - Effect size calculation
"""

from .Trainer import Trainer, TrainerMetrics, TrainerState
from .Evaluator import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    calculate_sequence_classification_metrics,
    calculate_sequence_regression_metrics,
    metrics_to_df,
    onehot_encode,
    show_auc_curve,
    show_pr_curve,
)
from .Explain import (
    DEFAULT_ATTRIBUTION_METHOD,
    Seqlet_Calling,
    calculate_attributions,
    calculate_attributions_on_ds,
    calculate_shap,
    calculate_shap_on_ds,
    motif_enrich,
    save_attribution_artifacts,
)
from .Predict import (
    VariantDataset,
    metric_predicted_effect,
    mutate,
    read_vcf,
    variant_effect_prediction,
    variants_to_intervals,
    vep_centred_from_files,
    vep_centred_on_ds,
)
from .Profile import (
    ProfileCountLoss,
    calculate_profile_metrics,
    count_targets_from_profile,
    ensure_profile_ncl,
    profile_multinomial_nll_loss,
    profile_poisson_loss,
    save_profile_predictions_h5,
    save_profile_predictions_npz,
    write_profile_predictions_bigwig,
)
from .Design import (
    SequenceDesignResult,
    gradient_ascent_sequence_design,
    greedy_ism_sequence_design,
)

__all__ = [
    "Trainer",
    "TrainerMetrics",
    "TrainerState",
    "calculate_classification_metrics",
    "calculate_regression_metrics",
    "calculate_sequence_classification_metrics",
    "calculate_sequence_regression_metrics",
    "metrics_to_df",
    "onehot_encode",
    "show_auc_curve",
    "show_pr_curve",
    "DEFAULT_ATTRIBUTION_METHOD",
    "Seqlet_Calling",
    "calculate_attributions",
    "calculate_attributions_on_ds",
    "calculate_shap",
    "calculate_shap_on_ds",
    "motif_enrich",
    "save_attribution_artifacts",
    "VariantDataset",
    "metric_predicted_effect",
    "mutate",
    "read_vcf",
    "variant_effect_prediction",
    "variants_to_intervals",
    "vep_centred_from_files",
    "vep_centred_on_ds",
    "ProfileCountLoss",
    "calculate_profile_metrics",
    "count_targets_from_profile",
    "ensure_profile_ncl",
    "profile_multinomial_nll_loss",
    "profile_poisson_loss",
    "save_profile_predictions_h5",
    "save_profile_predictions_npz",
    "write_profile_predictions_bigwig",
    "SequenceDesignResult",
    "gradient_ascent_sequence_design",
    "greedy_ism_sequence_design",
]
