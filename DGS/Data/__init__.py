"""
DGS Data Processing Module

This module provides comprehensive tools for genomic data processing and management:

Core Components:
1. Interval Operations (Interval.py):
   - Genomic interval manipulation
   - Overlap detection and merging
   - Distance calculations
   - Statistical analysis

2. Sequence Processing (Sequence.py):
   - DNA sequence manipulation
   - One-hot encoding/decoding
   - Sequence complexity metrics
   - FASTA file integration

3. Target Data Management (Target.py):
   - Multi-task target handling
   - BED and BigWig support
   - Data encoding and statistics
   - Task configuration

4. Dataset Classes (Dataset.py):
   - PyTorch dataset implementations
   - Sequence extraction
   - Batch processing
   - Multi-task learning support

5. Data Sampling (Sampler.py):
   - Train/test splitting
   - Chromosome-based splitting
   - Random sampling
   - Cross-validation utilities

The module is designed for efficient processing of genomic data in deep learning
applications, with particular focus on sequence analysis tasks.
"""

from .Interval import *
from .Sequence import *
from .Target import *
from .Sampler import *
from .Dataset import *
from .Loader import *

__all__ = [
    "Interval",
    "NamedInterval",
    "find_overlaps",
    "merge_intervals",
    "find_closest",
    "get_interval_stats",
    "DNASeq",
    "Genome",
    "validate_sequence",
    "get_reverse_complement",
    "mutate_sequence",
    "sequence_to_onehot",
    "onehot_to_sequence",
    "batch_to_onehot",
    "batch_from_onehot",
    "reverse_complement",
    "one_hot_encode",
    "one_hot_decode",
    "calculate_gc_content",
    "calculate_sliding_gc",
    "calculate_complexity",
    "Target",
    "get_class_distribution",
    "get_imbalance_metrics",
    "get_rare_label_stats",
    "random_split",
    "chromosome_split",
    "SeqDataset",
    "GenomicDataset",
    "create_dataloader",
    "StreamingSeqDataset",
    "StreamingGenomicDataset",
    "StreamingProfileDataset",
    "build_sequence_dataloader",
    "build_profile_dataloader",
    "build_profile_dataloaders",
    "build_supervised_dataloader",
    "build_supervised_dataloaders",
]
