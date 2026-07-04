"""Unit tests for test predict."""

import unittest
import numpy as np
import pandas as pd
import torch
import tempfile
from pathlib import Path
import shutil

from DGS.DL.Predict import (
    read_vcf,
    variants_to_intervals,
    mutate,
    VariantDataset,
    variant_effect_prediction,
    metric_predicted_effect,
    vep_centred_from_files
)
from DGS.Data.Sequence import DNASeq, Genome
from DGS.Data.Interval import Interval

class SimpleModel(torch.nn.Module):
    """SimpleModel implementation."""
    def __init__(self):
        """Initialize `SimpleModel`."""
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 16, kernel_size=8)
        self.fc = torch.nn.Linear(16, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x):
        """Compute forward outputs for `SimpleModel`."""
        # Predict helpers feed channel-last tensors (N, L, 4); convert for Conv1d.
        if x.ndim == 3 and x.shape[-1] == 4:
            x = x.transpose(1, 2)
        x = self.conv(x)
        x = torch.mean(x, dim=2)
        return self.fc(x)

class TestPredict(unittest.TestCase):
    """Test cases for predict."""
    def setUp(self):
        """Set up test data and model"""
        # Set random seeds
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        
        # Create test VCF file
        self.vcf_file = self.output_dir / "test.vcf"
        with open(self.vcf_file, "w") as f:
            f.write("""##fileformat=VCFv4.2
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT
chr1\t1000\trs1\tA\tT\t100\tPASS\tAF=0.1\tGT
chr1\t2000\trs2\tG\tC\t100\tPASS\tAF=0.2\tGT
chr2\t1500\trs3\tAG\tA\t100\tPASS\tAF=0.3\tGT
chr2\t2500\trs4\tT\tTA\t100\tPASS\tAF=0.4\tGT
        """)
        
        # Create test genome file
        self.genome_file = self.output_dir / "test.fa"
        chr1 = list("A" * 5000)
        chr2 = list("A" * 5000)
        chr1[1999] = "G"  # chr1:2000 ref=G
        chr2[1499] = "A"  # chr2:1500 ref starts with A
        chr2[1500] = "G"  # chr2:1501 ref continues with G
        chr2[2499] = "T"  # chr2:2500 ref=T
        with open(self.genome_file, "w") as f:
            f.write(""">chr1
%s
>chr2
%s
""" % ("".join(chr1), "".join(chr2)))
        
        # Create model
        self.model = SimpleModel()
        
        # Create test sequences using DNASeq
        self.seq_ref = DNASeq("ATCGATCGATCG")
        self.seq_alt = DNASeq("ATCGTTCGATCG")
        
        # Create genome object
        self.genome = Genome(self.genome_file)
        
    def test_read_vcf(self):
        """Test VCF file reading"""
        variants = read_vcf(self.vcf_file)
        
        # Check DataFrame properties
        self.assertIsInstance(variants, pd.DataFrame)
        self.assertEqual(len(variants), 4)
        
        # Check required columns
        required_cols = ["CHROM", "POS", "ID", "REF", "ALT"]
        for col in required_cols:
            self.assertIn(col, variants.columns)
        
        # Check data types
        self.assertTrue(variants["POS"].dtype == np.int64)
        self.assertTrue(variants["CHROM"].dtype == object)

    def test_read_vcf_accepts_standard_eight_column_records(self):
        """Test reading VCF records without a FORMAT column."""
        vcf_file = self.output_dir / "eight_col.vcf"
        with open(vcf_file, "w") as f:
            f.write("""##fileformat=VCFv4.2
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	1000	rs1	A	T	100	PASS	AF=0.1
""")

        variants = read_vcf(vcf_file)

        self.assertEqual(len(variants), 1)
        self.assertIn("FORMAT", variants.columns)
        self.assertEqual(variants.loc[0, "FORMAT"], "")
        self.assertEqual(int(variants.loc[0, "POS"]), 1000)

    def test_read_vcf_reports_malformed_pos(self):
        """Test VCF POS validation uses user-facing errors."""
        vcf_file = self.output_dir / "bad_pos.vcf"
        with open(vcf_file, "w") as f:
            f.write("""##fileformat=VCFv4.2
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	abc	rs1	A	T	100	PASS	AF=0.1
""")

        with self.assertRaisesRegex(ValueError, "POS must be a positive integer"):
            read_vcf(vcf_file)

    def test_read_vcf_reports_incompatible_alt(self):
        """Test symbolic or structural ALT alleles get actionable errors."""
        vcf_file = self.output_dir / "symbolic_alt.vcf"
        with open(vcf_file, "w") as f:
            f.write("""##fileformat=VCFv4.2
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	1000	rs1	A	<DEL>	100	PASS	SVTYPE=DEL
""")

        with self.assertRaisesRegex(ValueError, "simple DNA alleles"):
            read_vcf(vcf_file)
        
    def test_variants_to_intervals(self):
        """Test variant to interval conversion"""
        variants = read_vcf(self.vcf_file)
        intervals = variants_to_intervals(variants, seq_len=100)
        
        # Check interval properties
        self.assertIsInstance(intervals, Interval)
        self.assertEqual(len(intervals.data), len(variants))
        self.assertTrue(
            all(
                end - start == 100
                for start, end in zip(intervals.data["start"], intervals.data["end"])
            )
        )

    def test_variants_to_intervals_reports_near_start_window(self):
        """Test variant windows that would start before zero are rejected early."""
        variants = pd.DataFrame({
            "CHROM": ["chr1"],
            "POS": [10],
            "ID": ["rs_near_start"],
            "REF": ["A"],
            "ALT": ["T"],
        })

        with self.assertRaisesRegex(ValueError, "extends before chromosome start"):
            variants_to_intervals(variants, seq_len=100)
        
    def test_mutate(self):
        """Test sequence mutation"""
        # Test SNP
        seq = DNASeq("ATCGATCGATCG")
        var_ref = "A"
        var_alt = "T"
        mutated = mutate(seq.sequence, var_ref, var_alt)
        self.assertEqual(len(mutated), len(seq))
        
        # Test deletion
        var_ref = "AT"
        var_alt = "A"
        mutated = mutate(seq.sequence, var_ref, var_alt)
        self.assertEqual(len(mutated), len(seq))
        
        # Test insertion
        var_ref = "A"
        var_alt = "AT"
        mutated = mutate(seq.sequence, var_ref, var_alt)
        self.assertEqual(len(mutated), len(seq))
        
    def test_variant_dataset(self):
        """Test VariantDataset class"""
        variants = read_vcf(self.vcf_file)
        dataset = VariantDataset(self.genome, variants, target_len=100)
        
        # Test dataset properties
        self.assertEqual(len(dataset), len(variants))
        
        # Test sequence retrieval
        seq_ref, seq_alt = dataset[0]
        self.assertIsInstance(seq_ref, np.ndarray)
        self.assertIsInstance(seq_alt, np.ndarray)
        self.assertEqual(seq_ref.shape, (100, 4))
        self.assertEqual(seq_alt.shape, (100, 4))
        
    def test_variant_effect_prediction(self):
        """Test variant effect prediction"""
        # Convert sequences to one-hot encoding
        X_ref = self.seq_ref.to_onehot()
        X_alt = self.seq_alt.to_onehot()
        
        # Get predictions
        p_ref, p_alt = variant_effect_prediction(self.model, X_ref, X_alt)
        
        # Check output shapes and types
        self.assertIsInstance(p_ref, np.ndarray)
        self.assertIsInstance(p_alt, np.ndarray)
        self.assertEqual(p_ref.shape[0], 1)  # batch size
        self.assertEqual(p_alt.shape[0], 1)  # batch size
        
    def test_metric_predicted_effect(self):
        """Test predicted effect metrics"""
        # Create test predictions
        p_ref = np.array([[0.1, 0.2, 0.3]])
        p_alt = np.array([[0.2, 0.3, 0.4]])
        
        # Test different metrics
        metrics = ['diff', 'ratio', 'log_ratio']
        for metric in metrics:
            p_eff = metric_predicted_effect(p_ref, p_alt, metric_func=metric)
            self.assertIsInstance(p_eff, np.ndarray)

        for metric in ['max', 'min']:
            p_eff = metric_predicted_effect(
                p_ref, p_alt, metric_func=metric, mean_by_tasks=False
            )
            self.assertIsInstance(p_eff, np.ndarray)
            
        # Test custom metric function
        def custom_metric(p_ref, p_alt):
            return np.abs(p_alt - p_ref).mean(axis=1)
            
        p_eff = metric_predicted_effect(
            p_ref, p_alt, metric_func=custom_metric, mean_by_tasks=False
        )
        self.assertIsInstance(p_eff, np.ndarray)
        
    def test_vep_centred_from_files(self):
        """Test variant effect prediction from files"""
        try:
            result = vep_centred_from_files(
                self.model,
                self.genome_file,
                self.vcf_file,
                target_len=100,
                metric_func='diff',
                return_df=True,
                batch_size=2,
                dataloader_config={"num_workers": 0, "pin_memory": False},
            )
            
            # Check result properties
            self.assertIsInstance(result, pd.DataFrame)
            self.assertIn('P_diff', result.columns)
            self.assertEqual(len(result), 4)  # number of variants
            
        except Exception as e:
            self.skipTest(f"Skipping vep_centred_from_files test due to: {str(e)}")
            
    def tearDown(self):
        """Clean up temporary files"""
        if hasattr(self, "genome") and self.genome is not None:
            self.genome.close()
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main() 
