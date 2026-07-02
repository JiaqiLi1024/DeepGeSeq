"""VCF parsing helpers used by prediction workflows.

Purpose:
    Parse core VCF columns into a pandas DataFrame for downstream processing.

Main Responsibilities:
    - Read standard VCF records while skipping metadata/comment lines.
    - Enforce stable column naming for `CHROM`, `POS`, `REF`, `ALT`, and peers.
    - Provide lightweight logging for success and failure cases.

Key Runtime Notes:
    - Only the first 9 standard VCF columns are loaded.
    - `POS` is parsed as integer and other columns are parsed as strings.
    - Parsing errors are logged and re-raised.
"""

from pathlib import Path

import pandas as pd

from . import logger

def read_vcf(filename: str) -> pd.DataFrame:
    """Read a VCF file into a pandas DataFrame.

    Args:
    filename : str
        Path to the VCF file to read

    Returns:
    pd.DataFrame
        DataFrame containing VCF records with columns:
        - CHROM: Chromosome name (str)
        - POS: Position (int)
        - ID: Variant identifier (str)
        - REF: Reference allele (str)
        - ALT: Alternate allele(s) (str)
        - QUAL: Quality score (str)
        - FILTER: Filter status (str)
        - INFO: Additional information (str)
        - FORMAT: Genotype format (str)

    Notes:
    - Comments (lines starting with #) are automatically filtered
    - Only the first 9 standard VCF columns are read
    - All columns except POS are read as strings
    - Missing values are preserved as is
    """
    path = Path(filename)
    logger.debug("Reading VCF file: %s", path)

    if not path.exists():
        raise FileNotFoundError(f"VCF file not found: {path}")
    if not path.is_file():
        raise ValueError(f"VCF path is not a file: {path}")

    names = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    required = names[:8]

    first_record_columns = None
    with path.open("r") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            first_record_columns = len(line.rstrip("\n").split("\t"))
            break

    if first_record_columns is None:
        raise ValueError(f"VCF file contains no variant records: {path}")
    if first_record_columns < len(required):
        raise ValueError(
            "Malformed VCF record: expected at least 8 tab-delimited columns "
            f"(CHROM POS ID REF ALT QUAL FILTER INFO), found {first_record_columns} in {path}."
        )

    n_cols = min(first_record_columns, len(names))
    try:
        vcf = pd.read_csv(
            path,
            delimiter="\t",
            comment="#",
            names=names[:n_cols],
            dtype=str,
            usecols=range(n_cols),
            keep_default_na=False,
        )
    except Exception as exc:
        logger.error("Failed to read VCF file %s: %s", path, exc)
        raise ValueError(f"Failed to parse VCF file {path}: {exc}") from exc

    if "FORMAT" not in vcf.columns:
        vcf["FORMAT"] = ""

    for column in required:
        missing = vcf[column].astype(str).str.strip().eq("")
        if missing.any():
            rows = (vcf.index[missing] + 1).tolist()[:5]
            raise ValueError(f"Malformed VCF: missing required '{column}' value in record row(s) {rows}.")

    pos_text = vcf["POS"].astype(str).str.strip()
    invalid_pos = ~pos_text.str.fullmatch(r"[0-9]+")
    if invalid_pos.any():
        examples = pos_text[invalid_pos].head(5).tolist()
        raise ValueError(f"Malformed VCF: POS must be a positive integer. Invalid value(s): {examples}.")
    vcf["POS"] = pos_text.astype(int)
    if (vcf["POS"] <= 0).any():
        rows = (vcf.index[vcf["POS"] <= 0] + 1).tolist()[:5]
        raise ValueError(f"Malformed VCF: POS must be >= 1 in record row(s) {rows}.")

    for allele_column in ["REF", "ALT"]:
        alleles = vcf[allele_column].astype(str).str.strip().str.upper()
        invalid = ~alleles.str.fullmatch(r"[ACGTN]+")
        if invalid.any():
            examples = alleles[invalid].head(5).tolist()
            raise ValueError(
                f"Unsupported VCF allele in '{allele_column}': {examples}. "
                "DGS variant-effect prediction currently supports simple DNA alleles "
                "containing only A/C/G/T/N; split multiallelic records and remove "
                "symbolic or breakend alleles before prediction."
            )
        vcf[allele_column] = alleles

    vcf["CHROM"] = vcf["CHROM"].astype(str).str.strip()
    logger.debug("Successfully read %s variants", len(vcf))
    return vcf
