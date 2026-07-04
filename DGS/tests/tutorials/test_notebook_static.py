"""Static and optional execution checks for tutorial notebooks.

The tutorial notebooks are manuscript-aligned artifacts, so these tests do not
rewrite or simplify them. Full execution is opt-in and runs only when the
original data/assets and optional runtime dependencies are available.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TUTORIAL_ROOT = REPO_ROOT / "Tutorials"
NOTEBOOK_RUNTIME_PACKAGES = ("nbformat", "nbclient", "ipykernel")

TUTORIAL_EXECUTION_REQUIREMENTS = {
    "0_DGS_usage_example/dgs_minimal.ipynb": {
        "files": [
            "complete_config.json",
            "Test/reference_grch38p13/GRCh38.p13.genome.fa.gz",
            "Test/random_regions.bed",
            "Test/hg38.gc5Base.bw",
            "Test/recombAvg.bw",
            "Test/test.vcf",
        ],
        "packages": [],
        "commands": [],
    },
    "1_DGS_workflow_on_synthetic_dataset/0_Generate_synthetic_dataset.ipynb": {
        "files": [
            "Dataset_compare/JASPAR2022_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
        ],
        "packages": ["deepomics"],
        "commands": [],
    },
    "1_DGS_workflow_on_synthetic_dataset/1_DGS_workflow_demonstration_on_synthetic_dataset.ipynb": {
        "files": [
            "../Dataset/synthetic_dataset_simple.h5",
        ],
        "packages": ["tangermeme"],
        "commands": ["modisco"],
    },
    "2_DGS_reproduce_DeepSEA/2_DGS_reproduce_DeepSEA.ipynb": {
        "files": [
            "Dataset.sciATAC1_train_test.h5",
        ],
        "packages": ["tangermeme"],
        "commands": ["modisco"],
    },
    "2_DGS_reproduce_DeepSEA/3_DGS_SeqAnalysis_DeepSEA.ipynb": {
        "files": [
            "../Dataset/Dataset.DeepSEA_919features.h5",
            "../../PHD_works/DeepZJ/pretrain_deepSea/data/DeepSEA.metric.tsv",
            "../2_DGS_CaseStudy_DeepSEA/DGS_DeepSEA_DeepSEA/Log/best_model.pth",
            "../../Resource/Reference/Human_hg19/Homo_sapiens.GRCh37.dna.toplevel.fa.gz",
            "../Dataset/VEP_data/lt0.05.vcf",
            "../Dataset/VEP_data/gt0.50.vcf",
        ],
        "packages": ["statsmodels"],
        "commands": [],
    },
    "3_DGS_seqAnalysis_sciATAC/2_DGS_reproduce_published_models(DeepSEA)_on_Dataset(sci-ATAC1).ipynb": {
        "files": [
            "Dataset.sciATAC1_train_test.h5",
        ],
        "packages": ["tangermeme"],
        "commands": ["modisco"],
    },
    "4_DGS_single_cell_data/4_DGS_single_cell.ipynb": {
        "files": [
            "data_scbasset_tutorial/Dataset.buen_ad_sc.X.h5",
            "data_scbasset_tutorial/buen_ad_sc.h5ad",
        ],
        "packages": ["anndata"],
        "commands": [],
    },
    "4_DGS_single_cell_data/CellEmb.ipynb": {
        "files": [
            "../data_scbasset_tutorial/buen_ad_sc.h5ad",
            "Log/chekc_model.pth",
        ],
        "packages": ["anndata", "scanpy"],
        "commands": [],
    },
    "5_DGS_MPRA/5_DGS_MPRA.ipynb": {
        "files": [
            "data_MPRA/Dataset.CRE_Multi.h5",
        ],
        "packages": ["logomaker"],
        "commands": [],
    },
}


def _tutorial_notebooks():
    return sorted(
        path
        for path in TUTORIAL_ROOT.rglob("*.ipynb")
        if not path.name.endswith(".bak.ipynb")
    )


def _missing_python_packages(packages):
    return [package for package in packages if importlib.util.find_spec(package) is None]


def _missing_notebook_runtime_packages():
    return _missing_python_packages(NOTEBOOK_RUNTIME_PACKAGES)


def _execution_requirements_for(path):
    return TUTORIAL_EXECUTION_REQUIREMENTS.get(path.relative_to(TUTORIAL_ROOT).as_posix(), {})


def _missing_execution_requirements():
    missing = []
    for path in _tutorial_notebooks():
        requirements = _execution_requirements_for(path)
        for package in requirements.get("packages", []):
            if importlib.util.find_spec(package) is None:
                missing.append(f"{path.relative_to(REPO_ROOT)} requires Python package `{package}`")
        for command in requirements.get("commands", []):
            if not _command_exists(command):
                missing.append(f"{path.relative_to(REPO_ROOT)} requires command `{command}` in PATH")
        for relative in requirements.get("files", []):
            resolved = (path.parent / relative).resolve()
            if not resolved.exists():
                missing.append(f"{path.relative_to(REPO_ROOT)} requires file `{relative}`")
    return missing


def _command_exists(command):
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if directory and (Path(directory) / command).exists():
            return True
    return False


def test_all_tutorial_notebooks_have_valid_json_structure():
    notebooks = _tutorial_notebooks()

    assert notebooks, "No tutorial notebooks found."
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook.get("nbformat") == 4, path
        assert isinstance(notebook.get("metadata", {}), dict), path
        cells = notebook.get("cells")
        assert isinstance(cells, list) and cells, path

        for index, cell in enumerate(cells):
            cell_id = f"{path} cell {index}"
            assert cell.get("cell_type") in {"markdown", "code", "raw"}, cell_id
            assert "source" in cell, cell_id
            assert isinstance(cell.get("metadata", {}), dict), cell_id
            if cell.get("cell_type") == "code":
                assert "execution_count" in cell, cell_id
                assert isinstance(cell.get("outputs", []), list), cell_id


def test_notebook_execution_runtime_dependencies_are_available_when_requested():
    if os.environ.get("DGS_RUN_TUTORIAL_NOTEBOOKS") != "1":
        pytest.skip("Set DGS_RUN_TUTORIAL_NOTEBOOKS=1 to execute tutorials.")

    missing = _missing_notebook_runtime_packages()
    if missing:
        pytest.skip(f"Notebook execution dependencies are missing: {missing}")


def test_tutorial_source_assets_are_available_when_required():
    """Report all missing manuscript tutorial assets in one place.

    Set ``DGS_REQUIRE_TUTORIAL_ASSETS=1`` to make missing assets fail CI. The
    default is non-failing because several notebooks intentionally depend on
    external manuscript datasets that are not stored in the git repository.
    """
    missing = _missing_execution_requirements()
    if missing and os.environ.get("DGS_REQUIRE_TUTORIAL_ASSETS") == "1":
        raise AssertionError("Missing tutorial execution requirements:\n" + "\n".join(missing))
    if missing:
        pytest.skip("Tutorial source execution requirements are missing:\n" + "\n".join(missing))


def test_execute_tutorial_notebooks_when_explicitly_enabled(tmp_path):
    if os.environ.get("DGS_RUN_TUTORIAL_NOTEBOOKS") != "1":
        pytest.skip("Set DGS_RUN_TUTORIAL_NOTEBOOKS=1 to execute tutorials.")

    missing = _missing_notebook_runtime_packages()
    if missing:
        pytest.skip(f"Notebook execution dependencies are missing: {missing}")
    missing_requirements = _missing_execution_requirements()
    if missing_requirements:
        pytest.skip(
            "Tutorial notebooks were not executed because original source assets "
            "or optional dependencies are missing:\n" + "\n".join(missing_requirements)
        )

    import nbformat
    from nbclient import NotebookClient

    previous_pythonpath = os.environ.get("PYTHONPATH")
    previous_path = os.environ.get("PATH", "")
    os.environ["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not previous_pythonpath
        else f"{REPO_ROOT}{os.pathsep}{previous_pythonpath}"
    )
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{previous_path}"

    for path in _tutorial_notebooks():
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            allow_errors=False,
            resources={"metadata": {"path": str(path.parent)}},
        )
        try:
            client.execute()
        except Exception as exc:
            raise AssertionError(f"Tutorial notebook failed during execution: {path}") from exc
