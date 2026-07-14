import re
import sys
from pathlib import Path

from setuptools import find_packages, setup

if sys.version_info < (3, 7):
    sys.exit("DGS requires Python >= 3.7")

ROOT = Path(__file__).resolve().parent
INIT_FILE = ROOT / "DGS" / "__init__.py"
README_FILE = ROOT / "README.md"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_meta(field: str) -> str:
    pattern = rf'^__{field}__\s*=\s*["\']([^"\']+)["\']'
    match = re.search(pattern, read_text(INIT_FILE), flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"Unable to find __{field}__ in {INIT_FILE}")
    return match.group(1)


INSTALL_REQUIRES = [
    "numpy",
    "pandas>=0.21",
    "scikit-learn>=0.21.2",
    "scipy>=1.0",
    "torch>=1.10.1",
    "tensorboard>=2.7.0",
    "tqdm",
    "matplotlib>=3.0",
    "h5py>=2.10.0",
    "pysam",
    "pyBigWig",
]

EXTRAS_REQUIRE = {
    "explain": ["captum", "tangermeme"],
    "notebook": ["nbformat", "nbclient", "ipykernel"],
    "tune": ["ray[tune]>=1.12.0"],
    "llm": ["transformers>=4.0.0", "einops>=0.6.0"],
    "dev": ["pytest"],
}
EXTRAS_REQUIRE["all"] = sorted(
    {dep for deps in EXTRAS_REQUIRE.values() for dep in deps}
)

SKILL_DATA_FILES = [
    ("share/dgs/dgs-skill", ["dgs-skill/SKILL.md"]),
    ("share/dgs/dgs-skill/agents", ["dgs-skill/agents/openai.yaml"]),
    ("share/dgs/dgs-skill/references", ["dgs-skill/references/dgs-api.md"]),
    ("share/dgs/dgs-skill/scripts", ["dgs-skill/scripts/dgs_helper.py"]),
]

setup(
    name="dgs",
    version=read_meta("version"),
    description="DeepGeSeq is a deep learning package for genomic sequence analysis.",
    long_description=read_text(README_FILE),
    long_description_content_type="text/markdown",
    url="https://github.com/JiaqiLi1024/DeepGeSeq",
    author=read_meta("author"),
    author_email=read_meta("email"),
    maintainer="jiaqili@zju.edu.cn",
    maintainer_email="jiaqili@zju.edu.cn",
    license="BSD",
    python_requires=">=3.7",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    packages=find_packages(),
    include_package_data=True,
    data_files=SKILL_DATA_FILES,
    entry_points={
        "console_scripts": [
            "dgs=DGS.Main:main",
        ],
    },
    zip_safe=False,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Framework :: Jupyter",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
