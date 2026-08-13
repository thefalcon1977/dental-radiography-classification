"""Sphinx configuration for densNet."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "densNet"
author = "densNet contributors"
copyright = "2026, densNet contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_baseurl = "https://thefalcon1977.github.io/dental-radiography-classification/"

# Mock heavy ML deps so docs build without installing PyTorch in CI.
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "PIL",
    "numpy",
    "matplotlib",
    "seaborn",
    "sklearn",
    "tqdm",
]

autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
