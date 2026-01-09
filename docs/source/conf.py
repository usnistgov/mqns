# Configuration file for the Sphinx documentation builder.

import os.path
import sys

mqns_root = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, os.path.join(mqns_root, "examples"))
sys.path.insert(0, mqns_root)

# -- Project information

project = "MQNS"
copyright = "2025, Amar Abane"
author = "Amar Abane"

release = "0.1"
version = "0.1.0"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

autodoc_default_options = {
    # https://github.com/sphinx-doc/sphinx/issues/4961#issuecomment-1543858623
    "ignore-module-all": True
}

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
