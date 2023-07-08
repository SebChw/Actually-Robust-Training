# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "art"
copyright = "2023, Sebastian Chwilczynski"
author = "Sebastian Chwilczynski"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "autodoc2",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
]

autodoc2_packages = [
    "../../art",
]

# You need custom extensions to allow fancy stuff like math or images
myst_enable_extensions = ["dollarmath", "amsmath", "html_image"]
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
