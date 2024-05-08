# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('./../..'))


# -- Project information -----------------------------------------------------

project = 'arbitragelab'
author = 'Hudson & Thames Quantitative Research'

# The full version, including alpha/beta/rc tags
release = "1.0.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx_copybutton',
    'myst_parser',
    'autoapi.extension',
    'releases',
]

# AUTOAPI SETTINGS
autoapi_type = 'python'
autoapi_dirs = ["../../arbitragelab"]
autoapi_root = "technical/api"
autoapi_add_toctree_entry = False
autoapi_ignore = ["*arbitragelab/network/imports*", "*arbitragelab/util/segment*"]
autoapi_options = [
    "members",
    "undoc-members",
    "inherited-members",
    "special-members",
    "show-inheritance",
    "show-module-summary",
]

suppress_warnings = ["autoapi.python_import_resolution" ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'hudsonthames_sphinx_theme'
add_module_names = False

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

html_logo = '_static/logo_white.png'
html_theme_options = {
    'logo_only': True,
    'display_version': True,
}
html_favicon = '_static/favicon_arbitragelab.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_copy_source = True

# 'releases' (changelog) settings
releases_github_path = 'hudson-and-thames/arbitragelab'
releases_unstable_prehistory = True
