# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import sphinx_bootstrap_theme
import ConfigSpace


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'ConfigSpace'
copyright = '2014-2016, Matthias Feurer, Katharina Eggensperger, Syed Mohsin Ali, Christina Hernandez Wunsch, Julien-Charles Levesque, Jost Tobias Springenberg, Marius Lindauer'
author = 'Matthias Feurer, Katharina Eggensperger, Syed Mohsin Ali, Christina Hernandez Wunsch, Julien-Charles Levesque, Jost Tobias Springenberg, Marius Lindauer'

# The short X.Y version
version = '0.4.6'
# The full version, including alpha/beta/rc tags
release = ''


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx_gallery.gen_gallery',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

from sphinx_gallery.sorting import ExampleTitleSortKey
# Now to declare your project structure, we add a configuration dictionary
# for Sphinx-Gallery. The examples directory ../examples is declared
# with a relative path from the conf.py file location:


sphinx_gallery_conf = {
                        # path to your examples scripts
                        'examples_dirs': '../../ConfigSpace/example',
                        # path where to save gallery generated examples
                        'gallery_dirs': 'auto_examples',
                        # ignore files with this pattern.
                        'ignore_pattern': '__init__\.py|.*\.sh',
                        'within_subsection_order': ExampleTitleSortKey,
                      }

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bootstrap'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
# Insert options
    # Navigation bar title. (Default: ``project`` value)
    # 'navbar_title': "Title",

    # Tab name for entire site. (Default: "Site")
    # 'navbar_site_name': "Site",

    # A list of tuples containting pages to link to.  The value should
    # be in the form [(name, page), ..]
    'navbar_links': [
        ('Start', 'index'),
        ('Quickstart', 'quickstart'),
        ('Advanced Example', 'AdvancedExample'),
        ('Serialization', 'serialization'),
        ('Hyperparameters', 'hyperparameter'),
        ('Constraints', 'constraints'),
    ],
    # Render the next and previous page links in navbar. (Default: true)
    'navbar_sidebarrel': False,

    # Render the current pages TOC in the navbar. (Default: true)
    'navbar_pagenav': False,

    # Tab name for the current pages TOC. (Default: "Page")
    'navbar_pagenav_name': "On this page",

    # Global TOC depth for "site" navbar tab. (Default: 1)
    # Switching to -1 shows all levels.
    'globaltoc_depth': 2,

    # Include hidden TOCs in Site navbar?
    #
    # Note: If this is "false", you cannot have mixed ``:hidden:`` and
    # non-hidden ``toctree`` directives in the same page, or else the build
    # will break.
    #
    # Values: "true" (default) or "false"
    'globaltoc_includehidden': "false",

    # HTML navbar class (Default: "navbar") to attach to <div> element.
    # For black navbar, do "navbar navbar-inverse"
    'navbar_class': "navbar",

    # Fix navigation bar to top of page?
    # Values: "true" (default) or "false"
    'navbar_fixed_top': "true",

    # Location of link to source.
    # Options are "nav" (default), "footer" or anything else to exclude.
    'source_link_position': "footer",

    # Bootswatch (http://bootswatch.com/) theme.
    #
    # Options are nothing with "" (default) or the name of a valid theme
    # such as "amelia" or "cosmo".
    'bootswatch_theme': "cosmo",

    # Choose Bootstrap version.
    # Values: "3" (default) or "2" (in quotes)
    'bootstrap_version': "3",
}

html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
# using_rtd_theme = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = "BOHB.png"


# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
html_sidebars = {'**': ['localtoc.html']}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'ConfigSpacedoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'ConfigSpace.tex', 'ConfigSpace Documentation',
     'Matthias Feurer, Katharina Eggensperger, Syed Mohsin Ali, Christina Hernandez Wunsch, Julien-Charles Levesque, Jost Tobias Springenberg, Marius Lindauer', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'configspace', 'ConfigSpace Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'ConfigSpace', 'ConfigSpace Documentation',
     author, 'ConfigSpace', 'One line description of project.',
     'Miscellaneous'),
]


# -- Extension configuration -------------------------------------------------
# Show init as well as moduledoc
autoclass_content = 'both'