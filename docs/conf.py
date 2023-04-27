import datetime

import automl_sphinx_theme
from ConfigSpace import __authors__, __version__

authors = ", ".join(__authors__)


options = {
    "copyright": f"""Copyright {datetime.date.today().strftime('%Y')}, {authors}""",
    "author": authors,
    "version": __version__,
    "name": "ConfigSpace",
    "html_theme_options": {
        "github_url": "https://github.com/automl/ConfigSpace",
        "twitter_url": "https://twitter.com/automl_org?lang=de",
    },
}

# Import conf.py from the automl theme
automl_sphinx_theme.set_options(globals(), options)

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx.ext.doctest',
]

autodoc_typehints = "description"
autoclass_content = "both"
autodoc_default_options = {
    "inherited-members": True,
}
