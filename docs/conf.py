import os, sys
import datetime

sys.path.insert(0, os.path.abspath(".."))

import automl_sphinx_theme  # Must come after the path injection above
from ConfigSpace import __version__, __authors__

authors = ", ".join(__authors__)


options = {
    "copyright": f"""Copyright {datetime.date.today().strftime('%Y')}, {authors}""",
    "author": authors,
    "version": __version__,
    "name": "ConfigSpace",
    "html_theme_options": {
        "github_url": "https://github.com/automl/automl_sphinx_theme",
        "twitter_url": "https://twitter.com/automl_org?lang=de",
    },
}

# Import conf.py from the automl theme
automl_sphinx_theme.set_options(globals(), options)
