"""Setup.py for ConfigSpace"""

import os

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


# Helper functions
def read_file(fname):
    """Get contents of file from the modules directory"""
    return open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read()


def get_version(fname):
    """Get the module version"""
    with open(fname, encoding='utf-8') as file_handle:
        return file_handle.readlines()[-1].split()[-1].strip("\"'")


class BuildExt(build_ext):
    """ build_ext command for use when numpy headers are needed.
    SEE tutorial: https://stackoverflow.com/questions/2379898
    SEE fix: https://stackoverflow.com/questions/19919905
    """

    def finalize_options(self):
        build_ext.finalize_options(self)
        import numpy
        self.include_dirs.append(numpy.get_include())


# Configure setup parameters
MODULE_NAME = 'ConfigSpace'
MODULE_URL = 'https://github.com/automl/ConfigSpace'
SHORT_DESCRIPTION = (
    'Creation and manipulation of parameter configuration spaces for '
    'automated algorithm configuration and hyperparameter tuning.'
)
KEYWORDS = (
    'algorithm configuration hyperparameter optimization empirical '
    'evaluation black box'
)
LICENSE = 'BSD 3-clause'
PLATS = ['Linux']
AUTHORS = ', '.join(["Matthias Feurer", "Katharina Eggensperger",
                     "Syed Mohsin Ali", "Christina Hernandez Wunsch",
                     "Julien-Charles Levesque", "Jost Tobias Springenberg", "Philipp Mueller"
                     "Marius Lindauer", "Jorn Tuyls"]),
AUTHOR_EMAIL = 'feurerm@informatik.uni-freiburg.de'
TEST_SUITE = "pytest"
SETUP_REQS = ['numpy', 'cython']
INSTALL_REQS = ['numpy', 'cython', 'pyparsing']
MIN_PYTHON_VERSION = '>=3.7'
CLASSIFIERS = ['Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8',
               'Programming Language :: Python :: 3.9',
               'Development Status :: 4 - Beta',
               'Natural Language :: English',
               'Intended Audience :: Developers',
               'Intended Audience :: Education',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: BSD License',
               'Operating System :: POSIX :: Linux',
               'Topic :: Scientific/Engineering :: Artificial Intelligence',
               'Topic :: Scientific/Engineering',
               'Topic :: Software Development']

# These do not really change the speed of the benchmarks
COMPILER_DIRECTIVES = {
    'boundscheck': False,
    'wraparound': False,
}

EXTENSIONS = [Extension('ConfigSpace.hyperparameters',
                        sources=['ConfigSpace/hyperparameters.pyx']),
              Extension('ConfigSpace.forbidden',
                        sources=['ConfigSpace/forbidden.pyx']),
              Extension('ConfigSpace.conditions',
                        sources=['ConfigSpace/conditions.pyx']),
              Extension('ConfigSpace.c_util',
                        sources=['ConfigSpace/c_util.pyx']),
              Extension('ConfigSpace.util',
                        sources=['ConfigSpace/util.pyx']),
              Extension('ConfigSpace.configuration_space',
                        sources=['ConfigSpace/configuration_space.pyx'])]

for e in EXTENSIONS:
    e.cython_directives = COMPILER_DIRECTIVES

extras_reqs = {
    "test": [
        "pytest>=4.6",
        "mypy",
        "pre-commit",
        "pytest-cov",
    ],
    "docs": ["sphinx", "sphinx-gallery", "sphinx_bootstrap_theme", "numpydoc"],
}


setup(
    name=MODULE_NAME,
    version=get_version('ConfigSpace/__version__.py'),
    cmdclass={'build_ext': BuildExt},
    url=MODULE_URL,
    description=SHORT_DESCRIPTION,
    long_description_content_type='text/markdown',
    ext_modules=EXTENSIONS,
    long_description=read_file("README.md"),
    license=LICENSE,
    platforms=PLATS,
    author=AUTHORS,
    author_email=AUTHOR_EMAIL,
    test_suite=TEST_SUITE,
    setup_requires=SETUP_REQS,
    install_requires=INSTALL_REQS,
    extras_require=extras_reqs,
    keywords=KEYWORDS,
    packages=find_packages(),
    python_requires=MIN_PYTHON_VERSION,
    classifiers=CLASSIFIERS,
)
