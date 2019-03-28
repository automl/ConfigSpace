"""Setup.py for ConfigSpace"""

import os
from setuptools import setup, find_packages
from setuptools.extension import Extension

try:
    import numpy as np
except ImportError:
    print("Numpy module not found. Attempting to install for user using pip..")
    from pip import main as pip
    pip(['install', '--user', 'numpy'])
    import numpy as np

# Read http://peterdowns.com/posts/first-time-with-pypi.html to figure out how
# to publish the package on PyPI

# Helper functions
def read_file(fname):
    """Get contents of file from the modules directory"""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()
def get_version(fname):
    """Get the module version"""
    with open(fname) as file_handle:
        return file_handle.readlines()[-1].split()[-1].strip("\"'")

# Configure setup parameters
MODULE_NAME = 'ConfigSpace'
MODULE_URL = 'https://github.com/automl/ConfigSpace'
SHORT_DESCRIPTION = 'Creation and manipulation of parameter configuration spaces for ' \
       'automated algorithm configuration and hyperparameter tuning.'
KEYWORDS = 'algorithm configuration hyperparameter optimization empirical ' \
           'evaluation black box'
LICENSE = 'BSD 3-clause'
PLATS = ['Linux']
AUTHORS = ', '.join(["Matthias Feurer", "Katharina Eggensperger",
                     "Syed Mohsin Ali", "Christina Hernandez Wunsch",
                     "Julien-Charles Levesque", "Jost Tobias Springenberg", "Philipp Mueller"
                     "Marius Lindauer", "Jorn Tuyls"]),
AUTHOR_EMAIL = 'feurerm@informatik.uni-freiburg.de'
TEST_SUITE = "pytest"
SETUP_REQS = ['Cython']
INSTALL_REQS = ['numpy', 'pyparsing', 'typing', 'Cython']
MIN_PYTHON_VERSION = '>=3.4.*'
CLASSIFIERS = ['Programming Language :: Python :: 3.4',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
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
                        sources=['ConfigSpace/hyperparameters.pyx'],
                        include_dirs=[np.get_include()]),
              Extension('ConfigSpace.forbidden',
                        sources=['ConfigSpace/forbidden.pyx'],
                        include_dirs=[np.get_include()]),
              Extension('ConfigSpace.conditions',
                        sources=['ConfigSpace/conditions.pyx'],
                        include_dirs=[np.get_include()]),
              Extension('ConfigSpace.c_util',
                        sources=['ConfigSpace/c_util.pyx'],
                        include_dirs=[np.get_include()]),
              Extension('ConfigSpace.util',
                        sources=['ConfigSpace/util.pyx'],
                        include_dirs=[np.get_include()]),
              Extension('ConfigSpace.configuration_space',
                        sources=['ConfigSpace/configuration_space.pyx'],
                        include_dirs=[np.get_include()])]

for e in EXTENSIONS:
    e.cython_directives = COMPILER_DIRECTIVES


setup(
    name=MODULE_NAME,
    version=get_version('ConfigSpace/__version__.py'),
    url=MODULE_URL,
    description=SHORT_DESCRIPTION,
    ext_modules=EXTENSIONS,
    long_description=read_file("README.md"),
    license=LICENSE,
    platforms=PLATS,
    author=AUTHORS,
    author_email=AUTHOR_EMAIL,
    test_suite=TEST_SUITE,
    setup_requires=SETUP_REQS,
    install_requires=INSTALL_REQS,
    keywords=KEYWORDS,
    packages=find_packages(),
    python_requires=MIN_PYTHON_VERSION,
    classifiers=CLASSIFIERS
)
