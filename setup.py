"""Setup.py for ConfigSpace"""

import os

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize  # must go after setuptools


# Helper functions
def read_file(fname):
    """Get contents of file from the modules directory"""
    return open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8").read()


def get_version(fname):
    """Get the module version"""
    with open(fname, encoding="utf-8") as file_handle:
        return file_handle.readlines()[-1].split()[-1].strip("\"'")


def get_authors(fname):
    """Get the authors"""
    with open(fname, "r") as f:
        content = f.read()

    return [
        line.replace(",", "").replace('"', "").replace("    ", "")  # Remove noise
        for line in content.split("\n")
        if line.startswith(" ")  # Lines with space
    ]


class BuildExt(build_ext):
    """build_ext command for use when numpy headers are needed.
    SEE tutorial: https://stackoverflow.com/questions/2379898
    SEE fix: https://stackoverflow.com/questions/19919905
    """

    def finalize_options(self):
        build_ext.finalize_options(self)
        import numpy

        self.include_dirs.append(numpy.get_include())


# Configure setup parameters
MODULE_NAME = "ConfigSpace"
MODULE_URL = "https://github.com/automl/ConfigSpace"
SHORT_DESCRIPTION = (
    "Creation and manipulation of parameter configuration spaces for "
    "automated algorithm configuration and hyperparameter tuning."
)
KEYWORDS = (
    "algorithm configuration hyperparameter optimization empirical "
    "evaluation black box"
)
LICENSE = "BSD 3-clause"
PLATS = ["Linux", "Windows", "Mac"]

AUTHOR_EMAIL = "feurerm@informatik.uni-freiburg.de"
TEST_SUITE = "pytest"
INSTALL_REQS = ["numpy", "pyparsing", "scipy", "typing_extensions", "more_itertools"]
MIN_PYTHON_VERSION = ">=3.7"
CLASSIFIERS = [
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Development Status :: 4 - Beta",
    "Natural Language :: English",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: MacOS",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering",
    "Topic :: Software Development",
]

# These do not really change the speed of the benchmarks
COMPILER_DIRECTIVES = {
    "boundscheck": False,
    "wraparound": False,
    "language_level": "3",
}


"""
# Profiling
Set the below flag to True to enable profiling of the code. This will cause some minor performance
overhead so it should only be used for debugging purposes.

Use [`py-spy`](https://github.com/benfred/py-spy) with [speedscope.app](https://www.speedscope.app/)
```bash
pip install py-spy
py-spy record --rate 800 --format speedscope --subprocesses --native -o profile.svg -- python <script>
# Open in speedscope.app
```

If timing something really really low in time, use a higher `--rate`
You'll want to create a basic script that does the bar minimum as this allows you to bump up
the --rate option and get much higher fidelity information.

# Refs
* https://mclare.blog/posts/further-adventures-in-cython-profiling
* https://cython.readthedocs.io/en/latest/src/tutorial/profiling_tutorial.html#enabling-profiling-for-a-complete-source-file  # noqa: E501
* py-spy
"""
PROFILING = False
if PROFILING:
    COMPILER_DIRECTIVES["profile"] = True
    COMPILER_DIRECTIVES["linetrace"] = True

EXTENSIONS = [
    Extension(
        "ConfigSpace.hyperparameters_.hyperparameter",
        sources=["ConfigSpace/hyperparameters_/hyperparameter.pyx"]
    ),
    Extension(
       "ConfigSpace.hyperparameters_.constant",
        sources=["ConfigSpace/hyperparameters_/constant.pyx"]
    ),
    Extension(
        "ConfigSpace.hyperparameters_.numerical",
        sources=["ConfigSpace/hyperparameters_/numerical.pyx"]
    ),
    Extension(
        "ConfigSpace.hyperparameters_.float_hyperparameter",
        sources=["ConfigSpace/hyperparameters_/float_hyperparameter.pyx"]
    ),
    Extension(
        "ConfigSpace.hyperparameters_.integer_hyperparameter",
        sources=["ConfigSpace/hyperparameters_/integer_hyperparameter.pyx"]
    ),
    Extension(
        "ConfigSpace.hyperparameters_.uniform_integer",
        sources=["ConfigSpace/hyperparameters_/uniform_integer.pyx"]
    ),
    Extension(
        "ConfigSpace.hyperparameters_.normal_integer",
        sources=["ConfigSpace/hyperparameters_/normal_integer.pyx"]
    ),
    Extension(
        "ConfigSpace.hyperparameters_.uniform_float",
        sources=["ConfigSpace/hyperparameters_/uniform_float.pyx"]
    ),
    Extension(
        "ConfigSpace.hyperparameters_.normal_float",
        sources=["ConfigSpace/hyperparameters_/normal_float.pyx"]
    ),
    Extension(
        "ConfigSpace.hyperparameters_.ordinal",
        sources=["ConfigSpace/hyperparameters_/ordinal.pyx"]
    ),
    Extension(
        "ConfigSpace.hyperparameters_.categorical",
        sources=["ConfigSpace/hyperparameters_/categorical.pyx"]
    ),
    Extension(
        "ConfigSpace.hyperparameters", sources=["ConfigSpace/hyperparameters.pyx"]
    ),
    Extension("ConfigSpace.forbidden", sources=["ConfigSpace/forbidden.pyx"]),
    Extension("ConfigSpace.conditions", sources=["ConfigSpace/conditions.pyx"]),
    Extension("ConfigSpace.c_util", sources=["ConfigSpace/c_util.pyx"]),
    Extension("ConfigSpace.util", sources=["ConfigSpace/util.pyx"]),
    Extension(
        "ConfigSpace.configuration_space",
        sources=["ConfigSpace/configuration_space.pyx"],
    ),
]

extras_reqs = {
    "dev": [
        "pytest>=4.6",
        "mypy",
        "pre-commit",
        "pytest-cov",
        "automl_sphinx_theme>=0.1.11",
    ],
}


setup(
    name=MODULE_NAME,
    version=get_version("ConfigSpace/__version__.py"),
    cmdclass={"build_ext": BuildExt},
    url=MODULE_URL,
    description=SHORT_DESCRIPTION,
    long_description_content_type="text/markdown",
    ext_modules=cythonize(EXTENSIONS, compiler_directives=COMPILER_DIRECTIVES),
    long_description=read_file("README.md"),
    license=LICENSE,
    platforms=PLATS,
    author=get_authors("ConfigSpace/__authors__.py"),
    author_email=AUTHOR_EMAIL,
    test_suite=TEST_SUITE,
    install_requires=INSTALL_REQS,
    extras_require=extras_reqs,
    keywords=KEYWORDS,
    packages=find_packages(),
    python_requires=MIN_PYTHON_VERSION,
    classifiers=CLASSIFIERS,
)
