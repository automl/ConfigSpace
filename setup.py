"""Setup.py for ConfigSpace.

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
from __future__ import annotations

from Cython.Build import cythonize  # must go after setuptools
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

PROFILING = False


class BuildExt(build_ext):
    """build_ext command for use when numpy headers are needed.
    SEE tutorial: https://stackoverflow.com/questions/2379898
    SEE fix: https://stackoverflow.com/questions/19919905.
    """

    def finalize_options(self):
        build_ext.finalize_options(self)
        import numpy

        self.include_dirs.append(numpy.get_include())


# These do not really change the speed of the benchmarks
COMPILER_DIRECTIVES = {
    "boundscheck": False,
    "wraparound": False,
    "language_level": "3",
}

if PROFILING:
    COMPILER_DIRECTIVES.update({"profile": True, "linetrace": True})

"""
EXTENSIONS = [
    Extension(
        "ConfigSpace.hyperparameters.beta_float",
        sources=["ConfigSpace/hyperparameters/beta_float.pyx"],
    ),
    Extension(
        "ConfigSpace.hyperparameters.beta_integer",
        sources=["ConfigSpace/hyperparameters/beta_integer.pyx"],
    ),
    Extension(
        "ConfigSpace.hyperparameters.categorical",
        sources=["ConfigSpace/hyperparameters/categorical.pyx"],
    ),
    Extension(
        "ConfigSpace.hyperparameters.constant",
        sources=["ConfigSpace/hyperparameters/constant.pyx"],
    ),
    Extension(
        "ConfigSpace.hyperparameters.float_hyperparameter",
        sources=["ConfigSpace/hyperparameters/float_hyperparameter.pyx"],
    ),
    Extension(
        "ConfigSpace.hyperparameters.hyperparameter",
        sources=["ConfigSpace/hyperparameters/hyperparameter.pyx"],
    ),
    Extension(
        "ConfigSpace.hyperparameters.integer_hyperparameter",
        sources=["ConfigSpace/hyperparameters/integer_hyperparameter.pyx"],
    ),
    Extension(
        "ConfigSpace.hyperparameters.normal_float",
        sources=["ConfigSpace/hyperparameters/normal_float.pyx"],
    ),
    Extension(
        "ConfigSpace.hyperparameters.normal_integer",
        sources=["ConfigSpace/hyperparameters/normal_integer.pyx"],
    ),
    Extension(
        "ConfigSpace.hyperparameters.numerical",
        sources=["ConfigSpace/hyperparameters/numerical.pyx"],
    ),
    Extension(
        "ConfigSpace.hyperparameters.ordinal",
        sources=["ConfigSpace/hyperparameters/ordinal.pyx"],
    ),
    Extension(
        "ConfigSpace.hyperparameters.uniform_float",
        sources=["ConfigSpace/hyperparameters/uniform_float.pyx"],
    ),
    Extension(
        "ConfigSpace.hyperparameters.uniform_integer",
        sources=["ConfigSpace/hyperparameters/uniform_integer.pyx"],
    ),
    Extension("ConfigSpace.forbidden", sources=["ConfigSpace/forbidden.pyx"]),
    Extension("ConfigSpace.conditions", sources=["ConfigSpace/conditions.pyx"]),
    Extension("ConfigSpace.c_util", sources=["ConfigSpace/c_util.pyx"]),
]
"""


setup(
    name="ConfigSpace",
    #cmdclass={"build_ext": BuildExt},
    #ext_modules=cythonize(EXTENSIONS, compiler_directives=COMPILER_DIRECTIVES),
    packages=find_packages(),
)
