from setuptools import setup, find_packages
from setuptools.extension import Extension
import os
import numpy as np

# Read http://peterdowns.com/posts/first-time-with-pypi.html to figure out how
# to publish the package on PyPI

here = os.path.abspath(os.path.dirname(__file__))
desc = 'Creation and manipulation of parameter configuration spaces for ' \
       'automated algorithm configuration and hyperparameter tuning.'
keywords = 'algorithm configuration hyperparameter optimization empirical ' \
           'evaluation black box'

# These do not really change the speed of the benchmarks
compiler_directives = {
    'boundscheck': False,
    'wraparound': False,
}

extensions = [
    Extension('ConfigSpace.hyperparameters',
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
              include_dirs=[np.get_include()])
]

for e in extensions:
    e.cython_directives = compiler_directives


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open("ConfigSpace/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")


setup(
    name='ConfigSpace',
    version=version,
    url='https://github.com/automl/ConfigSpace',
    description=desc,
    ext_modules=extensions,
    long_description=read("README.md"),
    license='BSD 3-clause',
    platforms=['Linux'],
    author=', '.join(["Matthias Feurer", "Katharina Eggensperger",
                      "Syed Mohsin Ali", "Christina Hernandez Wunsch",
                      "Julien-Charles Levesque", "Jost Tobias Springenberg", "Philipp Mueller"
                      "Marius Lindauer", "Jorn Tuyls"]),
    author_email='feurerm@informatik.uni-freiburg.de',
    test_suite="pytest",
    # https://stackoverflow.com/questions/24923003/organizing-a-package-with-cython
    setup_requires=[
        'Cython',
    ],
    install_requires=[
        'numpy',
        'pyparsing',
        'typing',
        'Cython',
    ],
    keywords=keywords,
    packages=find_packages(),
    python_requires='>=3.4.*',
    classifiers=[
        'Programming Language :: Python :: 3.4',
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
        'Topic :: Software Development',
    ]
)
