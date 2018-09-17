from setuptools import setup, find_packages
from setuptools.extension import Extension
import os
from Cython.Build import cythonize
from Cython.Distutils import build_ext
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

extensions = cythonize(
    [Extension('ConfigSpace.hyperparameters',
               sources=['ConfigSpace/hyperparameters.pyx',],
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
               sources=['ConfigSpace/util.py'],
               include_dirs=[np.get_include()]),
     Extension('ConfigSpace.configuration_space',
               sources=['ConfigSpace/configuration_space.py'],
               include_dirs=[np.get_include()]),
     ],
    compiler_directives=compiler_directives,
)

for e in extensions:
    e.cython_directives = {"embedsignature": True}

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open("ConfigSpace/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")


setup(
    name='ConfigSpace',
    version=version,
    url='https://github.com/automl/ConfigSpace',
    description=desc,
    cmdclass={'build_ext': build_ext},
    ext_modules=extensions,
    long_description=read("README.md"),
    license='BSD 3-clause',
    platforms=['Linux'],
    author=', '.join(["Matthias Feurer", "Katharina Eggensperger",
                      "Syed Mohsin Ali", "Christina Hernandez Wunsch",
                      "Julien-Charles Levesque", "Jost Tobias Springenberg",
                      "Marius Lindauer", "Jorn Tuyls"]),
    author_email='feurerm@informatik.uni-freiburg.de',
    test_suite="nose.collector",
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
