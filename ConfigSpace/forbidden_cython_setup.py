from distutils.core import setup
from Cython.Build import cythonize

setup(
  name='Forbidden Python',
  ext_modules=cythonize("forbidden_cython.pyx"),
)