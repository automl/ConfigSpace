from setuptools import setup, find_packages
import os
import ConfigSpace

# Read http://peterdowns.com/posts/first-time-with-pypi.html to figure out how
# to publish the package on PyPI

here = os.path.abspath(os.path.dirname(__file__))
desc = 'Creation and manipulation of parameter configuration spaces for ' \
       'automated algorithm configuration and hyperparameter tuning.'
keywords = 'algorithm configuration hyperparameter optimization empirical ' \
           'evaluation black box'


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open("ConfigSpace/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")


setup(
    name='ConfigSpace',
    version=ConfigSpace.__version__,
    url='https://github.com/automl/ConfigSpace',
    description=desc,
    long_description=read("README.rst"),
    license='BSD 3-clause',
    platforms=['Linux'],
    author=', '.join(ConfigSpace.__authors__),
    author_email='feurerm@informatik.uni-freiburg.de',
    test_suite="nose.collector",
    install_requires=['argparse',
                      'numpy',
                      'pyparsing',
                      'six'],
    keywords=keywords,
    packages=find_packages(),
    scripts=['scripts/HPOlib-convert'],
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
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
