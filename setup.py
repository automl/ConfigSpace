from setuptools import setup, find_packages
import os

import ParameterConfigurationSpace

here = os.path.abspath(os.path.dirname(__file__))
desc = 'Package to describe configuration spaces for automated algorithm ' \
       'configuration and hyperparameter tuning'
keywords = 'algorithm configuration hyperparameter optimization empirical ' \
           'evaluation black box'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='HPOlibConfigSpace',
    version=ParameterConfigurationSpace.__version__,
    url='https://github.com/automl/HPOlibConfigSpace',
    license='GPLv3',
    platforms=['Linux'],
    author=ParameterConfigurationSpace.__authors__,
    test_suite="nose.collector",
    install_requires=['argparse',
                      'numpy',
                      'pyparsing',
                      'six'
                      ],
    author_email='feurerm@informatik.uni-freiburg.de',
    description=desc,
    long_description=read("README.md"),
    keywords=keywords,
    packages=find_packages(),
    scripts=['scripts/HPOlib-convert'],
    classifiers=[
        'Programming Language :: Python :: 2.7 :: 3.4',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: General Public License v3 (LGPLv3)',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ]
)
