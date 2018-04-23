#!/usr/bin/env python

from setuptools import setup
import os.path


# version
here = os.path.dirname(os.path.abspath(__file__))
version = next(
    (line.split('=')[1].strip().replace("'", '')
     for line in open(os.path.join(here, 'bioimaging', '__init__.py'))
     if line.startswith('__version__ = ')),
    '0.0.dev0')  ## default

setup(
    name='bioimaging',
    version=version,
    description='Monte Carlo simulation toolkit for bioimaging systems',
    author='Masaki Watabe',
    author_email='masaki@riken.jp',
    maintainer='Kazunari Kaizu',
    maintainer_email='kaizu@riken.jp',
    packages=['bioimaging'],
    package_dir={'bioimaging': 'bioimaging'},
    package_data={'bioimaging': ['default_parameters.ini', 'catalog/*.txt', 'catalog/*/*.csv']},
    install_requires=['numpy', 'scipy'],
    test_suite='test',
    license='BSD-3-Clause',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        # 'Topic :: Scientific/Engineering :: Physics',
        # 'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: BSD License',
        ],
    )
