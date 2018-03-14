#!/usr/bin/env python

from setuptools import setup

setup(name='bioimaging',
      version='0.1',
      description='Monte Carlo simulation toolkit for bioimaging systems',
      author='Masaki Watabe',
      packages=['bioimaging'],
      package_dir={'bioimaging': 'bioimaging'},
      package_data={'bioimaging': ['catalog/*.txt', 'catalog/*/*.csv']},
      install_requires=['numpy', 'scipy'],
      test_suite='test',
      license='BSD-3-Clause',
      )
