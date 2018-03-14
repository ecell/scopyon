#!/usr/bin/env python

from distutils.core import setup

setup(name='bioimaging',
      version='0.1',
      description='Monte Carlo simulation toolkit for bioimaging systems',
      author='Masaki Watabe',
      packages=['bioimaging'],
      package_dir={'bioimaging': 'bioimaging'},
      package_data={'bioimaging': ['catalog/*.txt', 'catalog/*/*.csv']},
      )
