#!/usr/bin/env python

from setuptools import setup
import os.path


install_requires = [pkg_name.rstrip('\r\n') for pkg_name in open('requirements.txt').readlines()]

# version
here = os.path.dirname(os.path.abspath(__file__))
version = next(
    (line.split('=')[1].strip().replace("'", '')
     for line in open(os.path.join(here, 'scopyon', '__init__.py'))
     if line.startswith('__version__ = ')),
    '0.0.dev0')  ## default

setup(
    name='scopyon',
    version=version,
    url='https://github.com/ecell/scopyon',
    description='Monte Carlo simulation toolkit for bioimaging systems',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Masaki Watabe',
    author_email='masaki@riken.jp',
    maintainer='Kazunari Kaizu',
    maintainer_email='kaizu@riken.jp',
    packages=['scopyon'],
    package_dir={'scopyon': 'scopyon'},
    package_data={'scopyon': ['default_parameters.ini', 'catalog/*.txt', 'catalog/*/*.csv']},
    install_requires=install_requires,
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
