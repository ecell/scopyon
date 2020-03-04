#!/usr/bin/env python

from setuptools import setup
from setuptools.command.install import install
import os
import os.path
import sys
import glob


install_requires = [pkg_name.rstrip('\r\n') for pkg_name in open('requirements.txt').readlines()]

# version
here = os.path.dirname(os.path.abspath(__file__))
version = next(
    (line.split('=')[1].strip().replace("'", '')
     for line in open(os.path.join(here, 'scopyon', '__init__.py'))
     if line.startswith('__version__ = ')),
    '0.0.0dev')  ## default

def readme():
    with open('README.md') as f:
        return f.read()

class verify_version_command(install):
    description = "verify that the git tag matches the version"

    def run(self):
        tag = os.getenv('CIRCLE_TAG')
        if tag is not None and tag != version:
            info = "Git tag: {0} does not match the version of this library: {1}".format(tag, version)
            sys.exit(info)

setup(
    name='scopyon',
    version=version,
    url='https://github.com/ecell/scopyon',
    description='Monte Carlo simulation toolkit for bioimaging systems',
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='Masaki Watabe',
    author_email='masaki@riken.jp',
    maintainer='Kazunari Kaizu',
    maintainer_email='kaizu@riken.jp',
    packages=['scopyon'],
    package_dir={'scopyon': 'scopyon'},
    package_data={'scopyon': ['scopyon.yml', 'catalog/*.txt', 'catalog/*/*.csv']},
    data_files=[('examples', ['examples/tirf.py', 'examples/tirf_000.png', 'examples/twocolor.py', 'examples/twocolor_000.png'])],
    install_requires=install_requires,
    test_suite='test',
    license='BSD-3-Clause',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        # 'Topic :: Scientific/Engineering :: Physics',
        # 'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: BSD License',
        ],
    python_requires='>=3',
    cmdclass={'verify': verify_version_command}
    )
