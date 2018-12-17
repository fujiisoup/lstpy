#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from setuptools import setup
import re

# load version form _version.py
VERSIONFILE = "lstpy/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# module

setup(name='lstpy',
      version=verstr,
      author="Keisuke Fujii",
      author_email="fujiisoup@gmail.com",
      description=("Python library to read list file for MPA3 system"),
      license="BSD 3-clause",
      keywords="data acquisition",
      url="http://github.com/fujiisoup/lstpy",
      include_package_data=True,
      ext_modules=[],
      packages=["lstpy", ],
      package_dir={'lstpy': 'lstpy'},
      py_modules=['lstpy.__init__'],
      test_suite='test_public',
      install_requires="""
        numpy>=1.11
        numba>=0.39
        """,
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3.6',
                   'Topic :: Scientific/Engineering :: Physics']
      )
