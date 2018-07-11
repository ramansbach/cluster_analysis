from __future__ import absolute_import, division, print_function
from os.path import join as pjoin
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
#from Cython.Build import cythonize
# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "clustering: data analysis for simple MD cluster data"
# Long description will go up on the pypi page
long_description = """

Clustering
========
Clustering is a suite of code primarily intended for finding clusters and
performing data analysis on them.  These clusters are physical clusters in the
data. Also instantiated is fits to the mass-averaged cluster size versus time by
the Smoluchowski model.

License
=======
``clustering`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2017--, Rachael Mansbach, University of Illinois at
Urbana-Champaign
"""

NAME = "clustering"
MAINTAINER = "Rachael Mansbach"
MAINTAINER_EMAIL = "ramansbach@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/ramansbach/clustering"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Rachael Mansbach"
AUTHOR_EMAIL = "ramansbach@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'clustering': [pjoin('data', '*')]}
REQUIRES = ["numpy","scipy","Cython","scikit"]
BEXT = {'build_ext': build_ext}
CYTHONMODS=[Extension("cdistances",
                      sources=["cdistances.pyx","conoptdistance.c",
                               "aligndistance.c","subsquashrng.c",
                               "gyrtensxy.c"],
                      include_dirs=[numpy.get_include()]),
           Extension("cfractald",
                     sources=["cfractald.pyx","corrdim.c","getcoms.c"],
                      include_dirs=[numpy.get_include()]) ]
