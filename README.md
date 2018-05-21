## clustering
[![Build Status](https://travis-ci.org/uwescience/clustering.svg?branch=master)](https://travis-ci.org/uwescience/clustering)

This is a small project to perform data analysis for clustering applications.  Right now it is mostly specific to HOOMD file formats and it is intended to find physical clusters in the data, even for very large systems, as well as extract a number of features:

(1) Mass-averaged cluster size versus time
(2) Distribution of cluster sizes at given time steps
(3) Perform Smoluchowski fitting of variables [cf. Wattis, Jonathan AD. "An introduction to mathematical models of coagulation–fragmentation processes: a discrete deterministic mean-field approach." Physica D: Nonlinear Phenomena 222.1 (2006): 1-20.]
(4) Analyze gyration tensor: radius of gyration, ratios of gyration tensor moments, relative shape anisotropy
(5) Analyze internal structure: twist angles and orderedness in terms of cluster hierarchy
(6) Analyze system fractal dimension for a given timestep

It has recently been extended to work with Gromacs XTC and GRO files.

It can do optical and contact cluster analysis but not yet aligned cluster analysis for either HOOMD or Gromacs file types.

Note that MPI support is currently not fully implemented.

### Organization of the  project

The project has the following structure:

    clustering/
      |- README.md
      |- clustering/
         |- __init__.py
         |- clustering.py
         |- due.py
         |- data/
            |- ...
         |- tests/
            |- ...
      |- doc/
         |- Makefile
         |- conf.py
         |- sphinxext/
            |- ...
         |- _static/
            |- ...
      |- setup.py
      |- .travis.yml
      |- appveyor.yml
      |- LICENSE
      |- Makefile
      |- ipynb/
         |- ...

In "scripts" there is a number of automated scripts currently set up for use on the alf-clusters-rv.mrl.illinois.edu cluster; however, updating for any SGE submission system should be relatively straightforward.

