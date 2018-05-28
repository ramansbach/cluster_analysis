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
### Required packages

numpy (tested with v. 1.10.2)
pandas
gsd.hoomd (Installation instructions can be found here: https://bitbucket.org/glotzer/gsd )
scikit-learn v. 0.19.1 or greater !!!it will NOT work with lower versions
scipy (tested with v. 0.19.1)
mpi4py (although currently MPI is not fully supported)
os
matplotlib

## Optional packages

pdb (Useful for debugging)

### Installation and Checking


To install:
(from this folder)
> python setup.py build
> python setup.py install

To test:
(from this folder)
> pytest
test_MPI will fail because it is a dummy test to remind you to test MPI functionality when implemented

It is possible for test_nlFit_noise() to also fail--it should only do so occasionally; this is due to the fact it is a random sanity check that makes sure that one of the functions works in the presence of noise most of the time. If it fails repeatedly, there is a problem.

This takes a few minutes to run.
### Further notes


In "clustering/scripts" there is a number of automated scripts currently set up for use on the alf-clusters-rv.mrl.illinois.edu cluster; however, updating for any SGE submission system should be relatively straightforward.

There is also an additional README file specific to those scripts.
