# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:23:13 2017

Cython modules for distance
"""
import cython
import numpy as np
cimport numpy as np

cdef extern double conoptdistance ( double * x, double * y, int ats)
cdef extern double aligndistance ( double * dists, double * distsA, 
                                  double * distsB, double * x, double * y,
                                  int ats)
cdef extern void subsquashrng ( double * rng, double * molrng, int dim,
                               int apermol)
#@cython.boundscheck(False)
#@cython.wraparound(False)
def conOptDistanceCython(np.ndarray[double,ndim=1,mode="c"] x not None,np.ndarray[double,ndim=1,mode="c"] y not None):
    
    if len(x) % 3 != 0 or len(y) % 3 != 0:
        raise RuntimeError("3D array has a number of entries not divisible \
                            by 3.")
    
    cdef int ats = int(len(x)/3)
    mind = conoptdistance(&x[0],&y[0],ats)
    return mind

def alignDistancesCython(np.ndarray[double,ndim=1,mode="c"] x not None,np.ndarray[double,ndim=1,mode="c"] y not None):
    if len(x) % 3 != 0 or len(y) % 3 != 0:
        raise RuntimeError("3D array has a number of entries not divisible \
                            by 3.")
    cdef int ats = int(len(x)/3)
    cdef np.ndarray[double,ndim=1,mode="c"] dists = np.zeros([ats * ats])
    cdef np.ndarray[double,ndim=1,mode="c"] distsA = np.zeros([ats * ats])
    cdef np.ndarray[double,ndim=1,mode="c"] distsB = np.zeros([ats * ats])
    mind3 = aligndistance(&dists[0],&distsA[0],&distsB[0],&x[0],&y[0],ats)
    return mind3
    
def subsquashRNG( rng,
                  molrng,
                 int apermol):
    cdef int dim = int(np.shape(molrng)[0])
    cdef np.ndarray[double,ndim=2,mode="c"] rngc = np.array(rng)
    cdef np.ndarray[double,ndim=2,mode="c"] molrngc = np.array(molrng)
    subsquashrng(&rngc[0,0],&molrngc[0,0],dim,apermol)
    return molrngc