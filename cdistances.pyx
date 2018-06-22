# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:23:13 2017

Cython modules for distance
"""
import cython
import numpy as np

from scipy.sparse import csr_matrix,lil_matrix,coo_matrix
cimport numpy as np

cdef extern double conoptdistance ( double * x, double * y, int ats)
cdef extern double aligndistance ( double * dists, double * distsA, 
                                  double * distsB, double * x, double * y,
                                  int ats)
cdef extern void subsquashrng ( double * rng, double * molrng, int dim,
                               int apermol)
#@cython.boundscheck(False)
#@cython.wraparound(False)
def checkSymmetry(csr):
    """
    Checks whether a matrix in CSR sparse format is symmetric.
    
    Parameters
    ----------
    csr: matrix in CSR format
    
    Returns
    -------
    symyes: bool
        True if symmetric, False if not
    """

    symyes = not (csr!=csr.transpose()).max()
    return symyes
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
    
def squashRNGCOOCython(rng,apermol):
    """
    Reduces radius neighbors graph to a new graph based on molecules instead of
    atoms.
    Uses COO format
    
    Parameters
    ----------
    rng: a graph in CSR format as produced by a BallTree
    apermol: int
        the number of atoms in a molecule
        
    Returns
    -------
    molrng: a new graph in CSR format
    
    Raises
    ------
    RuntimeError: if the original rng is not symmetric
    
    """
    if not checkSymmetry(rng):
        raise RuntimeError("Graph is non-symmetrical")
    sh = rng.shape
    cdef int dim = int(sh[0]/apermol)
    newsh = (dim,dim)
    molrng = lil_matrix(newsh)
    rng = coo_matrix(rng)
    cdef np.ndarray[int,ndim=1,mode="c"] rows = rng.row//apermol
    cdef np.ndarray[int,ndim=1,mode="c"] cols = rng.col//apermol
    cdef np.ndarray[int,ndim=1,mode="c"] rowcols = rows * dim + cols
    cdef np.ndarray[int,ndim=1,mode="c"] urowcols = np.unique(rowcols)
    rows = urowcols // dim
    cols = urowcols % dim

    for i in range(len(rows)):

        row = rows[i]
        col = cols[i]
        if col > row:
            molrng[row,col] = 1
            molrng[col,row] = 1
    return csr_matrix(molrng)
