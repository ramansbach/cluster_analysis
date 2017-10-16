# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:19:02 2017
Cython modules for correlation dimension
@author: Rachael Mansbach
"""
from __future__ import absolute_import, division, print_function
import cython
import numpy as np
cimport numpy as np
from scipy.spatial.distance import pdist

cdef extern void corrdim ( double * epss, double * ce, double * distsq, 
                          int Nepss, int Nd2)
                          
cdef extern void getcoms( double * pos, double * coms, double * masslist, int beads, int mols)
@cython.boundscheck(False)
@cython.wraparound(False)

def getCOMs(np.ndarray[double,ndim=2,mode="c"] pos not None,
            np.ndarray[double,ndim=1,mode="c"] masslist not None):
    """ find the centers of mass of a set of molecules
    
    Parameters
    ----------
    pos: numpy array [M x 3*ats]
        M = molno, ats = number of beads per molecule
        array contains bead positions for all beads in a snapshot
    
    Returns
    -------
    coms: numpy array [M x 3]
        center of mass of each molecule in a snapshot
    """
    sz = np.shape(pos)
    cdef int beads = int(sz[1] / 3)
    cdef int mols = sz[0]
    cdef np.ndarray[double,ndim=2,mode="c"] coms = np.zeros([mols,3])
    getcoms(&pos[0,0], &coms[0,0], &masslist[0], beads, mols)
    return coms
    
def corrDim(comsnap,emax,estep):
    """ calculate C(eps) from 0 to emax with steps of estep, where C(eps) is 
    the correlation sum on the set of center-of-mass distances
    
    Parameters
    ----------
    comsnap: np array [N x 3]
        matrix of center of mass distances
    emax: float
        maximum radius out to which to compute the correlation integral
    estep: float
        step size for correlation integral calculation
        
    Returns
    -------
    epss: numpy vector [length E]
        list of epsilons at which the correlation integral is computed
    ce: numpy vector [length E]
        correlation integral
    """
    
    
    cdef np.ndarray[double,ndim=1,mode="c"] distsq = pdist(comsnap,
                                                           'sqeuclidean')

    cdef np.ndarray[double,ndim=1,mode="c"] epss = np.arange(estep,emax+estep,
                                                             estep) 
    
    cdef int Nepss = len(epss)
    cdef int Nd2 = len(distsq)
    cdef np.ndarray[double,ndim=1,mode="c"] ce = np.zeros(Nepss)
    corrdim(&epss[0],&ce[0],&distsq[0],Nepss,Nd2)
    
    N = len(comsnap)
    ce = ce/(N*(N-1))

    return (epss,ce)
