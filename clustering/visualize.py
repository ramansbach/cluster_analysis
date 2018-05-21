from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import gsd.hoomd
import sklearn
import scipy.optimize as opt
import os
import pdb
from sklearn.neighbors import BallTree
from sklearn.neighbors import radius_neighbors_graph
from scipy.spatial.distance import cdist
from scipy.special import erf
from scipy.sparse.csgraph import connected_components
#from .due import due, Doi
from .smoluchowski import massAvSize
#from mpi4py import MPI
from cdistances import conOptDistanceCython,alignDistancesCython

__all__ = ['writeFrameForVMD','writeFramesFromCIDFile']

def writeFrameForVMD(clusterIDs,molno,atomno,frameout):
    """ Function that writes out a single frame for coloring by cluster
    
    Parameters
    ----------
    clusterIDs: list of ints for the frame corresponding to each molecule's
        cluster index
    molno: int
        number of molecules in frame
    atomno: int
        number of atoms per molecule
    frameout: string
        name of output file
    
    Notes
    -----
    Format of output file has a line for each cluster consisting of a set 
    of ints. The first int is the colorID, and the rest are the atom indices
    that should be set to that color.  By assumption, there are 16 available
    different colors.
    
    First line of file contains the total number of subsequent lines 
    (# of clusters)
    
    """
    framefile = open(frameout,'w')
    ind = 0
    framefile.write('{0}\n'.format(max(clusterIDs)+1))
    for cID in range(max(clusterIDs)+1):
        #pdb.set_trace()
        line = ''
        colorID = ind % 16
        line += str(colorID) + ' '
        molinds = np.where(cID == np.array(clusterIDs))[0]
        ainds = molinds.copy()
        for molind in molinds:
            ainds = np.concatenate((ainds,
                                      molno+molind*(atomno-1)+np.arange(0,
                                                                   atomno-1)))
        for aind in ainds:
            line += str(aind) + ' '
        line += '\n'
        framefile.write(line)
        ind += 1
    framefile.close()
    
def writeFramesFromCIDFile(cIDfile,molno,atomno,frameoutbase):
    """ Function that writes out multiple frames for coloring by cluster
    
    Parameters
    ----------
    cIDfile: file containing cluster IDs
    molno: int
        number of molecules in frame
    atomno: int
        number of atoms per molecule
    frameoutbase: string
        base name of output files
    
    """
    cIDfile = open(cIDfile)
    lines = cIDfile.readlines()
    cIDfile.close()
    ind = 0
    for line in lines:
        cIDsf = [float(c) for c in line.split()]
        cIDs = [int(c) for c in cIDsf]
        writeFrameForVMD(cIDs,molno,atomno,frameoutbase+str(ind)+'.dat')
        ind+=1
