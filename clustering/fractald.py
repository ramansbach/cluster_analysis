from __future__ import absolute_import, division, print_function
import numpy as np

import pdb
from cfractald import corrDim, getCOMs
__all__ = ['corrcalc','getCOMsPy','getCOMs','getCOMnumpy']

def getCOMnumpy(poslist,masslist):
    #return com coordinates of a single molecule, written in python using numpy
    poslist3 = poslist.reshape([len(poslist)/3,3])
    com = np.dot(masslist,poslist3)/masslist.sum()
    return com
    
def getCOMsPy(poslist,masslist,ats):
    #return the center of mass of each molecule in a position list of peptides
    N = int(len(poslist)/3/ats) #total number of molecules
    comlocs = np.zeros(3*N)
    for i in range(N):
        Rcom = getCOMnumpy(poslist[3*ats*i:3*ats*i+3*ats],masslist)
        comlocs[3*i:3*i+3] = Rcom
    return comlocs

def getcomsPy( pos, coms, masslist, beads, mols): 
    pos = np.reshape(pos,[1,3 * beads * mols])[0]
    coms = np.reshape(coms,[1,3*mols])[0]
    for i in range(mols):
        
            X = 0;
            Y = 0;
            Z = 0;
            M = 0;
            for j in range(beads):
                    #if i == 1:
                    #    pdb.set_trace()
                    x = masslist[j] * pos[beads*i + 3*j];
                    y = masslist[j] * pos[beads*i + 3*j+1];
                    z = masslist[j] * pos[beads*i + 3*j+2];
                    X+=x;
                    Y+=y;
                    Z+=z;
                    M += masslist[j];
                
            X/=M;
            Y/=M;
            Z/=M;
            coms[3*i] = X;
            coms[3*i + 1] = Y;
            coms[3*i + 2] = Z;
    coms = np.reshape(coms,[mols,3]) 
    return coms
    
def corrcalc(coms,emax,estep,fname=None):
    """ Given the locations of the beads in a snapshot, find and write out
    the C(e) function
    
    Parameters
    ----------
    coms: numpy array [M x 3]
        M = molno
        array contains COM positions for all molecules in a snapshot
    emax: double
        distance out to which to compute the correlation integral
    estep: double
        size of steps for correlation integral calculation
    fname: file name to write to
        if None, do not write file
        
    Returns
    -------
    corrInt: numpy array [N x 2]
        matrix representing the correlation integral versus radius (eps)
    
    """
    
    (epsnap,cdsnap) = corrDim(coms,emax,estep) 
    if fname is not None:
        f = open(fname,'w')
        for i in range(len(cdsnap)):
            f.write('{0}\t{1}\n'.format(epsnap[i],cdsnap[i]))
        f.close()
    return np.array([epsnap,cdsnap])