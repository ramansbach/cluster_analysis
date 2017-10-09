# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:17:36 2017

@author: rachael

Debugging C versus Python versions of code
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy import weave
def conOptDistance(x,y):
    """
    Function that computes the distance between molecules for contact
    or optical clusters
    
    Parameters:
    -----------
    x : array
        The 1D array of size 3*ats representing the first molecule
    y : array
        The 1D array of size 3*ats representing the second molecule
        
    Returns
    -------
    r : float
        The distance between x and y computed as the minimum distance
        between any two beads in the molecules
    """
    if len(x) % 3 != 0 or len(y) % 3 != 0:
        raise RuntimeError("3D array has a number of entries not divisible \
                            by 3.")
    ats = len(x)/3
    xa = np.reshape(x,[ats,3])
    ya = np.reshape(y,[ats,3])
    #return np.min(euclidean_distances(xa,ya,squared=True))
    dists = cdist(xa,ya,metric='sqeuclidean')
    r = np.min(dists)    
    return r

def conOptDistanceC(x,y):
    """
        Function that computes the distance between molecules for contact
    or optical clusters
    
    Parameters:
    -----------
    x : array
        The 1D array of size 3*ats representing the first molecule
    y : array
        The 1D array of size 3*ats representing the second molecule
        
    Returns
    -------
    r : float
        The distance between x and y computed as the minimum distance
        between any two beads in the molecules
        
    Notes
    -----
    Uses scipy.weave to incorporate a little bit of C code to see if that
    will speed things up
    """
    if len(x) % 3 != 0 or len(y) % 3 != 0:
        raise RuntimeError("3D array has a number of entries not divisible \
                            by 3.")
   
    ats = int(len(x)/3)
    #xa = np.reshape(x,[ats,3])
    #ya = np.reshape(y,[ats,3])
    mind = 10000.0
    support = '#include <math.h>'
    
    code = """
    int i,j;
    return_val = 0;
    double d;
    for (i = 0; i < ats; i++) {
       for (j = 0; j < ats; j++) {
           d = (double(x[3*i]) - double(y[3*j])) 
               * (double(x[3*i]) - double(y[3*j]))
               + (double(x[3*i + 1]) - double(y[3*j + 1])) 
               * (double(x[3*i + 1]) - double(y[3*j + 1]))
               + (double(x[3*i + 2]) - double(y[3*j + 2])) 
               * (double(x[3*i + 2]) - double(y[3*j + 2]));
           if (d < mind){
               mind = d;            
           }
       }     
    }
    return_val = mind;
	
			"""
    mind = weave.inline(code,['x', 'y', 'ats','mind'],
                        support_code = support, libraries = ['m'])
    #return np.min(euclidean_distances(xa,ya,squared=True))    
    return mind
    
def conOptDistanceC_inPy(x,y):
    if len(x) % 3 != 0 or len(y) % 3 != 0:
        raise RuntimeError("3D array has a number of entries not divisible \
                            by 3.")
    mind = 10000.0
    ats = int(len(x)/3)
    ds = np.zeros([ats,ats])
    
    for i in range(ats):
        for j in range(ats):
            d = (x[3*i] - y[3*j]) * (x[3*i] - y[3*j]) \
               + (x[3*i + 1] - y[3*j + 1]) * (x[3*i + 1] - y[3*j + 1]) \
               + (x[3*i + 2] - y[3*j + 2]) * (x[3*i + 2] - y[3*j + 2])
            ds[i,j] = d
            if (d < mind):
                mind = d
    return mind
    
def alignedDistance(x,y):
    """
    Function that computes the distances between molecules for aligned clusters
    
    Parameters:
    -----------
    x : array
        The 1D array of size 3*ats representing the first molecule
    y : array
        The 1D array of size 3*ats representing the second molecule
        
    Returns
    -------
    r : float
        The distance between x and y computed as the minimum distance
        between any two beads in the molecules
        
    Raises
    ------
    RuntimeError
        if the array does not have a number of entries divisible by three
        because it's supposed to be a flattened array of positions
        
    Notes
    -----
    Compute the minimum distance of each COM to another COM
    Take the three minimum distances of this list
    Return the maximum of these three
    
    """
    if len(x) % 3 != 0 or len(y) % 3 != 0:
        raise RuntimeError("3D array has a number of entries not divisible \
                            by 3.")
    ats = int(len(x)/3)
    xa = np.reshape(x,[ats,3])
    ya = np.reshape(y,[ats,3])
    distmat = cdist(xa,ya,metric='sqeuclidean')
    
    dists = np.zeros([ats * ats, 3])
    dind = 0
    for i in range(ats):
        for j in range(ats):
            dists[dind,0] = distmat[i,j]
            dists[dind,1] = i
            dists[dind,2] = j
            dind += 1
    sdists = dists[dists[:,0].argsort()]
    
    i1 = sdists[0,1]
    j1 = sdists[0,2]
    i2 = i1
    j2 = j1
    ind2 = 1
    while (i2 == i1) or (j2 == j1):
        
        i2 = sdists[ind2,1]
        j2 = sdists[ind2,2]
        ind2 += 1
    ind3 = ind2
    
    i3 = sdists[ind3,1]
    j3 = sdists[ind3,2]
    while (i3 == i1) or (i3 == i2) or (j3 == j1) or (j3 == j2):
        
        i3 = sdists[ind3,1]
        j3 = sdists[ind3,2]
        ind3 += 1
    return sdists[ind3-1,0]

def alignedDistanceC(x,y):
    """
    Function that computes the distances between molecules for aligned clusters
    
    Parameters:
    -----------
    x : array
        The 1D array of size 3*ats representing the first molecule
    y : array
        The 1D array of size 3*ats representing the second molecule
        
    Returns
    -------
    r : float
        The distance between x and y computed as the minimum distance
        between any two beads in the molecules
        
    Raises
    ------
    RuntimeError
        if the array does not have a number of entries divisible by three
        because it's supposed to be a flattened array of positions
        
    Notes
    -----
    Compute the minimum distance of each COM to another COM
    Take the three minimum distances of this list
    Return the maximum of these three
    Use scipy.weave and C++ to speed things up
    
    """
    if len(x) % 3 != 0 or len(y) % 3 != 0:
        raise RuntimeError("3D array has a number of entries not divisible \
                            by 3.")
    ats = int(len(x)/3)
    dists = np.zeros([ats * ats])
    distsA = np.zeros([ats * ats])
    distsB = np.zeros([ats * ats])

    support = '#include <math.h>'
    code = """
    int i,j,dind = 0;
    return_val = 0;
    for (i = 0; i < ats; i++){
        for (j = 0; j < ats; j++){
            dists[dind] = (x[3 * i] - y[3 * j]) * (x[3 * i] - y[3 * j])
            + (x[3 * i + 1] - y[3 * j + 1]) * (x[3 * i + 1] - y[3 * j + 1])
            + (x[3 * i + 2] - y[3 * j + 2]) * (x[3 * i + 2] - y[3 * j + 2]);
            distsA[dind] = i;
            distsB[dind] = j;
            dind++;
        }    
    }
    double mind = 10000.0;
    int mindi, mindj;
    for (int k = 0; k < ats * ats; k++){
        if (dists[k] < mind){
            mind = dists[k];
            mindi = distsA[k];
            mindj = distsB[k];
            
        }
    }
    double mind2 = 10000.0;
    int mind2i, mind2j;
    for (int k = 0; k < ats * ats; k++){
        if ((dists[k] < mind2) && (distsA[k] != mindi) && (distsB[k] != mindj))
        {
            mind2 = dists[k];
            mind2i = distsA[k];
            mind2j = distsB[k];
        }
    }
    double mind3 = 10000.0;
    for (int k = 0; k < ats * ats; k++){
        if ((dists[k] < mind3) && (distsA[k] != mindi) && (distsB[k] != mindj) 
        && (distsA[k] != mind2i) && (distsB[k] != mind2j)){
            mind3 = dists[k];
        }
    }
    return_val = mind3;
    """
    mind3 = weave.inline(code,['x', 'y','dists','distsA','distsB','ats'],
                        support_code = support, libraries = ['m'])
    return mind3
    
def alignedDistanceC_inPy(x,y):
    """
    Function that computes the distances between molecules for aligned clusters
    Test instantiation of C algorithm in python
    Parameters:
    -----------
    x : array
        The 1D array of size 3*ats representing the first molecule
    y : array
        The 1D array of size 3*ats representing the second molecule
        
    Returns
    -------
    r : float
        The distance between x and y computed as the minimum distance
        between any two beads in the molecules
        
    Raises
    ------
    RuntimeError
        if the array does not have a number of entries divisible by three
        because it's supposed to be a flattened array of positions
        
    Notes
    -----
    Compute the minimum distance of each COM to another COM
    Take the three minimum distances of this list
    Return the maximum of these three
    Use scipy.weave and C++ to speed things up
    
    """
    if len(x) % 3 != 0 or len(y) % 3 != 0:
        raise RuntimeError("3D array has a number of entries not divisible \
                            by 3.")
    ats = int(len(x)/3)
    dists = np.zeros([ats * ats])
    distsA = np.zeros([ats * ats])
    distsB = np.zeros([ats * ats])

    dind = 0
    for i in range(ats):
        for j in range(ats):
            
            dists[dind] = (x[3 * i] - y[3 * j]) * (x[3 * i] - y[3 * j]) \
            + (x[3 * i + 1] - y[3 * j + 1]) * (x[3 * i + 1] - y[3 * j + 1]) \
            + (x[3 * i + 2] - y[3 * j + 2]) * (x[3 * i + 2] - y[3 * j + 2])
            distsA[dind] = i
            distsB[dind] = j
            dind += 1
    mind = 10000.0
    for k in range(ats * ats):
        if dists[k] < mind:
            mind = dists[k]
            mindi = distsA[k]
            mindj = distsB[k]
    mind2 = 10000.0
    for k in range(ats * ats):
        if (dists[k] < mind2) and (distsA[k] != mindi) \
        and (distsB[k] != mindj):
            mind2 = dists[k]
            mind2i = distsA[k]
            mind2j = distsB[k]
    mind3 = 10000.0
    for k in range(ats * ats):
        if (dists[k] < mind3) and (distsA[k] != mindi) \
        and (distsB[k] != mindj) \
        and (distsA[k] != mind2i) and (distsB[k] != mind2j):
            mind3 = dists[k]

    return mind3
    
if __name__ == "__main__":
    mol1 = [ 3.07497501,  1.65221345, -4.89888906,  3.84725761,  1.76435292,
            4.99291611,\
  3.17241406,  1.36399543, -4.50210714,  3.94469643,  1.4761349,  -4.61030197,
  3.2745471,   0.96399653, -4.18767405,  4.0468297 ,  1.07613599, -4.29586887,
  3.07966948,  1.54043269, -4.98123837,  3.26858568,  2.22882891,  4.21227455,
  3.17710829,  1.25221467, -4.58445644,  3.94939089,  1.36435401, -4.69265079,
  3.85321903,  0.49952048, -3.40703297,  3.46346354,  1.65239286, -4.99416113,
  3.65834141,  1.0759567 , -4.20059681,  4.04213524,  1.18791676, -4.21352005,
  3.56090236,  1.36417472 , -4.59737921,  3.85195184,  1.65257215,  4.91056681,
  3.26985288,  1.07577729, -4.10532522]

    mol2 = [ 1.95292628,  0.12512627, -0.12933734,  1.7895844,  -0.36124897,
            -0.72721708,
  2.76484418, -0.41619316,  0.08920857,  2.19554329, -0.63190871, -0.61794412,
  2.60150242, -0.9025684 , -0.50867116,  2.67731237, -0.82326859, -0.59389299,
  2.84065413, -0.33689335,  0.00398675,  1.86539423, -0.28194916, -0.81243891,
  2.02873611,  0.20442609, -0.21455915,  2.35888529, -0.14553344, -0.02006438,
  2.43469524, -0.06623362, -0.1052862 ,  1.09724224,  0.462908,   -0.68943405,
  3.53299642, -1.16105032, -0.0337963 ,  1.90916026, -0.07841144, -0.47088811,
  2.7210784 , -0.61973089, -0.25234219,  2.31511927, -0.34907117, -0.36161515,
  2.27135324, -0.55260885, -0.70316595]

    rpy = conOptDistance(mol1,mol2)
    rc = conOptDistanceC(mol1,mol2)
    rcpy = conOptDistanceC_inPy(mol1,mol2)
    print("rpy: ",rpy)
    print("rc: ",rc)
    print("rcpy: ",rcpy)
    apos1 = np.array([-1.,0.,0.,-1.,-0.5,0.,
                      -1.,-1.,0.,0.,-1.,0.,0.,-0.5,0.,0.,0.,0.])
    apos2 = np.array([1.5,0.,0.,1.5,-0.5,0.,
                      1.5,-1.,0.,0.5,-1.,0.,0.5,-0.5,0.,0.5,0.,0.])
    arpy = alignedDistance(apos1,apos2)
    arc = alignedDistanceC(apos1,apos2)
    arcpy = alignedDistanceC_inPy(apos1,apos2)
    print("arpy: ",arpy)
    print("arc: ",arc)
    print("arcpy: ",arcpy)
