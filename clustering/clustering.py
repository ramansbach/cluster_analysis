from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import gsd.hoomd
import sklearn
import scipy.optimize as opt
import os
import os.path
import pdb
from sklearn.neighbors import BallTree
from sklearn.neighbors import radius_neighbors_graph
from scipy.spatial.distance import cdist,pdist
from scipy.special import erf
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix,lil_matrix,coo_matrix
#from .due import due, Doi
from .smoluchowski import massAvSize
from mpi4py import MPI
from cdistances import conOptDistanceCython,alignDistancesCython,subsquashRNG
from cdistances import squashRNGCOOCython


__all__ = ["ClusterSnapshot", "ContactClusterSnapshot",
           "OpticalClusterSnapshot","AlignedClusterSnapshot",
           "ContactClusterSnapshotXTC","OpticalClusterSnapshotXTC",
           "SnapSystem",
           "conOptDistance","conOptDistanceC","alignedDistance",
           "alignedDistanceC","fixMisplacedArom","checkSymmetry",
           "squashRNG","squashRNGCython","squashRNGPy","squashRNGCOO",
           "squashRNGCOOCython"]


# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
'''
due.cite(Doi("10.1167/13.9.30"),
         description="Simple data analysis for clustering application",
         tags=["data-analysis","clustering"],
         path='clustering')
'''

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
    
def squashRNG(rng,apermol):
    """
    Reduces radius neighbors graph to a new graph based on molecules instead of
    atoms.
    
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
    rng = rng.toarray()
    newsh = (int(sh[0]/apermol),int(sh[1]/apermol))
    #pdb.set_trace()
    #molrng = lil_matrix(newsh)
    molrng = np.zeros(newsh)
    for i in range(0,newsh[0]):
        for j in range(i+1,newsh[1]):
            subrng = rng[apermol*i:apermol*(i+1),apermol*j:apermol*(j+1)]
            if subrng.max():
                molrng[i,j] = 1.0
                molrng[j,i] = 1.0
    return csr_matrix(molrng)
    
def squashRNGCOO(rng,apermol):
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
    newsh = (int(sh[0]/apermol),int(sh[1]/apermol))
    molrng = lil_matrix(newsh)
    rng = coo_matrix(rng)
    rows = rng.row//apermol
    cols = rng.col//apermol
    rowcols = rows * molrng.shape[1] + cols
    urowcols = np.unique(rowcols)
    rows = urowcols // molrng.shape[1]
    cols = urowcols % molrng.shape[1]
    #pdb.set_trace()
    for i in range(len(rows)):

        row = rows[i]
        col = cols[i]
        if col > row:
            molrng[row,col] = 1
    #pdb.set_trace()
    return csr_matrix(molrng)

def squashRNGCython(rng,apermol):
    """
    Reduces radius neighbors graph to a new graph based on molecules instead of
    atoms, but uses Cython code to improve speed.
    
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
    rng = rng.toarray()
    newsh = (int(sh[0]/apermol),int(sh[1]/apermol))
    #pdb.set_trace()
    #molrng = lil_matrix(newsh)
    molrng = np.zeros(newsh)
   
    molrng = subsquashRNG(rng,molrng,apermol)

                
    return csr_matrix(molrng)
    

def squashRNGPy(rng,apermol):
    """
    Reduces radius neighbors graph to a new graph based on molecules instead of
    atoms.  Dummy python debug test of Cython algorithm.
    
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
    rng = rng.toarray()
    newsh = (int(sh[0]/apermol),int(sh[1]/apermol))
    #pdb.set_trace()
    #molrng = lil_matrix(newsh)
    molrng = np.zeros(newsh)

    molrng = subsquashRNGPy(rng,molrng,apermol)

   #pdb.set_trace()
    return csr_matrix(molrng)  
    
def subsquashRNGPy(rng,molrng,apermol):
    """
    Python version of c algorithm that sets the block to 0 when all are 0
    and 1 if at least 1 is 1
    
     Parameters
    ----------
    rng: a numpy array as produced by a BallTree
    apermol: int
        the number of atoms in a molecule
        
    Returns
    -------
    molrng: a new graph
    """
    dim = np.shape(molrng)[0]
    sz = np.shape(rng)
    rng = rng.reshape((1,sz[0]*sz[1]))[0]
    molrng = molrng.reshape((1,dim*dim))[0]


    for i in range(dim):
        for j in range(i+1,dim):
            istart = apermol*i;
            iend = apermol*(i+1);
            jstart = apermol*j;
            jend = apermol*(j+1);
            curr = 0;
            #pdb.set_trace()
            for k in range(istart,iend):
                for m in range(jstart,jend):
                    if (rng[k*dim*apermol+m] != 0.):
                        curr = 1;
            #pdb.set_trace()                        

            if (curr == 1):
                molrng[dim*i+j] = 1.0;
                molrng[dim*j+i] = 1.0;
                        
    molrng = molrng.reshape((dim,dim))
    return molrng
 
    
def fixMisplacedArom(gsdfile,gsdout,idMiss,idPartner,idNotMiss,idNotPartner
                    ,molno,ats,ts):
    """
    opens a gsd file, gets the trajectory, then writes out in place with 
    the incorrectly placed aromatic placed correctly
    
    Parameters
    ----------
    gsdfile: string
        filename of the file to be rewritten
    gsdout: string
        where to write new stuff
    idMiss: the id of the misplaced aromatic within the molecule
    idPartner: the id of the partner to the misplaced aromatic within the mol
    idNotMiss: the complementary correctly placed aromatic
    idNotPartner: idNotMiss's partner
    ts: which timesteps of the trajectory to rewrite
    
    Notes
    -----
    pos(idMiss) = pos(idPartner) + (pos(idNotMiss) - pos(idNotPartner))
    """   
    traj = gsd.hoomd.open(gsdfile)
    trajnew = gsd.hoomd.open(gsdout,'wb')
    offset = molno
    idMisses = offset+idMiss + np.arange(0,molno*(ats-1),ats-1)
    idPartners = offset + idPartner + np.arange(0,molno*(ats-1),ats-1)
    idNotMisses = offset + idNotMiss + np.arange(0,molno*(ats-1),ats-1)
    idNotPartners = offset + idNotPartner + np.arange(0,molno*(ats-1),ats-1)
    for t in ts:
        snapshot = traj[t]
        box = snapshot.configuration.box[0:3]
        pos = snapshot.particles.position
        pds = pos[idNotMisses] - pos[idNotPartners]
        pds = pds - np.around(pds / box) * box
        pos[idMisses] = pos[idPartners] + pds
        snapnew = snapshot
        snapnew.particles.position = pos
        trajnew.append(snapnew)
        
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
    return np.min(cdist(xa,ya,metric='sqeuclidean'))

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
   
    #xa = np.reshape(x,[ats,3])
    #ya = np.reshape(y,[ats,3])
    mind = 10000.0
    support = '#include <math.h>'
    
    code = """
    int i,j;
    return_val = 0;
    double d;
    for (i = 0; i < Nx[0]/3; i++) {
       for (j = 0; j < Nx[0]/3; j++) {
           d = (x[3*i] - y[3*j]) * (x[3*i] - y[3*j])
               + (x[3*i + 1] - y[3*j + 1]) * (x[3*i + 1] - y[3*j + 1])
               + (x[3*i + 2] - y[3*j + 2]) * (x[3*i + 2] - y[3*j + 2]);
           if (d < mind){
               mind = d;            
           }
       }     
    }
    return_val = mind;
	
			"""
    mind = weave.inline(code,['x', 'y', 'mind'],
                        support_code = support, libraries = ['m'])
    #return np.min(euclidean_distances(xa,ya,squared=True))    
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

def fixCoords(pos,posinit,box):
    """
    fixes all coords based on the initial coordinate and 
    the periodic boundary conditions
    
    Parameters
    ----------
    pos: 1 x 3*ats numpy array
        positions of all the beads in the molecule
    posinit: 1 x 3 numpy array
        initial position on which the fixing is based
    box: 1 x 3 numpy array
        box dimensions
    """
    
    for i in range(int(len(pos)/3)):
        #pdb.set_trace()
        dr = pos[3*i:3*i+3] - posinit
        dr = dr - box*np.round(dr/box)
        pos[3*i:3*i+3] = dr + posinit
    return pos
    
class SnapSystem(object):
    """Class for running the full suite of analysis software """
    
    def __init__(self, traj, ats, molno, cldict,
                 clfunc={'contact':conOptDistanceCython,
                         'optical':conOptDistanceCython,
                         'aligned':alignDistancesCython}, 
                 compairs=np.array([[0,6],[1,7],[2,8],[3,9],[4,10],[5,11]]),
                 atype=u'LS',ttotal=-1,tstart=0,tpr=None,outGro='conf'):
        """ Initialize a full system of gsd snapshots over a trajectory.

        Parameters
        ----------
        traj: a gsd.hoomd trajectory or a gro or an xtc file name
        
        ats: dictionary
        the number of beads in a single molecule for each cluster type
        
        molno: int
        the number of molecules in the system
        
        cldict: dictionary
        keys are strings representing cluster types, ie contact, optical,
        aligned.  values are cutoffs
    
        clfunc: dictionary
            keys are strings representing cluster types. values are
            functions for distance computation
        
        compairs: numpy array
            for finding COM of aromatics for aligned clusters
        
        atype: label
            referring to how the aromatic beads are labeled in the trajectory
        
        ttotal: int
            the total length of the trajectory to be studied
            if -1, assume it is the same as the length of the provided
            trajectory
        tstart: int
            timestep to start at, defaults to zero 
            (last timestep = tstart + ttotal)
        tpr: string
            name of tpr file, used only with xtc trajectory
            
        outGro: string
            name of file to safe individual gro files to
        
        Attributes
        ----------
        mpi: bool
            True if the system can run in MPI, false if it has to run serially
        trajectory: gsd.hoomd trajectory
            the trajectory of snapshots in the system
        ats: int
            the number of beads per molecule
        molno: int
            the number of molecules in the system
        cldict: dict
            keys are strings representing cluster types, ie contact, optical,
            aligned.  values are cutoffs
        clsnaps: list of lists of clusterSnapshots
            a list for each cluster type in the dictionary
            each list is the snapshot at each timestep of the appropriate 
            type.  If mpi is True, then this list is padded with dummy clusters
            with NaN positions to make Scatter work correctly.
        atype = label
            referring to how aromatic beads are labeled in the trajectory
        comm: MPI communicator
        
        ------
        Raises
        ------
        NotImplementedError: 
            - if traj isn't a hoomd trajectory or a file ending
            in xtc or gro
            - if self.mpi is set to true for non hoomd stuff
            
        ValueError:
            - if tpr is set to None with an xtc file
            
        
        Notes
        -----
        Allows for MPI implementation of system if the size of the 
        MPI communicator is greater than 1 AND it's a gsd system rather than
        an XTC one
        
        """
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        self.comm = comm

        if size > 1:
            self.mpi = True
        else:
            self.mpi = False
        #pdb.set_trace()
        if (type(traj) is not str) and (type(traj) is not gsd.hoomd.HOOMDTrajectory):
            raise NotImplementedError("Invalid trajectory type")
        if (type(traj) is gsd.hoomd.HOOMDTrajectory):
            if self.mpi:
                raise NotImplementedError("MPI is only available for HOOMD trajectory types")
        if (type(traj) is str):
            spl = traj.split('.')
            ext = spl[len(spl)-1]
            if ext != 'gro' and ext != 'xtc':
                raise NotImplementedError("Invalid trajectory type")
            if ext == 'xtc' and tpr is None:
                raise ValueError("tpr must have a value for xtc trajectories")
        self.trajectory = traj
        self.ats = ats
        self.molno = molno
        self.cldict = cldict
        self.clfunc = clfunc
        self.clsnaps = {}
        self.atype = atype
        if ttotal == -1:
            ttotal = len(traj)
        if self.mpi:
            rank = comm.Get_rank()
            num = int(np.floor(ttotal / size))
            rem = ttotal % size
            if rank == 0:
                tslist = np.zeros((num + 1) * size).astype(int)    
                currid = 0
                for r in range(size):
                    if rem != 0:
                        if r < rem:
                            ts = r * (num + 1) + np.arange(num + 1) + tstart
                            tslist[currid:(len(ts)+currid)] = ts
                        else: 
                            ts = r * (num + 1) - (r - rem) + np.arange(num) + tstart
                            tslist[currid:(len(ts)+currid)] = ts
                            tslist[(len(ts)+currid):(len(ts) \
                            + currid + (r-rem)+1)] = -1
                        currid += num + 1
                    else:
                        tslist = np.arange(num * size) + tstart
                for ctype in cldict.keys():
                    if ctype == 'contact':
                        clusters = [ContactClusterSnapshot(t,traj,ats[ctype],
                                                           molno) \
                                                           for t in tslist]
                    elif ctype == 'optical':
                        clusters = [OpticalClusterSnapshot(t,traj,ats[ctype],
                                                           molno,
                                                           atype=atype) \
                                                           for t in tslist]
                    elif ctype == 'aligned':
                        clusters = [AlignedClusterSnapshot(t,traj,ats[ctype],
                                                           molno,
                                                           compairs=compairs,
                                                           atype=atype) \
                                                           for t in tslist]
                    else:
                        raise NotImplementedError("Unknown cluster type")
                    self.clsnaps[ctype] = clusters
        else:
            for ctype in cldict.keys():
                if ctype == 'contact':
                    if type(traj) is str:
                        if ext == 'gro':
                            clusters = [ContactClusterSnapshotXTC(t, traj, ats, 
                                                            molno) \
                                    for t in range(tstart,ttotal+tstart)]
                        else:
                            #pdb.set_trace()
                            flag = False
                            for t in range(tstart,ttotal+tstart):
                                if not os.path.isfile(outGro+str(t)+'.gro'):
                                    flag = True
                                    break
                            if flag:
                                grocall = \
    'echo 0 | trjconv -s {0} -f {1} -o {2}.gro -sep'.format(tpr,traj,outGro)
                                os.system(grocall)
                            clusters = [ContactClusterSnapshotXTC(t, 
                                            outGro+str(t)+'.gro',ats,molno) \
                                        for t in range(tstart,ttotal+tstart)]
                             
                    else:
                        clusters = \
                        [ContactClusterSnapshot(t,traj,ats[ctype],molno) \
                                for t in range(tstart,ttotal+tstart)]
                elif ctype == 'optical':
                    if type(traj) is str:
                        if ext == 'gro':
                            clusters = [OpticalClusterSnapshotXTC(t,traj,ats,
                                                              molno,compairs) \
                                    for t in range(tstart,ttotal+tstart)]
                        else:
                            flag = False
                            for t in range(tstart,ttotal+tstart):
                                if not os.path.isfile(outGro+str(t)+'.gro'):
                                    flag = True
                                    break
                            if flag:
                                grocall = \
    'echo 0 | trjconv -s {0} -f {1} -o {2}.gro -sep'.format(tpr,traj,outGro)
                                os.system(grocall)
                            clusters = [OpticalClusterSnapshotXTC(t, 
                                            outGro+str(t)+'.gro',ats,molno,
                                            compairs) \
                                        for t in range(tstart,ttotal+tstart)]
                    else:
                        clusters = \
                        [OpticalClusterSnapshot(t,traj,ats[ctype],molno,
                                                       atype=atype) \
                                        for t in range(tstart,ttotal+tstart)]
                elif ctype == 'aligned':
                    if type(traj) is str:
                        raise NotImplementedError("Aligned cluster only available for HOOMD type trajectories")
                    else:
                        clusters = \
                        [AlignedClusterSnapshot(t,traj,ats[ctype],molno,
                                                       compairs=compairs,
                                                       atype=atype) \
                                        for t in range(tstart,ttotal+tstart)]
                else:
                    raise NotImplementedError("Unknown cluster type")
                self.clsnaps[ctype] = clusters
        
    def get_clusters_mpi(self,ctype,ttotal=-1):
        """ Compute the clusters in each snapshot of the trajectory, using
        MPI parallelization.
        
        Parameters
        ----------
        ctype: string
            cluster type (contact, optical, aligned, etc)
    
        ttotal: int
            number of timesteps to compute for
        Raises
        ------
        NotImplementedError
            If the cluster type isn't one that's been programmed yet.        
        
        Notes
        ------        
        Partition up the snapshots, turn them into numpy arrays of 
        relevant data, and have each processor compute the cluster IDs,
        which is the only step that takes ages right now. Once computed,
        gather them all back up at root.
        
        """
        
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        if ttotal == -1:
            ttotal = len(self.trajectory)
        num = int(np.floor(ttotal / size))
        rem = ttotal % size
        traj = self.trajectory
        ats = self.ats
        molno = self.molno
        atype = self.atype
        if ctype not in ['contact','optical','aligned']:
            raise NotImplementedError('Unknown cluster type \
                                      in get_clusters_mpi')
    
        cutoff = self.cldict[ctype]    
        if rank == 0:
                
            clusters = self.clsnaps[ctype]
            carraylen = clusters[0].getCArrayLen()
            clusterarray = np.zeros(carraylen * len(clusters))
            cind = 0
            for cls in clusters:
                carray = cls.toArray()
                clusterarray[(cind * carraylen):(cind * carraylen + carraylen)]\
                = carray
                cind += 1
        else:
            if ctype == 'contact':
                tCSnap = ContactClusterSnapshot(0,traj,ats[ctype],molno)
            elif ctype == 'optical':
                tCSnap = OpticalClusterSnapshot(0,traj,ats[ctype],molno,
                                                atype=atype)
            elif ctype == 'aligned':
                tCSnap = AlignedClusterSnapshot(0,traj,ats[ctype],molno,
                                                atype=atype)
            else:
                tCSnap = ClusterSnapshot(0,traj,ats)
            carraylen = tCSnap.getCArrayLen()
            clusterarray = None
        if rem == 0:
            ncsnaps = num
        else:
            ncsnaps = num + 1
        carray_local = np.zeros(ncsnaps * carraylen)
        self.comm.Scatter(clusterarray,carray_local,root=0)
        
        #for each local cluster array, turn it into a cluster, compute the 
        #clusterIDs, pack the whole thing up as an array again, and send back
        #to root
        
        for i in range(ncsnaps):
            carrayi = carray_local[carraylen * i : (carraylen * i + carraylen)]
            #print("From rank {0}, snap {1}, array{2}".format(rank,i,carrayi))
            if not np.isnan(carrayi[4]):
                if ctype == 'contact':
                    clustSnap = ContactClusterSnapshot(0,carrayi,ats[ctype],
                                                       molno)
                elif ctype == 'optical':
                    clustSnap = OpticalClusterSnapshot(0,carrayi,ats[ctype],
                                                       molno,atype=atype)
                elif ctype == 'aligned':
                    clustSnap = AlignedClusterSnapshot(0,carrayi,ats[ctype],
                                                       molno,atype=atype)
               
                clustSnap.setClusterID(cutoff)
                try:
                    carray_local[carraylen * i : (carraylen * i + carraylen)]\
                    = clustSnap.toArray()
                except:
                    pdb.set_trace()
            #print("Part 2: From rank {0}, snap {1}, array{2}".format(rank,i,carrayi))
        self.comm.Barrier()
        self.comm.Gather(carray_local,clusterarray,root=0)
        
        if rank == 0:
           
            ind = 0
            nind = 0
            while ind < ttotal:
                carrayi = clusterarray[(carraylen * nind) : \
                                       (carraylen * nind + carraylen)]

                if not np.isnan(carrayi[4]):
                    if ctype == 'contact':
                        clustSnap = ContactClusterSnapshot(0,carrayi,
                                                           ats[ctype],molno)
                    elif ctype == 'optical':
                        clustSnap = OpticalClusterSnapshot(0,carrayi,
                                                           ats[ctype],molno,
                                                           atype=atype)
                    elif ctype == 'aligned':
                        clustSnap = AlignedClusterSnapshot(0,carrayi,
                                                           ats[ctype],molno,
                                                           atype=atype)
                    self.clsnaps[ctype][nind].clusterIDs = clustSnap.clusterIDs
                    #print("current pos: ",clustSnap.pos[0])
                    #print("current csizes: ",clustSnap.idsToSizes())
                    ind += 1
                nind +=1
        
    def get_clusters_serial(self,ctype,box,lcompute=None):
        """ Compute the clusters in each snapshot of the trajectory, doing
        so simply in serial.
        Parameters
        ----------
        ctype: string
            cluster type (contact, optical, aligned, etc)
        box: 3 x 1 numpy array
            box side lengths
        lcompute: string or None
            if a string, this is the filename to write the length distributions
            to after computation
        
        Raises
        ------
        NotImplementedError
            If the cluster type isn't one that's been programmed yet.       
        """
        if ctype not in self.cldict.keys():
            raise NotImplementedError("Unknown cluster type \
                                       in get_clusters_serial.")
        clusters = self.clsnaps[ctype]
        cutoff = self.cldict[ctype]
        func = self.clfunc[ctype]
        if lcompute is not None:
            lfile = open(lcompute,'w')

        for clustSnap in clusters:

            BT = clustSnap.setClusterID(cutoff)
            if lcompute is not None:
                ldistrib = clustSnap.getLengthDistribution(cutoff,box,func,
                                                BT=BT)
                for lmol in ldistrib:
                    lfile.write('{0} '.format(lmol))
                lfile.write('\n')
        if lcompute is not None:
            lfile.close()
        self.clsnaps[ctype] = clusters
    
    
        
    def get_clusters_from_file(self,ctype,fname):
        """ Compute the clusters in each snapshot of the trajectory from a
        given file name, assuming serial.
        
        Parameters
        ----------
        ctype: string
            cluster type (contact, optical, aligned, etc)
        fname: string
            file name where the cluster ID data is saved
            
        Raises
        ------
        NotImplementedError
            If the cluster type isn't one that's been programmed yet
        """
        if ctype not in self.cldict.keys():
            raise NotImplementedError("Unknown cluster type \
                                       in get_clusters_from_file.")
        clusters = self.clsnaps[ctype]
        for clustSnap in clusters:
            clustSnap.setClusterIDFromFile(fname)
        
        self.clsnaps[ctype] = clusters
        
    def getLengthDistribution(self,ctype,cutoff,box,func=conOptDistanceCython,writegsd=None,
                              writeldistrib=None):
        """ Gets the length distribution at each timestep and optionally
        writes it out to file.
        
        Parameters
        ----------
        ctype: string
            cluster type (contact, optical, aligned, etc)
        cutoff: float
            Cutoff for BallTree computation for unwrapping
        box: 1x3 numpy array
            box side lengths
        func: python function
            distance metric for BallTree computation
        writegsd: string or None
            used as the base filename to write out all clusters as separate
            gsd files. Mostly useful for debugging purposes.
        writeldistrib: string or None
            the filename to write out the length distributions of the clusters
            
        Returns
        -------
        ldistribt: T x molno numpy array
            contains the approximate end-end length that the cluster each
            molecule participates in is at each timestep
            
        Raises
        ------
        NotImplementedError
            If the cluster type isn't one that's been programmed yet
            
        Notes
        -----
        Computes an approximation to the end-end length as the largest 
        distance between two participating COM beads. This is not the
        best approximation if the aggregates are not very linear or if
        they are linear but curl up a lot. It fails for a spanning cluster.

        """
        
        
        if ctype not in self.cldict.keys():
            raise NotImplementedError("Unknown cluster type \
                                       in get_clusters_from_file.")
        clusters = self.clsnaps[ctype]
        ldistribt = np.zeros([len(self.trajectory),self.molno])
        ind = 0
        if writeldistrib is not None:
            f = open(writeldistrib,'w')
        for clustSnap in clusters:
            
            ldistrib = clustSnap.getLengthDistribution(cutoff,box,func,
                                            writegsd=writegsd)
            ldistribt[ind,:] = ldistrib
            if writeldistrib is not None:
                for endendl in ldistrib:
                    f.write('{0} '.format(endendl))
                f.write('\n')
            ind += 1
        if writeldistrib is not None:
            f.close()  
        return ldistribt                          
    
    def getMassAvVsTime(self,ctype,tstep=1):
        """ Returns a numpy array of two columns, with time on the left and 
        mass-averaged cluster size on the right.

        Parameters
        ----------
        ctype: string
            refers to cluster type for which to calculate this    
        tstep: float
            converts timestep to some non-reduced value if desired
            default is just 1            
            
        Returns
        -------
        mu2vtime: numpy array
        
        Raises
        ------
        NotImplementedError
            If the cluster type is one that hasn't been programmed yet
        """
        if self.comm.Get_rank() == 0:
            if ctype not in self.cldict.keys():
                raise NotImplementedError("Unknown cluster type.")
            clsnaps = self.clsnaps[ctype]
            mu2vtime = float('NaN')*np.ones([2,len(clsnaps)])
            ind = 0
            for clsnap in clsnaps:
                if not np.isnan(clsnap.pos[0][0]):
                    mu2vtime[0,ind] = ind * tstep
                    mu2vtime[1,ind] = massAvSize(clsnap.idsToSizes())
                    ind += 1
	m1 = mu2vtime[0,np.where(~np.isnan(mu2vtime[0]))[0]]
	m2 = mu2vtime[1,np.where(~np.isnan(mu2vtime[0]))[0]]
	mu2vtime = np.array([m1,m2])
        return mu2vtime
        
        
    def writeCIDs(self,ctype,fname):
        """ Write out the cluster indices as a file that can be opened
        and loaded later
        
        Parameters
        ----------
        ctype: string
            cluster type
        fname: string
            file name to write to                
            
        Raises
        ------
        NotImplementedError
            If the cluster type is one that hasn't been programmed yet
        """
        if ctype not in self.clsnaps.keys():
            raise NotImplementedError("Unknown cluster type in writeCIDs.")
        if self.comm.Get_rank() == 0:
            fid = open(fname,'w')
            clsnaps = self.clsnaps[ctype]
            for clsnap in clsnaps:
                if not np.isnan(clsnap.pos[0][0]):
                    cids = clsnap.clusterIDs
                    for cid in cids:
                        fid.write('{0} '.format(cid))
                    fid.write('\n')
            fid.close()
            
    def writeSizes(self,ctype,fname):
        """ Write out the cluster sizes as a file that can be opened
        and loaded later
        
        Parameters
        ----------
        ctype: string
            cluster type
        fname: string
            file name to write to                
            
        Raises
        ------
        NotImplementedError
            If the cluster type is one that hasn't been programmed yet
        """
        if ctype not in self.clsnaps.keys():
            raise NotImplementedError("Unknown cluster type in writeSizes")
        if self.comm.Get_rank() == 0:
            fid = open(fname,'w')
            clsnaps = self.clsnaps[ctype]
            for clsnap in clsnaps:
                if not np.isnan(clsnap.pos[0][0]):
                    csizes = clsnap.idsToSizes()
                    for csize in csizes:
                        fid.write('{0} '.format(csize))
                    fid.write('\n')
            fid.close()
        
        
class ClusterSnapshot(object):
    """Class for tracking the location of clusters at each time step"""
    
    def __init__(self, t, traj, ats):
        """ Initialize a ClusterSnapshot object.

        Parameters
        ----------
        t: timestep

        traj: a gsd.hoomd trajectory
        
        ats: the number of beads in a single molecule
        
        Raises
        ------
        RuntimeError
            if the number of particles doesn't divide evenly into molecules
        """
        snapshot = traj[t]
        self.timestep = t
        self.ats = ats
        binds = np.argsort(snapshot.particles.body)
        self.pos = snapshot.particles.position[binds]
        sz = np.shape(self.pos)
        if sz[0] % ats != 0:
            raise RuntimeError("Number of particles not divisible by number \
                                of beads per molecules.")
        self.nclusts = ats
        self.clusterIDs = np.zeros(sz[0]/ats)
        


class ContactClusterSnapshot(ClusterSnapshot):
    """Class for tracking the location of contact clusters at each time step
    
    Attributes
        ----------
    timestep: float
        timestep
    ats: int
        number of beads per molecule
    nclusts: int
        number of clusters in the snapshot
    pos: numpy array [M x 3*ats]
        locations of molecules and beads within molecules
        each molecule is its own line and then the locations of beads
        are flattened within that
    clusterIDs: list [len M]    
    """
   
    def __init__(self, t, trajectory, ats, molno):
        """ Initialize a ClusterSnapshot object.

        Parameters
        ----------
        t: timestep

        trajectory: gsd.hoomd trajectory or numpy array 
            numpy array is of size 4 + 3 * ats * molno 
            (ats is different for optical and aligned clusters)
        
        ats: the number of beads in a single molecule
        molno: the number of molecules in the system
 
        Raises
        ------
        RuntimeError
            if the number of particles does not divide evenly up into molecules
        
        Notes
        -----
        You can create a ClusterSnapshot object from either an array (for use
        with MPI) or from a HOOMD trajectory
        
        """
        self.timestep = t
        self.ats = ats
        if type(trajectory) is np.ndarray:
            carray = trajectory
            self.timestep = int(carray[0])
            self.ats = int(carray[2])
           
            self.nclusts = carray[1]
            pend = 4 + 3 * ats * molno
            self.pos = np.reshape(carray[4:pend],[molno,3*ats])
            self.clusterIDs = carray[pend:len(carray)]
        else:
            if t != -1: 
                snapshot = trajectory[t]
            
                binds = np.argsort(snapshot.particles.body)
                self.pos = snapshot.particles.position[binds]
                sz = np.shape(self.pos)
                if sz[0] % ats != 0:
                    raise RuntimeError("Number of particles not divisible by \
                                        number of beads per molecules.")
                #pdb.set_trace()
                self.pos = np.reshape(self.pos,[sz[0] / ats , 3 * ats])
            else:#create a dummy object to help with mpi scattering
                snapshot = trajectory[0]
                self.pos = snapshot.particles.position
                sz = np.shape(self.pos)
                self.pos = np.reshape(self.pos,[sz[0] / ats , 3 * ats])
                self.pos = float('NaN') * self.pos
            self.nclusts = molno
            self.clusterIDs = range(int(sz[0] / ats))
        
    def getCArrayLen(self):
        """
        returns the numpy length of an array made by toArray
        
        Returns
        -------
        carrayLen: int that is the length of the numpy array made by the
        toArray fcn
        """
        sz = np.shape(self.pos)
        molno = sz[0]
        carrayLen = 4 + 3 * self.ats * molno + molno
        return carrayLen
        
    def toArray(self):
        """
        Put all the cluster information into a numpy array, for use with
        mpi4py
        
        Returns
        -------
        carray: numpy array containing all information in a specific order
        Can be turned back into a cluster by calling the arrayToCluster
        function in this module
        
        Notes
        -----
        Contains:
        [timestep (size 1) nclusts (size 1) ats (size 1) 
        positions (size 3 * ats * molno)  clusterIDs (size molno)]
        """
        sz = np.shape(self.pos)
        carray = np.zeros(4 + 3 * self.ats * sz[0] + sz[0])
        carray[0] = self.timestep
        carray[1] = self.nclusts
        carray[2] = self.ats
        molno = sz[0]
        carray[3] = molno
        pend = 4 + 3 * self.ats * molno
        
        
        carray[4:pend] = np.reshape(self.pos,[1,3*self.ats*molno])
        
        clen = molno
        carray[pend:(pend + clen)] = self.clusterIDs
        return carray
    
        
    def setClusterID(self,cutoff):
        """
        Set the cluster IDs using getClusterID
        
        
        Parameters
        ----------
        cutoff: the squared distance molecules have to be within to be
        part of the same cluster

        Returns
        -------
        BT: BallTree of the system
        """        
        (nclusts,clusterIDs,BT) = \
        self.getClusterID(self.pos,cutoff,conOptDistanceCython)
        self.nclusts = nclusts
        self.clusterIDs = clusterIDs
        return BT
        
    def setClusterIDFromFile(self,fname,line=None):
        """
        Set the cluster IDs by opening a file and checking what they are
        
        Parameters
        ----------
        fname: string
            the name of the file that contains the clusterIDs
        line: int
	    the line number if it differs from the timestep of the cluster snap
        
        Returns
        -------
        None, just sets clusterIDs
        
        Notes
        -----
        File format is as written out by this code package
        """
        f = open(fname)
        lines = f.readlines()
        f.close()
        if line is None:
            line = self.timestep
        cIDs = lines[line].split()
        self.clusterIDs = np.array([int(float(cID)) for cID in cIDs])
        
    def getClusterID(self, positions,cutoff,func):
        """
        Find the ID of which cluster each molecule is in

        Parameters
        ----------
        cutoff: the squared distance molecules have to be within to be
        part of the same cluster

        Returns
        -------
        clusterIDs: numpy array of the cluster index of the cluster that
        each molecule occupies
        nclusts: number of clusters
        BT: BallTree for possible other computations

        """
        sz = np.shape(positions)
        pos3 = positions.reshape((int(sz[0]*sz[1]/3),3))
        BT = BallTree(pos3,metric='euclidean')
        rng = radius_neighbors_graph(BT,np.sqrt(cutoff))
        rng = squashRNGCOOCython(rng,int(sz[1]/3))
        (nclusts,clusterIDs) = connected_components(rng,directed=False,
                                            return_labels=True,
                                            connection='weak')
                                        
        #pdb.set_trace()
        return (nclusts,clusterIDs,BT)
    
    def idsToSizes(self):
        """
        Takes the cluster IDs and returns a list that for each molecule
        gives the size of the cluster it is participating in
        
        Returns
        -------
        clustSizes: numpy array
        """
        clustSizes = np.arange(len(self.clusterIDs))
        u,counts = np.unique(self.clusterIDs,return_counts=True)
        dcounts = dict(zip(u,counts))
        for cid in range(len(self.clusterIDs)):
            clustSizes[cid] = dcounts[self.clusterIDs[cid]]
        return clustSizes
    
    def fixPBC(self,cID,cutoff,box,func,writegsd=None,BT=None):
        """
        return positions for a particular cluster fixed across PBCs for 
        calculation of structural metrics like end-to-end length
        
        Parameters
        ----------
        cID: int
            the cluster index for this particular cluster
        cutoff: float
            distance within which to search for neighbors
        writegsd: bool
            if not none, write out a gsd file to this name that shows the
            resultant cluster
        box: 1x3 numpy array
            box side lengths
        func: python function
            distance metric for BallTree computation
        BT: precomputed BallTree for cluster
            if this is none, recompute the BallTree
            
        Returns
        -------
        pos: numpy array of floats
    
            the resultant positions of the cluster
            
        Raises
        ------
        RuntimeError: if there is more than one connected component
        
        Notes
        -----
        Currently origin is in the center of the box; for these purposes,
        all positions are reset such that the origin is at the corner.
        """
        inds = np.where(self.clusterIDs==cID)[0]
        positions = self.pos[inds,:]
        sz = np.shape(positions)
        
        
        fixedXYZ = positions.copy()
        potInds = range(1,int(sz[0]))
        #if BT is None:
        BT = BallTree(positions.reshape((int(sz[0]*sz[1]/3),3)),
                                         metric='euclidean')
        rng = radius_neighbors_graph(BT,np.sqrt(cutoff))
        rng = squashRNGCOOCython(rng,int(sz[1]/3))
        (nCC,CC) = connected_components(rng,connection='weak')
        
        if nCC != 1:
            raise RuntimeError("This isn't a fully connected cluster.")
        fixedXYZ[0,:] = fixCoords(fixedXYZ[0,:].copy(),fixedXYZ[0,0:3].copy(),
                                  box)
        correctInds = [0]
        while len(correctInds) > 0:
            mol = correctInds.pop()
            
            
            #neighs = BT.query_radius(positions[mol,:].reshape(1,-1),r=cutoff)[0]
            #neighs = neighs.remove(mol)
            neighs = np.where(rng[mol,:].toarray()[0]==1)[0]
            for n in neighs:
                #pdb.set_trace()
                if n in potInds:
                    potInds.remove(n)
                    correctInds.append(n)
                    fixedXYZ[n,:] = fixCoords(fixedXYZ[n,:].copy(),
                                              fixedXYZ[mol,0:3].copy(),box)
                else:
                    continue
        if writegsd is not None:
            f = gsd.hoomd.open(writegsd,'wb')
            s = gsd.hoomd.Snapshot()
            s.particles.N = sz[0]*sz[1]/3
            s.particles.position = fixedXYZ
            s.configuration.box = np.concatenate((box,[0,0,0]))
            f.append(s)
        return fixedXYZ

    def getLengthDistribution(self,cutoff,box,func=conOptDistanceCython,
                              writegsd=None,BT=None):
        """ Finds the end-to-end cluster length distribution
        
        Parameters
        ----------
        cutoff: float
            Cutoff for BallTree computation for unwrapping
        box: 1x3 numpy array
            box side lengths
        func: python function
            distance metric for BallTree computation
        writegsd: string or None
            used as the base filename to write out all clusters as separate
            gsd files. Mostly useful for debugging purposes.
        BT: None or BallTree
            BallTree for cluster computation
            Recomputes if None
        Returns
        -------
        ldistrib: 1 x molno numpy array
            length of the cluster each molecule belongs to
        
        """
        
        ldistrib = np.zeros(len(self.pos))
        
        for cID in range(self.nclusts):
            
            inds = np.where(self.clusterIDs==cID)[0]
            if len(inds) > 1:
                if writegsd is not None:
                    cIDpos = self.fixPBC(cID,cutoff,box,func,
                                         writegsd=writegsd+str(cID)+'.gsd',
                                         BT=BT)
                else:
                    
                    cIDpos = self.fixPBC(cID,cutoff,box,func,BT=BT)
                sz = np.shape(cIDpos)
                #extract COM positions
                xcom = np.sum(cIDpos[:,range(0,sz[1],3)],axis=1)/(sz[1]/3.)
                ycom = np.sum(cIDpos[:,range(1,sz[1],3)],axis=1)/(sz[1]/3.)
                zcom = np.sum(cIDpos[:,range(2,sz[1],3)],axis=1)/(sz[1]/3.)
                cIDposcom = np.array([xcom,ycom,zcom])
                endendl = np.sqrt(max(pdist(cIDposcom.transpose(),metric='sqeuclidean')))
                
                ldistrib[inds] = endendl
        return ldistrib
        
class OpticalClusterSnapshot(ContactClusterSnapshot):
    """Class for tracking the location of optical clusters at each time step"""
    
            
    def __init__(self, t, trajectory, ats, molno, atype=u'LS'):
        """ Initialize a ClusterSnapshot object.

        Parameters
        ----------
        t: timestep

        trajectory: gsd.hoomd trajectory or numpy array 
            numpy array is of size 4 + 3 * ats * molno 
            (ats is different for optical and aligned clusters)
        
        ats: the number of aromatics in a single molecule
        molno: the number of molecules in the system
        compairs: m x n numpy array
            these are the comparative indices of the beads making up each
            aromatic group, where m is the number of aromatics and n is the
            number of beads in the group, eg for two beads representing a
            ring in the 3-core model, this should be
            [[0,6],[1,7],[2,8],[3,9],[4,10],[5,11]] 
        atype: hoomd bead type
            should be the type referring to the aromatic beads
        Raises
        ------
        RuntimeError
            if the number of particles does not divide evenly up into molecules
        
        Notes
        -----
        You can create a ClusterSnapshot object from either an array (for use
        with MPI) or from a HOOMD trajectory
        
        An optical cluster snapshot tracks the positions of the COMs of the
        optical clusters, rather than the positions of the separate beads,
        as the contact cluster does
        """
        self.timestep = t
        self.ats = ats
        if type(trajectory) is np.ndarray:
            carray = trajectory
            self.timestep = int(carray[0])
            self.ats = int(carray[2])
           
            self.nclusts = carray[1]
            pend = 4 + 3 * ats * molno
            self.pos = np.reshape(carray[4:pend],[molno,3*ats])
            self.clusterIDs = carray[pend:len(carray)]
        else:
            if t != -1: 
                snapshot = trajectory[t]
                
                
                #self.pos = self.getComs(compairs,atype,trajectory[t],molno)
                tind = snapshot.particles.types.index(atype)
                types = snapshot.particles.typeid
       
                self.pos = \
                snapshot.particles.position[np.where(types==tind)[0]]
                sz = np.shape(self.pos)
                if sz[0] % ats != 0:
                    raise RuntimeError("Number of particles not divisible by \
                                        number of beads per molecules.")
                self.pos = np.reshape(self.pos,[sz[0] / ats , 3 * ats])
            else:#create a dummy object to help with mpi scattering
                snapshot = trajectory[0]
                #self.pos = self.getComs(compairs,atype,snapshot,molno)
                
                
                tind = snapshot.particles.types.index(atype)
                types = snapshot.particles.typeid
       
                self.pos = \
                snapshot.particles.position[np.where(types==tind)[0]]
                sz = np.shape(self.pos)
                self.pos = np.reshape(self.pos,[sz[0] / ats , 3 * ats])
                self.pos = float('NaN') * self.pos
            self.nclusts = molno
            self.clusterIDs = range(int(sz[0] / ats))
    
class AlignedClusterSnapshot(OpticalClusterSnapshot):
    """Class for tracking the location of aligned clusters at each time step"""
    
    def getComsGeneral(self,compairs,atype,snapshot,molno):
        """Helper function to get the COMs of a subset of beads
        
        Parameters
        ----------
        compairs:  m x n numpy array
            these are the comparative indices of the beads making up each
            aromatic group, where m is the number of aromatics and n is the
            number of beads in the group, eg for two beads representing a
            ring in the 3-core model, this should be
            [[0,6],[1,7],[2,8],[3,9],[4,10],[5,11]] 
        atype: hoomd bead type
            should be the type referring to the aromatic beads
        snapshot: gsd snapshot at the particular time of interest
        molno: int
            number of molecules in snapshot
        
        Returns
        -------
        aCOMS: nPairs x 3 numpy array
            array of COM positions for each bead
            
        Raises
        ------
        RuntimeError 
            if the number of beads in the aromatics isn't equal to the 
            total number of aromatics * beads in an aromatic
        
            
        Notes
        -----
        This is the more general way of finding COM and can be used in the 
        future but currently should not be called.
        """
        tind = snapshot.particles.types.index(atype)
        types = snapshot.particles.typeid
        ats = self.ats
        aBeads = snapshot.particles.position[np.where(types==tind)[0]]
        pairShape = np.shape(compairs)
        nPairs = pairShape[0]
        aromSize = pairShape[1]
        beadNo = np.shape(aBeads)[0]
        if nPairs * aromSize != beadNo / molno:
            raise RuntimeError("number of beads ({0} in {1} molecules)\
            does not divide cleanly \
            among aromatics ({2}) of size {3}".format(beadNo,molno,nPairs,
                                                     aromSize))
        aCOMs = np.zeros([nPairs * molno,3])
        for moli in range(molno):
            aBeadsMol = aBeads[(moli * beadNo / molno):(moli * beadNo / molno)\
                                + beadNo / molno,:]
            for m in range(nPairs):
                
                    aCOMs[moli*nPairs + m,:] = np.mean(aBeadsMol[compairs[m]],
                                                       axis=0)

        return aCOMs

    
    def __init__(self, t, trajectory, ats, molno, 
                 compairs=np.array([[0,6],[1,7],[2,8],[3,9],[4,10],[5,11]]), 
                 atype=u'LS'):
        """ Initialize a ClusterSnapshot object.

        Parameters
        ----------
        t: timestep

        trajectory: gsd.hoomd trajectory or numpy array 
            numpy array is of size 4 + 3 * ats * molno 
            (ats is different for optical and aligned clusters)
        
        ats: the number of aromatics in a single molecule
        molno: the number of molecules in the system
        compairs: m x n numpy array
            these are the comparative indices of the beads making up each
            aromatic group, where m is the number of aromatics and n is the
            number of beads in the group, eg for two beads representing a
            ring in the 3-core model, this should be
            [[0,6],[1,7],[2,8],[3,9],[4,10],[5,11]] 
        atype: hoomd bead type
            should be the type referring to the aromatic beads
        Raises
        ------
        RuntimeError
            if the number of particles does not divide evenly up into molecules
        
        Notes
        -----
        You can create a ClusterSnapshot object from either an array (for use
        with MPI) or from a HOOMD trajectory
        
        An aligned cluster snapshot just uses a different distance metric
        from an optical cluster snapshot
        """
        self.timestep = t
        self.ats = ats
        
        if type(trajectory) is np.ndarray:
            carray = trajectory
            self.timestep = int(carray[0])
            self.ats = int(carray[2])
           
            self.nclusts = carray[1]
            pend = 4 + 3 * ats * molno
            self.pos = np.reshape(carray[4:pend],[molno,3*ats])
            self.clusterIDs = carray[pend:len(carray)]
            self.box = None
        else:
            if t != -1: 
                snapshot = trajectory[t]
                
                self.box = snapshot.configuration.box
                self.pos = self.getComs(compairs,atype,trajectory[t],molno)
                sz = np.shape(self.pos)
                if sz[0] % ats != 0:
                    raise RuntimeError("Number of particles not divisible by \
                                        number of beads per molecules.")
                self.pos = np.reshape(self.pos,[sz[0] / ats , 3 * ats])
                
            else:#create a dummy object to help with mpi scattering
                snapshot = trajectory[0]
                self.box = snapshot.configuration.box
                self.pos = self.getComs(compairs,atype,snapshot,molno)
                sz = np.shape(self.pos)
                self.pos = np.reshape(self.pos,[sz[0] / ats , 3 * ats])
                self.pos = float('NaN') * self.pos
                
            self.nclusts = molno
            self.clusterIDs = range(int(sz[0] / ats))
        
    def getComs(self,compairs,atype,snapshot,molno,missingID=None):
        """Helper function to get the COMs of a subset of beads
        
        Parameters
        ----------
        compairs:  m x n numpy array
            these are the comparative indices of the beads making up each
            aromatic group, where m is the number of aromatics and n is the
            number of beads in the group, eg for two beads representing a
            ring in the 3-core model, this should be
            [[0,6],[1,7],[2,8],[3,9],[4,10],[5,11]] 
        atype: hoomd bead type
            should be the type referring to the aromatic beads
        snapshot: gsd snapshot at the particular time of interest
        molno: int
            number of molecules in snapshot

        
        Returns
        -------
        aCOMS: nPairs x 3 numpy array
            array of COM positions for each bead
            
        Raises
        ------
        RuntimeError 
            if the number of beads in the aromatics isn't equal to the 
            total number of aromatics * beads in an aromatic
            
        RuntimeError
            if the pairs aren't pairs -> that requires a DIFFERENT KIND OF
            CLUSTER

            
        NotImplementedError
            if box isn't set
            
        Notes
        -----
        For this type of cluster, we check the vector pointing between the 
        first bead pair and assume that the COM is located at bead1 + 1/2(vec)
        for all three COMs
        
        This will *only* work for COM pairs of beads and you need to know
        which way rvec should be going! (which depends on which bead is
        missing, if any of them are.)  If there is a bead missing from the
        pairs you MUST check which one it is and pass in whether rvec
        should be reversed.
        """
       
        if self.box is None:
            raise NotImplementedError("You are running on a cluster created from an array, which does not yet support box type analysis.")
        tind = snapshot.particles.types.index(atype)
        types = snapshot.particles.typeid
     
        aBeads = snapshot.particles.position[np.where(types==tind)[0]]
        pairShape = np.shape(compairs)
        nPairs = pairShape[0]
        aromSize = pairShape[1]
        if pairShape[1] != 2:
            raise RuntimeError("Not pairs.  Call the general getCOM function")
        beadNo = np.shape(aBeads)[0]
        if nPairs * aromSize != beadNo / molno:
            raise RuntimeError("number of beads ({0} in {1} molecules)\
            does not divide cleanly \
            among aromatics ({2}) of size {3}".format(beadNo,molno,nPairs,
                                                     aromSize))
        aCOMs = np.zeros([nPairs * molno,3])
        for moli in range(molno):
            aBeadsMol = aBeads[(moli * beadNo / molno):(moli * beadNo / molno)\
                                + beadNo / molno,:]
            #pdb.set_trace()
            
            rvec = (aBeadsMol[compairs[0][1]] - aBeadsMol[compairs[0][0]])/2
            rvec = rvec - np.around(rvec/self.box[0:3])*self.box[0:3]
            
            for m in range(nPairs):
                if compairs[m][1] == missingID:
                    rvec = (aBeadsMol[compairs[0][1]] \
                           - aBeadsMol[compairs[0][0]])/2
                    rvec = rvec - np.around(rvec/self.box[0:3])*self.box[0:3]
                    comloc = aBeadsMol[compairs[m][0]]+rvec
                    #pdb.set_trace()
                elif compairs[m][0] == missingID:
                    rvec = (aBeadsMol[compairs[0][0]] \
                           - aBeadsMol[compairs[0][1]])/2
                    rvec = rvec - np.around(rvec/self.box[0:3])*self.box[0:3]
                    comloc = aBeadsMol[compairs[m][0]]+rvec
                    #pdb.set_trace()
                else:
                    cv = aBeadsMol[compairs[m][0]] - aBeadsMol[compairs[m][1]]
                    cv = cv - np.around(cv/self.box[0:3])*self.box[0:3]
                    comloc = aBeadsMol[compairs[m][1]]+cv/2
                    #comloc = np.mean(aBeadsMol[compairs[m]],axis=0)
                    #pdb.set_trace()
                aCOMs[moli*nPairs + m,:] = comloc

        return aCOMs       
    def writeCOMsGSD(self,gsdname):
        """ Write out a GSD file of this snapshot that shows the locations of
        the aligned COMs after initialization
        
        Parameters
        ----------
        gsdname: string
            what name to save the file to
        
        """
        try:
            gsdf = gsd.hoomd.open(gsdname,'ab')
        except IOError:
            gsdf = gsd.hoomd.open(gsdname,'wb')
        sz = np.shape(self.pos)
        molno = sz[0]
        pos = np.reshape(self.pos,[sz[0]*self.ats,3])
        #pdb.set_trace()
        pN = sz[0]*self.ats
        ptypes = ['A']
        ptypeid = np.zeros(molno*self.ats).astype(int)
        pbox = self.box
        s = gsd.hoomd.Snapshot()
        s.particles.N = pN
        s.configuration.step = self.timestep
        s.particles.types = ptypes
        s.particles.typeid = ptypeid
        s.configuration.box = pbox
        s.particles.position = pos
        gsdf.append(s)
        
    def setClusterID(self,cutoff):
        """
        Set the cluster IDs using getClusterID
        
        
        Parameters
        ----------
        cutoff: the squared distance molecules have to be within to be
        part of the same cluster

        Returns
        -------
        BT: BallTree
            for length computation
        """        
        (nclusts,clusterIDs,BT) = \
        self.getClusterID(self.pos,cutoff,alignDistancesCython)
        self.nclusts = nclusts
        self.clusterIDs = clusterIDs
        return BT
        
    def getClusterID(self, positions,cutoff,func):
        """
        Find the ID of which cluster each molecule is in

        Parameters
        ----------
        cutoff: the squared distance molecules have to be within to be
        part of the same cluster

        Returns
        -------
        clusterIDs: numpy array of the cluster index of the cluster that
        each molecule occupies
        nclusts: number of clusters
        BT: BallTree for possible other computations

        """
        
        BT = BallTree(positions,metric='pyfunc',
                                        func=func)
        rng = radius_neighbors_graph(BT,cutoff)
        (nclusts,clusterIDs) = connected_components(rng,directed=False,
                                            return_labels=True)
        
        return (nclusts,clusterIDs,BT)
        
class ContactClusterSnapshotXTC(ContactClusterSnapshot):
    """ Class for tracking contact cluster locations that are initialized
    from an xtc/Gromacs file instead of a HOOMD one
    
    Attributes
        ----------
    timestep: float
        timestep
    ats: int
        number of beads per molecule
    nclusts: int
        number of clusters in the snapshot
    pos: numpy array [M x 3*ats]
        locations of molecules and beads within molecules
        each molecule is its own line and then the locations of beads
        are flattened within that
    clusterIDs: list [len M]        
    """
    
    def readGro(self,fName): 
        """ Get a list of positions from a Gromacs .gro file
        Parameters
        ----------
        fname: string
            name of .gro file
        Returns
        -------
        pos: numpy vector [len 3 * molecules * ats]
            1D list of positions in .gro file
        """
        with open(fName, 'r') as myF:
            myLns = myF.read().splitlines()
        boxL1 = float(myLns[len(myLns)-1].split()[0])
        boxL2 = float(myLns[len(myLns)-1].split()[1])
        boxL3 = float(myLns[len(myLns)-1].split()[2])
        return (np.array([[float(myLns[i][20:].split()[0]),
                           float(myLns[i][20:].split()[1]),
                           float(myLns[i][20:].split()[2])]\
                           for i in range(2, len(myLns)-1)]).flatten(),
                           np.array([boxL1,boxL2,boxL3]))



    
    def __init__(self, t, trj, ats, molno):
        """ Initialize a ContactClusterSnapshotXTC object.

        Parameters
        ----------
        t: timestep

        trj: Gromacs trajectory name (xtc format)
        
        
        ats: the number of beads in a single molecule
        
        molno: the number of molecules in the system
        
        
            the index of the cluster that each molecule belongs to
            
            
        Raises
        ------
        RuntimeError
            if the number of particles does not divide evenly up into molecules
        
        Notes
        -----
        You can create a ClusterSnapshot object from either an array (for use
        with MPI) or from a HOOMD trajectory
        
        """
        self.timestep = t
        self.ats = ats
        self.nclusts = molno
        self.clusterIDs = np.zeros(molno)
        self.pos = self.readGro(trj)[0]
        
        if len(self.pos) != 3 * molno * ats:
            raise RuntimeError("incorrect number of atoms or molecules")
        #pdb.set_trace()
        self.pos = np.reshape(self.pos,[molno,3*ats])
        

class OpticalClusterSnapshotXTC(ContactClusterSnapshotXTC):
    """ Class for tracking optical cluster locations that are initialized
    from an xtc/Gromacs file instead of a HOOMD one
    
    Attributes
        ----------
    timestep: float
        timestep
    ats: int
        number of beads per molecule
    nclusts: int
        number of clusters in the snapshot
    pos: numpy array [M x 3*ats]
        locations of molecules and beads within molecules
        each molecule is its own line and then the locations of beads
        are flattened within that
    clusterIDs: list [len M]        
    """
    
    def __init__(self, t, trj, ats, molno, comIDs):
        """ Initialize a ContactClusterSnapshotXTC object.

        Parameters
        ----------
        t: timestep

        trj: Gromacs trajectory name (xtc format)
        
        tpr: Gromacs run file name (tpr format)
        
        outGro: name for an output Gromacs .gro file
        
        ats: the number of beads in a single molecule
        
        molno: the number of molecules in the system
        
        
            the index of the cluster that each molecule belongs to
            
        comIDs: N x M numpy array of ints
            bead IDs of the beads in the N cores with M participating beads
            each
            
        Raises
        ------
        RuntimeError
            if the number of particles does not divide evenly up into molecules
        
        Notes
        -----
        You can create a ClusterSnapshot object from either an array (for use
        with MPI) or from a HOOMD trajectory
        
        """
        self.timestep = t
        self.ats = ats
        self.nclusts = molno
        self.clusterIDs = np.zeros(molno)
        self.pos = self.readGro(trj)[0]
        if len(self.pos) != 3 * molno * ats:
            raise RuntimeError("incorrect number of atoms or molecules")
        #pdb.set_trace()
        self.pos = np.reshape(self.pos,[molno,3*ats])
        M = np.shape(comIDs)[1]
        pos = np.zeros((molno,3*np.shape(comIDs)[0]))
        for mol in range(molno):
            for com in range(np.shape(comIDs)[0]):
                inds = comIDs[com,:]
                compos = np.array([self.pos[mol,3*inds+i].sum()/M \
                for i in range(3)])
                pos[mol,3*com:(3*com+3)] = compos
        self.pos = pos
