from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import gsd.hoomd
import sklearn
import scipy.optimize as opt
from sklearn.neighbors import BallTree
from sklearn.neighbors import radius_neighbors_graph
from scipy.spatial.distance import cdist
from scipy.special import erf
from scipy.sparse.csgraph import connected_components
from .due import due, Doi
from mpi4py import MPI

__all__ = ["ClusterSnapshot", "ContactClusterSnapshot","SnapSystem", "transform_data","conOptDistance","getContactClusterID","fromArray"]


# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("10.1167/13.9.30"),
         description="Simple data analysis for clustering application",
         tags=["data-analysis","clustering"],
         path='clustering')

def fromArray(carray,trajectory,ctype='contact'):
    """
    Function that takes a numpy array as input and outputs a cluster with
    the given data
    
    Parameters
    ----------
    carray: numpy array
    
    Returns
    -------
    C: cluster of chosen type
    """
    if ctype == 'contact':
        t = int(carray[0])
        ats = int(carray[2])
        molno = int(carray[3])
        C = ContactClusterSnapshot(t,trajectory,ats)
        C.nclusts = carray[1]
        pend = 4 + 3 * ats * molno
        C.pos = np.reshape(carray[4:pend],[molno,3*ats])
        C.clusterIDs = carray[pend:len(carray)]
    else:
        raise NotImplementedError("Incorrect cluster type")
    return C
    
def transform_data(data):
    """
    Function that takes experimental data and gives us the
    dependent/independent variables for analysis.

    Parameters
    ----------
    data : Pandas DataFrame or string.
        If this is a DataFrame, it should have the columns `contrast1` and
        `answer` from which the dependent and independent variables will be
        extracted. If this is a string, it should be the full path to a csv
        file that contains data that can be read into a DataFrame with this
        specification.

    Returns
    -------
    x : array
        The unique contrast differences.
    y : array
        The proportion of '2' answers in each contrast difference
    n : array
        The number of trials in each x,y condition
    """
    
def getContactClusterID(positions,cutoff):
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

        """
        
        BT = BallTree(positions,metric='pyfunc',
                                        func=conOptDistance)
        rng = radius_neighbors_graph(BT,cutoff)
        (nclusts,clusterIDs) = connected_components(rng,directed=False,
                                            return_labels=True)
        
        return (nclusts,clusterIDs)
        
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
    r : float
        The distance between x and y computed as the minimum distance
        between any two beads in the molecules
    """
    if len(x) % 3 != 0:
        raise RuntimeError("3D array has a number of entries not divisible \
                            by 3.")
    ats = len(x)/3
    xa = np.reshape(x,[ats,3])
    ya = np.reshape(y,[ats,3])
    #return np.min(euclidean_distances(xa,ya,squared=True))    
    return np.min(cdist(xa,ya,metric='sqeuclidean'))
            
class SnapSystem(object):
    """Class for running the full suite of analysis software """
    
    def __init__(self, traj, ats, molno, cldict,ttotal=-1):
        """ Initialize a full system of gsd snapshots over a trajectory.

        Parameters
        ----------
        traj: a gsd.hoomd trajectory
        
        ats: int
        the number of beads in a single molecule
        
        molno: int
        the number of molecules in the system
        
        cldict: dictionary
        keys are strings representing cluster types, ie contact, optical,
        aligned.  values are cutoffs
        
        ttotal: int
            the total length of the trajectory to be studied
            if -1, assume it is the same as the length of the provided
            trajectory
        
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
        comm: MPI communicator
        
        
        Notes
        -----
        Allows for MPI implementation of system if the size of the 
        MPI communicator is greater than 1.
        
        """
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        self.comm = comm

        if size > 1:
            self.mpi = True
        else:
            self.mpi = False
        
        
        self.trajectory = traj
        self.ats = ats
        self.molno = molno
        self.cldict = cldict
        self.clsnaps = {}
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
                            ts = r * (num + 1) + np.arange(num + 1)
                            tslist[currid:(len(ts)+currid)] = ts
                        else: 
                            ts = r * (num + 1) - (r - rem) + np.arange(num)
                            tslist[currid:(len(ts)+currid)] = ts
                            tslist[(len(ts)+currid):(len(ts) \
                            + currid + (r-rem)+1)] = -1
                        currid += num + 1
                    else:
                        tslist = np.arange(num * size)
                for ctype in cldict.keys():
                    if ctype == 'contact':
                        clusters = \
                        [ContactClusterSnapshot(t,traj,ats) for t in tslist]
                    else:
                        raise NotImplementedError("Unknown cluster type")
                    self.clsnaps[ctype] = clusters
        else:
            for ctype in cldict.keys():
                if ctype == 'contact':
                   
                    clusters = \
                    [ContactClusterSnapshot(t,traj,ats) for t in range(ttotal)]
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
        
        if ctype not in ['contact']:
            raise NotImplementedError('Unknown cluster type')
    
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
                tCSnap = ContactClusterSnapshot(0,traj,ats)
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
                clustSnap = fromArray(carrayi,traj,ctype=ctype)
                clustSnap.setClusterID(cutoff)
                carray_local[carraylen * i : (carraylen * i + carraylen)]\
                = clustSnap.toArray()
            #print("Part 2: From rank {0}, snap {1}, array{2}".format(rank,i,carrayi))
        self.comm.Barrier()
        self.comm.Gather(carray_local,clusterarray,root=0)
        
        if rank == 0:
           
            ind = 0
            nind = 0
            while ind < ttotal:
                carrayi = clusterarray[carraylen * nind : (carraylen * nind + carraylen)]

                if not np.isnan(carrayi[4]):
                    clustSnap = fromArray(carrayi,traj,ctype='contact')
                    self.clsnaps[ctype][nind].clusterIDs = clustSnap.clusterIDs
                    #print("current pos: ",clustSnap.pos[0])
                    #print("current csizes: ",clustSnap.idsToSizes())
                    ind += 1
                nind +=1
        
    def get_clusters_serial(self,ctype):
        """ Compute the clusters in each snapshot of the trajectory, doing
        so simply in serial.
        Parameters
        ----------
        ctype: string
            cluster type (contact, optical, aligned, etc)
        
        Raises
        ------
        NotImplementedError
            If the cluster type isn't one that's been programmed yet.       
        """
        if ctype not in ['contact']:
            raise NotImplementedError("Unknown cluster type.")
        clusters = self.clsnaps[ctype]
        cutoff = self.cldict[ctype]
        if ctype == 'contact':
            for clustSnap in clusters:
                clustSnap.setClusterID(cutoff)
        self.clsnaps[ctype] = clusters
    
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
                    mu2vtime[1,ind] = clsnap.massAvSize(clsnap.idsToSizes())
                    ind += 1
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
        if self.comm.Get_rank() == 0:
            print("really writing")
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
        if self.comm.Get_rank() == 0:
            print("really writing sizes")
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
    """Class for tracking the location of contact clusters at each time step"""
    
    def __init__(self, t, trajectory, ats):
        """ Initialize a ClusterSnapshot object.

        Parameters
        ----------
        t: timestep

        trajectory: gsd.hoomd trajectory
        
        ats: the number of beads in a single molecule
        
        Raises
        ------
        RuntimeError
            if the number of particles does not divide evenly up into molecules
        
        """
        self.timestep = t
        self.ats = ats
        
        if t != -1: 
            snapshot = trajectory[t]
        
            binds = np.argsort(snapshot.particles.body)
            self.pos = snapshot.particles.position[binds]
            sz = np.shape(self.pos)
            if sz[0] % ats != 0:
                raise RuntimeError("Number of particles not divisible by \
                                    number of beads per molecules.")
            self.pos = np.reshape(self.pos,[sz[0] / ats , 3 * ats])
        else:#create a dummy object to help with mpi scattering
            snapshot = trajectory[0]
            self.pos = snapshot.particles.position
            sz = np.shape(self.pos)
            self.pos = np.reshape(self.pos,[sz[0] / ats , 3 * ats])
            self.pos = float('NaN') * self.pos
        self.nclusts = ats
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
        None, just sets clusterIDs
        """        
        (nclusts,clusterIDs) = getContactClusterID(self.pos,cutoff)
        self.nclusts = nclusts
        self.clusterIDs = clusterIDs
    
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
        
    def massAvSize(self,csizes):
        """
        Given the cluster sizes list, returns the mass averaged cluster size
        of the snapshot
        
        Parameters
        ----------
        csizes: numpy array as returned by idsToSizes
        
        Returns
        -------
        mu2: float, the mass-averaged cluster size
        """
        umass,counts = np.unique(csizes,return_counts=True)
        mu2 = (umass*umass*counts).sum() / (umass*counts).sum()
        return mu2
        



