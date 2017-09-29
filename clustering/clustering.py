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

__all__ = ["ClusterSnapshot", "ContactClusterSnapshot","Fit", "transform_data","conOptDistance","getContactClusterID","fromArray"]


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
            


class ClusterSnapshot(object):
    """Class for tracking the location of clusters at each time step"""
    
    def __init__(self, t, traj, ats):
        """ Initialize a ClusterSnapshot object.

        Parameters
        ----------
        t: timestep

        snapshot: a gsd.hoomd snapshot at some time t
        
        ats: the number of beads in a single molecule
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

        snapshot: a gsd.hoomd snapshot at some time t
        
        ats: the number of beads in a single molecule
        
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
        


class Fit(object):
    """
    Class for representing a fit of a model to data
    """
    def __init__(self, model, params):
        """
        Initialize a :class:`Fit` object.

        Parameters
        ----------
        model : a :class:`Model` instance
            An object representing the model used

        params : array or list
            The parameters of the model evaluated for the data

        """
        self.model = model
        self.params = params

    def predict(self, x):
        """
        Predict values of the dependent variable based on values of the
        indpendent variable.

        Parameters
        ----------
        x : float or array
            Values of the independent variable. Can be values presented in
            the experiment. For out-of-sample prediction (e.g. in
            cross-validation), these can be values
            that were not presented in the experiment.

        Returns
        -------
        y : float or array
            Predicted values of the dependent variable, corresponding to
            values of the independent variable.
        """
        return self.model.func(x, *self.params)
