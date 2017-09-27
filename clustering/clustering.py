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

__all__ = ["ClusterSnapshot", "ContactClusterSnapshot","Fit", "transform_data","conOptDistance"]


# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("10.1167/13.9.30"),
         description="Simple data analysis for clustering application",
         tags=["data-analysis","clustering"],
         path='clustering')


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
    return np.min(cdist(xa,ya,metric='euclidean'))
            


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
        self.binds = np.argsort(snapshot.particles.body)
        self.pos = snapshot.particles.position[self.binds]
        sz = np.shape(self.pos)
        if sz[0] % ats != 0:
            raise RuntimeError("Number of particles not divisible by number \
                                of beads per molecules.")
        self.clusterIDs = np.zeros(sz[0]/ats)
        


class ContactClusterSnapshot(ClusterSnapshot):
    """Class for tracking the location of contact clusters at each time step"""
    
    def __init__(self, t, trajectory, ats, cutoff):
        """ Initialize a ClusterSnapshot object.

        Parameters
        ----------
        t: timestep

        snapshot: a gsd.hoomd snapshot at some time t
        
        ats: the number of beads in a single molecule
        
        cutoff: cutoff for cluster distance metric (nm)
        """
        snapshot = trajectory[t]
        self.timestep = t
        self.binds = np.argsort(snapshot.particles.body)
        self.pos = snapshot.particles.position[self.binds]
        sz = np.shape(self.pos)
        if sz[0] % ats != 0:
            raise RuntimeError("Number of particles not divisible by number \
                                of beads per molecules.")
        self.pos = np.reshape(self.pos,[sz[0] / ats , 3 * ats])
        (cc,cclabels) = self.getClusterID(self.pos,cutoff)
        self.nclusts = cc
        self.clusterIDs = cclabels
        
    def getClusterID(self,positions,cutoff):
        """
        Find the ID of which cluster each molecule is in

        Parameters
        ----------
        positions: numpy array of locations of each molecule in 3D space

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
