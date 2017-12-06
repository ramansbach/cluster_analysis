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
from mpi4py import MPI
from cdistances import conOptDistanceCython,alignDistancesCython

__all__ = []

# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
'''
due.cite(Doi("10.1167/13.9.30"),
         description="Simple data analysis for clustering application",
         tags=["data-analysis","clustering"],
         path='clustering')
'''