from __future__ import absolute_import, division, print_function
import numpy as np
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

__all__ = ['fixPBC']

# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
'''
due.cite(Doi("10.1167/13.9.30"),
         description="Simple data analysis for clustering application",
         tags=["data-analysis","clustering"],
         path='clustering')
'''

def fixPBC(peps,box,ats,cutoff):
	#return positions fixed across PBCs for calculation of structural metrics like Rh and Rg
	#create the list of fixed positions
	fixedXYZ = peps.copy()
	potInds = range(1,len(peps)/(ats*3))
	#the first ats*3 coordinates are the coordinates of the first atom
	fixedXYZ[0:3*ats] = fixCoords(fixedXYZ[0:3*ats].copy(),fixedXYZ[0:3].copy(),box)
	correctInds = [0]
	while len(correctInds) > 0:
		atom = correctInds.pop()
		neighs = getNeigh(atom,cutoff,peps,potInds,ats)
		for n in neighs:

			potInds.remove(n)
			correctInds.append(n)
			fixedXYZ[3*ats*n:3*ats*(n+1)] = fixCoords(fixedXYZ[3*ats*n:3*ats*(n+1)].copy(),fixedXYZ[3*atom*ats:3*atom*ats+3].copy(),box)
	return fixedXYZ

def fixCoords(pos,posinit,box):
	#fix all coords based on the initial coordinate and the periodic boundary conditions
	for i in range(len(pos)/3):
		dr = pos[3*i:3*i+3] - posinit
		dr = dr - box*np.round(dr/box)
		pos[3*i:3*i+3] = dr + posinit
	return pos
 
