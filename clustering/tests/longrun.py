# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:34:03 2017

@author: rachael

test long run with mpi (run with mpiexec)
"""
from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import numpy.testing as npt
import clustering as cl
import gsd.hoomd
import sys

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data_path = op.join(cl.__path__[0], 'data')

ats = 17
molno = 10000
trajname = op.join(data_path,'mols10000_740-0-0_run1.gsd')
cutoff = 1.1*1.1
traj = gsd.hoomd.open(trajname)
cldict = {'contact':1.1*1.1}
syst = cl.SnapSystem(traj,ats,molno,cldict,12)
syst.get_clusters_mpi('contact',12)
if rank == 0:
    print("writing clusterIDs")
    syst.writeCIDs('contact',op.join(data_path,'mols10KcIDs.dat'))
    print("writing cluster sizes")
    syst.writeSizes('contact',op.join(data_path,'mols10KcSizes.dat'))