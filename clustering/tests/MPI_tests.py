# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:53:21 2017

@author: rachael

Test an MPI instantiation of finding cluster IDs, if it works, it can be folded
into the main code

Best tests: run as 
> mpiexec -n 3 python MPI_tests.py
> mpiexec -n 5 python MPI_tests.py
> mpiexec -n 8 python MPI_tests.py
"""

from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import numpy.testing as npt
import clustering as cl
import gsd.hoomd
from mpi4py import MPI

data_path = op.join(cl.__path__[0], 'data')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ttotal = 8 #test on 8 snapshots
ats = 17
molno = 8
trajname = op.join(data_path,'mols8.gsd')
cutoff = 1.1*1.1
#assuming that there are at least as many trajectories as processors
num = int(np.floor(ttotal / size))
rem = ttotal % size
traj = gsd.hoomd.open(trajname)
if rank == 0:
    
    tslist = np.zeros((num + 1) * size).astype(int)

    currid = 0
    for r in range(size):
        if r < rem:
            ts = r * (num + 1) + np.arange(num + 1)
            tslist[currid:(len(ts)+currid)] = ts
           
        else: 
            ts = r * (num + 1) - (r - rem) + np.arange(num)
            tslist[currid:(len(ts)+currid)] = ts
            tslist[(len(ts)+currid):(len(ts) + currid + (r-rem)+1)] = -1
        currid += num + 1
    print(tslist)
    clusters = [cl.ContactClusterSnapshot(t,traj,ats) for t in tslist]
    carraylen = clusters[0].getCArrayLen()
    clusterarray = np.zeros(carraylen * len(clusters))
    for cls in clusters:
        carray = cls.toArray()
        clusterarray[0:carraylen] = carray
else:
    tCSnap = cl.ContactClusterSnapshot(0,traj,ats)
    carraylen = tCSnap.getCArrayLen()
    clusterarray = None
if rem == 0:
    ncsnaps = num
else:
    ncsnaps = num + 1
carray_local = np.zeros(ncsnaps * carraylen)
print("From rank {0}, length of carray_local: {1}".format(rank,len(carray_local)))
if rank == 0:
    print("From rank 0, length of clusterarray: ",len(clusterarray))
    print("From rank 0, length of clusterarray / # of ranks",len(clusterarray)/size)
comm.Scatter(clusterarray,carray_local,root=0)

#for each local cluster array, turn it into a cluster, compute the 
#clusterIDs, pack the whole thing up as an array again, and send back to 
#root



#root unpacks each cluster array into a cluster, gets the cluster sizes
#and checks them against assertions to make sure everything was computed'
#correctly

'''
if rank < rem:
    ts = rank * (num + 1) + np.arange(num + 1)
else:
    ts = rank * (num + 1) - (rank - rem) + np.arange(num)

local_clusterIDs = np.zeros(len(ts)*ats)


for t in ts:
    snap = traj[int(t)]
    pos = snap.particles.position
    (nclust,clusterIDs) = cl.getContactClusterID(pos,cutoff)

if rank == 0:
    clusterIDGather = np.zeros(ttotal*ats)
    print("Size of Gather Array: ",len(clusterIDGather))
else:
    clusterIDGather = None
sendcounts = np.arange(size)
displacements = np.arange(size)
for i in range(size):
    if i < rem:
        sendcounts[i] = ats * ( rank * (num+1) + num + 1)
        displacements[i] = rank * (num + 1)
    else:
        sendcounts[i] = ats * (rank * (num+1) - (rank - rem) + num) 
        displacements[i] = rank * (num + 1) - (rank - rem)

print("From rank {0}, length of cIDs: {1}\n".format(rank,len(clusterIDs)))
print("From rank {0}, sendcounts: {1}\n".format(rank,sendcounts))
print("From rank {0}, displacements: {1}\n".format(rank,displacements))
    
#comm.Gatherv(clusterIDs,[clusterIDGather,sendcounts,displacements,MPI.DOUBLE])
    '''