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
ats = 2
molno = 8
trajname = op.join(data_path,'dummy8.gsd')
cutoff = 1.1*1.1
#assuming that there are at least as many trajectories as processors
num = int(np.floor(ttotal / size))
rem = ttotal % size
traj = gsd.hoomd.open(trajname)
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
                tslist[(len(ts)+currid):(len(ts) + currid + (r-rem)+1)] = -1
            currid += num + 1
        else:
            tslist = np.arange(num * size)
    print(tslist)
    clusters = [cl.ContactClusterSnapshot(t,traj,ats) for t in tslist]
    carraylen = clusters[0].getCArrayLen()
    clusterarray = np.zeros(carraylen * len(clusters))
    cind = 0
    for cls in clusters:
        carray = cls.toArray()
        clusterarray[(cind * carraylen):(cind * carraylen + carraylen)] = carray
        cind += 1
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
    #print("Cluster array: ",clusterarray)
comm.Scatter(clusterarray,carray_local,root=0)

#for each local cluster array, turn it into a cluster, compute the 
#clusterIDs, pack the whole thing up as an array again, and send back to 
#root

for i in range(ncsnaps):
    carrayi = carray_local[carraylen * i : (carraylen * i + carraylen)]
    #print("From rank {0}, snap {1}, array{2}".format(rank,i,carrayi))
    if not np.isnan(carrayi[4]):
        clustSnap = cl.fromArray(carrayi,traj,ctype='contact')
        clustSnap.setClusterID(cutoff)
        carray_local[carraylen * i : (carraylen * i + carraylen)] = clustSnap.toArray()

comm.Barrier()
comm.Gather(carray_local,clusterarray,root=0)
if rank == 0:
    print("gathered")
#root unpacks each cluster array into a cluster, gets the cluster sizes
#and checks them against assertions to make sure everything was computed
#correctly
clustSizesActual = [[1,1,1,1,1,1,1,1],[1,2,2,1,3,3,1,3],[1,3,3,3,3,3,1,3],
                    [2,3,3,3,3,3,2,3],[5,5,5,5,3,3,5,3],[5,5,5,5,3,3,5,3],
                    [8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8]]
if rank == 0:
    ind = 0
    nind = 0
    while ind < ttotal:
        carrayi = clusterarray[carraylen * nind : (carraylen * nind + carraylen)]
        if not np.isnan(carrayi[4]):
            clustSnap = cl.fromArray(carrayi,traj,ctype='contact')
            csizes = clustSnap.idsToSizes()

            assert (csizes == clustSizesActual[ind]).all()
            ind += 1
        nind +=1

def test_system_init_MPI():
    """
    Test the initialization of the system using MPI
    """
    cldict = {'contact':1.1*1.1}
    syst = cl.SnapSystem(traj,ats,molno,cldict)
    assert syst.mpi
    assert syst.ats == ats
    assert syst.molno == molno
    assert syst.cldict == cldict

def test_system_set_CIDs_MPI():
    """
    Test setting the cluster IDs with MPI
    """
    cldict = {'contact':1.1*1.1}
    syst = cl.SnapSystem(traj,ats,molno,cldict)
    syst.get_clusters_mpi('contact')
    clustSizesActual = [[1,1,1,1,1,1,1,1],[1,2,2,1,3,3,1,3],[1,3,3,3,3,3,1,3],
                    [2,3,3,3,3,3,2,3],[5,5,5,5,3,3,5,3],[5,5,5,5,3,3,5,3],
                    [8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8]]
    if rank == 0:
        ind = 0
        nind = 0
        print(len(syst.clsnaps['contact']))
        tclustSnap = syst.clsnaps['contact'][7]
        print(tclustSnap.pos[0])
        print(tclustSnap.idsToSizes())
        while ind < ttotal:
            clustSnap = syst.clsnaps['contact'][nind]
            
            if not np.isnan(clustSnap.pos[0][0]):
                print("ind is: ",ind)
                print("nind is: ",nind)
                csizes = clustSnap.idsToSizes()
                print("csizes is: ",csizes)
                print("actual is: ",clustSizesActual[ind])
                assert (csizes == clustSizesActual[ind]).all()
                ind += 1
            nind +=1

def test_writeout():
    """
    Test the MPI writing out of cluster stuff
    """
    cldict = {'contact':1.1*1.1}
    syst = cl.SnapSystem(traj,ats,molno,cldict)
    syst.get_clusters_mpi('contact')
    clustSizesActual = [[1,1,1,1,1,1,1,1],[1,2,2,1,3,3,1,3],[1,3,3,3,3,3,1,3],
                    [2,3,3,3,3,3,2,3],[5,5,5,5,3,3,5,3],[5,5,5,5,3,3,5,3],
                    [8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8]]
    if rank == 0:
        syst.writeCIDs('contact','dummyCIDs.dat')
        syst.writeSizes('contact','dummysizes.dat')
        try:
            
            fCIDs = open('dummyCIDs.dat')
        except IOError:
            print("ERROR! cluster index file was not written.")
            return False
        try:
            fsizes = open('dummysizes.dat')
        except IOError:
            print("ERROR! cluster sizes file was not written.")
            return False
        lind = 0
        for line in fsizes:
            try:
            
                assert (clustSizesActual[lind]==[int(n) for n in line.split()])
                lind += 1
            except AssertionError:
                print("{0} is not equal to {1}.".format(clustSizesActual[lind],[int(n) for n in line.split()]))
                
                return False
        fCIDs.close()
        fsizes.close()
        return True
    
#poor man's pytest
if __name__ == "__main__":
    try:
        test_system_init_MPI()
        if rank == 0:
            print("System initialized correctly.")
    except AssertionError:
        if rank == 0:
            print("Failure of system initialization with MPI.")
        
    try:
        test_system_set_CIDs_MPI()
        if rank == 0:
            print("System correctly set cluster indices.")
    except AssertionError:
        if rank == 0:
            print("System failed at setting cluster indices.")
    
    writebool = test_writeout()
    if rank == 0:
        if writebool:
            print("System passed writing test.")
        else:
            print("ERROR! Unknown error in writing test.")