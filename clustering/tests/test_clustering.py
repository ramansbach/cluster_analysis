from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import numpy.testing as npt
import clustering as cl
import gsd.hoomd
import sys

data_path = op.join(cl.__path__[0], 'data')

def test_MPI():
    """
    Dummy function reminder to run the MPI tests as well
    """
    print("Please run mpiexec -n 5 python MPI_tests.py. \
    Currently this hasn't yet been automated.")

def test_ClusterSnapshot_init():
    """
    Testing the instantiation of a Cluster Snapshot
    """
    fname = 'mols2.gsd'
    traj = gsd.hoomd.open(op.join(data_path, fname))
    
    ats = 17
    clustSnap = cl.ClusterSnapshot(0,traj,ats)
    assert clustSnap.timestep == 0
    sz = np.shape(traj[0].particles.position)
    assert len(clustSnap.clusterIDs) == sz[0] / ats   
 
def test_ContactClusterSnapshot_init():
    """
    Testing the instantiation of a Cluster Snapshot
    """
    fname = 'mols8.gsd'
    traj = gsd.hoomd.open(op.join(data_path, fname))
    
    ats = 17
    molno = 8
    cutoff= 1.1*1.1
    t = 97
    clustSnap = cl.ContactClusterSnapshot(t,traj,ats,molno)
    clustSnap.setClusterID(cutoff)
    assert clustSnap.timestep == t
    sz = np.shape(traj[t].particles.position)
    assert len(clustSnap.clusterIDs) == sz[0] / ats   
    assert clustSnap.nclusts == 2
    
       
def test_conOptDist():
    """
    Testing the contact and optical distance metric
    
    Two test cases, both essentially in 2D
    (i) two parallel molecules (so you get the same distance as minimum), r = 1.0
    (ii) two angled molecules, r = 2.0
    """
    molA1 = np.array([0.,0.,0.,1.,1.,0.,2.,2.,0.,3.,3.,0.,4.,4.,0.])
    molB1 = np.array([0.,-2.,0.,1.,-3.,0.,2.,-4.,0.,3.,-5.,0.,4.,-6.,0.])
    molA2 = np.array([0.,1.,0.,1.,1.,0.,2.,1.,0.,3.,1.,0.,4.,1.,0.])
    molB2 = np.array([0.,0.,0.,1.,0.,0.,2.,0.,0.,3.,0.,0.,4.,0.,0.])
    
    r1 = cl.conOptDistance(molA1,molB1)
    r2 = cl.conOptDistance(molA2,molB2)
    npt.assert_almost_equal(r1,4.0,decimal=10)
    npt.assert_almost_equal(r2,1.0,decimal=10)
    
def test_clusterPackAndUnpack():
    """
    Test to and from array stuff which is basically my own version of 
    'pickling' in a numpy array for use with mpi4py.
    """
    fname = 'mols8.gsd'
    traj = gsd.hoomd.open(op.join(data_path, fname))
    
    ats = 17
    
    cutoff= 1.1*1.1
    molno = 8
    t = 97
    clustSnap = cl.ContactClusterSnapshot(t,traj,ats,molno)
    clustSnap.setClusterID(cutoff)
    carray = clustSnap.toArray()
    clustSnap2 = cl.ContactClusterSnapshot(0,carray,ats,molno)
    assert clustSnap.timestep == clustSnap2.timestep
    assert clustSnap.ats == clustSnap2.ats
    assert clustSnap.nclusts == clustSnap2.nclusts
    assert (clustSnap.pos == clustSnap2.pos).all()
    assert (clustSnap.clusterIDs == clustSnap2.clusterIDs).all()
 

def test_dummy8():
    """
    Makes sure that contact clusters return what is expected, employing
    the hand-crafted 8 snapshot system of 2 atoms in each molecule
    """
    fname = 'dummy8.gsd'
    traj = gsd.hoomd.open(op.join(data_path,fname))
    ats = 2
    molno = 8
    cutoff = 1.1*1.1
    ts = range(8)
    clustSizesActual = [[1,1,1,1,1,1,1,1],[1,2,2,1,3,3,1,3],[1,3,3,3,3,3,1,3],
                        [2,3,3,3,3,3,2,3],[5,5,5,5,3,3,5,3],[5,5,5,5,3,3,5,3],
                        [8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8]]
    for t in ts:
        clustSnap = cl.ContactClusterSnapshot(t,traj,ats,molno)
        clustSnap.setClusterID(cutoff)
        clustSize = clustSnap.idsToSizes()
        assert (clustSize == clustSizesActual[t]).all()
    

def test_mu2():
    """
    test mass-averaged cluster size instantiation
    """
    fname = 'dummy8.gsd'
    traj = gsd.hoomd.open(op.join(data_path,fname))
    ats = 2
    molno = 8
    cutoff = 1.1*1.1
    ts = range(8)
    mu2s = np.array([1,2.375,2.8,62/22,152/34,152/34,8,8])
    for t in ts[1:len(ts)]:
        clustSnap = cl.ContactClusterSnapshot(t,traj,ats,molno)
        clustSnap.setClusterID(cutoff)
        clustSize = clustSnap.idsToSizes()
        mu2 = clustSnap.massAvSize(clustSize)
        
        npt.assert_almost_equal(mu2,mu2s[t],10)
        
def test_get_clusters_serial():
    """
    test the serial implementation of setting all the clusters correctly
    """
    fname = 'dummy8.gsd'
    traj = gsd.hoomd.open(op.join(data_path,fname))
    ats = 2
    cutoff = 1.1*1.1
    molno = 8
    cldict = {'contact':cutoff}
    syst = cl.SnapSystem(traj,ats,molno,cldict)
    syst.get_clusters_serial('contact')
    clustSizesActual = [[1,1,1,1,1,1,1,1],[1,2,2,1,3,3,1,3],[1,3,3,3,3,3,1,3],
                    [2,3,3,3,3,3,2,3],[5,5,5,5,3,3,5,3],[5,5,5,5,3,3,5,3],
                    [8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8]]
    cid = 0
    clsnaps = syst.clsnaps['contact']
    for clsnap in clsnaps:
        csizes = clsnap.idsToSizes()
        assert (clustSizesActual[cid] == csizes).all()
        cid+=1

def test_mu2vtime():
    """
    test the (serial) version of getting the mass-averaged cluster size v time
    """
    fname = 'mols8.gsd'
    traj = gsd.hoomd.open(op.join(data_path, fname))
    
    ats = 17
    cutoff= 1.1*1.1
    molno = 8
    cldict = {'contact':cutoff}
    syst = cl.SnapSystem(traj,ats,molno,cldict)
    syst.get_clusters_serial('contact')
    mu2vtime = syst.getMassAvVsTime('contact')
    assert np.shape(mu2vtime)[1] == len(traj)
    assert not (np.isnan(mu2vtime)).all()

def test_writeCID():
    """
    test writing out the clusterIDs
    """
    fname = 'mols8.gsd'
    traj = gsd.hoomd.open(op.join(data_path, fname))
    
    ats = 17
    cutoff= 1.1*1.1
    molno = 8
    cldict = {'contact':cutoff}
    syst = cl.SnapSystem(traj,ats,molno,cldict)
    syst.get_clusters_serial('contact')
    syst.writeCIDs('contact',op.join(data_path,'mols8cIDs.dat'))
    
def test_writeSizes():
    """
    test writing out the clusterIDs
    """
    fname = 'mols8.gsd'
    traj = gsd.hoomd.open(op.join(data_path, fname))
    
    ats = 17
    cutoff= 1.1*1.1
    molno = 8
    cldict = {'contact':cutoff}
    syst = cl.SnapSystem(traj,ats,molno,cldict)
    syst.get_clusters_serial('contact')
    syst.writeSizes('contact',op.join(data_path,'mols8sizes.dat'))
    
if __name__ == "__main__":
    test_ClusterSnapshot_init()
    test_MPI()
    test_get_clusters_serial()
    test_writeCID()
    test_ContactClusterSnapshot_init()
    test_conOptDist()
    test_clusterPackAndUnpack()
    test_dummy8()
    test_mu2()
    test_get_clusters_serial()
    test_mu2vtime()
    test_writeCID()
    test_writeSizes()