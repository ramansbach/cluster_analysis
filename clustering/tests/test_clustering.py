from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import numpy.testing as npt

import gsd.hoomd
import sys
#import clustering as cl
from context import clustering as cl
from context import smoluchowski as smol
#import imp
#cl = imp.load_source('cl','/home/rachael/Analysis_and_run_code/analysis/cluster_analysis/clustering/clustering.py')
#data_path = op.join(cl.__path__[0], 'data')
data_path = '/home/rachael/Analysis_and_run_code/analysis/cluster_analysis/clustering/data'
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
    
def test_OpticalClusterSnapshot_init():
    """
    Testing the instantiation of an Optical Cluster Snapshot
    """
    fname = 'mols8.gsd'
    traj = gsd.hoomd.open(op.join(data_path, fname))
    
    ats = 12
    molno = 8
    cutoff= 0.35*0.35
    t = 97
    atype = u'LS'
    #compairs = np.array([[0,6],[1,7],[2,8],[3,9],[4,10],[5,11]])
    clustSnap = cl.OpticalClusterSnapshot(t,traj,ats,molno,
                                          atype=atype)
    clustSnap.setClusterID(cutoff)
    assert clustSnap.timestep == t
    sz = np.shape(clustSnap.pos)
    assert sz[0] == molno
    assert sz[1] == ats * 3

def test_AlignedClusterSnapshot_init():
    """
    Testing the instantiation of an Aligned Cluster Snapshot
    """
    fname = 'mols8.gsd'
    traj = gsd.hoomd.open(op.join(data_path, fname))
    
    ats = 6
    molno = 8
    cutoff= 0.35*0.35
    t = 97
    atype = u'LS'
    compairs = np.array([[0,6],[1,7],[2,8],[3,9],[4,10],[5,11]])
    clustSnap = cl.AlignedClusterSnapshot(t,traj,ats,molno,compairs=compairs,
                                          atype=atype)
    clustSnap.setClusterID(cutoff)
    assert clustSnap.timestep == t
    sz = np.shape(clustSnap.pos)
    assert sz[0] == molno
    assert sz[1] == 3 * np.shape(compairs)[0]      
    
    
def test_getComsAligned():
    """
    Check getComs alone for aligned snapshots
    """
    t = 910

    inputfname = "/home/rachael/coarsegraining/hoomd/patchytest/analysis_testing/mols2.gsd"
    traj = gsd.hoomd.open(inputfname)
    ats = 6
    molno = 2
    
    atype = u'LS'
    compairs = np.array([[0,6],[1,7],[2,8],[3,9],[4,10],[5,11]])
    clustSnap = cl.AlignedClusterSnapshot(t,traj,ats,molno,compairs=compairs,
                                          atype=atype)
    posact = np.array([[-0.51880336,1.19018042,-0.60373259],
                       [-0.56411886,1.88792956,-0.96676737],
                       [-0.94110215,1.29164505,-0.35600558],
                       [-0.98641765,1.98939419,-0.71904039],
                       [-0.09650461,1.08871579,-0.85145962],
                       [-0.1418201,1.78646493,-1.21449447],
                       [-0.69864309,0.95386416,-0.50236279],
                       [-0.63304943,0.32048941,-0.03842793],
                       [-0.29119688,0.81126034,-0.75465578],
                       [-0.22560325,0.17788559,-0.29072094],
                       [-1.10608935,1.09646797,-0.25006977],
                       [-1.04049563,0.46309322,0.21386507]])
    posact = np.reshape(posact,[2,18])
    npt.assert_array_almost_equal(clustSnap.pos,posact)

       
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

def test_conOptDistC():
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
    apos1 = np.array([-1.,0.,0.,-1.,-0.5,0.,
                      -1.,-1.,0.,0.,-1.,0.,0.,-0.5,0.,0.,0.,0.])
    apos2 = np.array([1.5,0.,0.,1.5,-0.5,0.,
                      1.5,-1.,0.,0.5,-1.,0.,0.5,-0.5,0.,0.5,0.,0.])
    
    r1 = cl.conOptDistanceC(molA1,molB1)
    r2 = cl.conOptDistanceC(molA2,molB2)
    r3 = cl.conOptDistanceC(apos1,apos2)
    npt.assert_almost_equal(r1,4.0,decimal=10)
    npt.assert_almost_equal(r2,1.0,decimal=10) 
    npt.assert_almost_equal(r3,0.25,decimal=10)
    
def test_pyCEquality():
    """
    Trying to find where / why conOptDist and conOptDistC don't
    return the same thing
    """
    fname = 'mols8.gsd'
    traj = gsd.hoomd.open(op.join(data_path, fname))
    
    ats = 17
    molno = 8
    cutoff= 1.1*1.1
    t = 97
    clustSnap = cl.ContactClusterSnapshot(t,traj,ats,molno)
    for i in range(molno):
        for j in range(molno):
            mol1 = clustSnap.pos[i,:]
            mol2 = clustSnap.pos[j,:]
            r = cl.conOptDistance(mol1,mol2) 
            rc = cl.conOptDistanceC(mol1,mol2)
            ar = cl.alignedDistance(mol1,mol2)
            arc = cl.alignedDistanceC(mol1,mol2)
            npt.assert_almost_equal(r,rc,5)
            npt.assert_almost_equal(ar,arc,5)
            
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
    nclustsActual = [8,5,4,3,2,2,1,1]
    for t in ts:
        clustSnap = cl.ContactClusterSnapshot(t,traj,ats,molno)
        clustSnap.setClusterID(cutoff)
        clustSize = clustSnap.idsToSizes()
        assert (clustSize == clustSizesActual[t]).all()
        assert clustSnap.nclusts == nclustsActual[t]
        
def test_dummyfull():
    """
    Makes sure that all three cluster types return what is expected,
    employing the hand-crafted 6 snapshot system of dummy molecules that
    are similar to the 17-bead HOOMD rigid bodies
    """
    fname = 'dummyfull4.gsd'
    traj = gsd.hoomd.open(op.join(data_path,fname))
    atsCC = 17
    atsOC = 12
    atsAC = 6
    molno = 4
    contactcut = 1.1*1.1
    opticalcut = 0.35*0.35
    alignedcut = 0.35*0.35
    ts = range(6)
    nclustsActualCC = [4,2,1,1,1,1]
    nclustsActualOC = [4,3,2,2,1,1]
    nclustsActualAC = [4,4,3,2,2,1]
    clustSizesActualCC = [[1,1,1,1],[2,2,2,2],[4,4,4,4],
                          [4,4,4,4],[4,4,4,4],[4,4,4,4]]
    clustSizesActualOC = [[1,1,1,1],[1,1,2,2],[2,2,2,2],
                          [2,2,2,2],[4,4,4,4],[4,4,4,4]]
    clustSizesActualAC = [[1,1,1,1],[1,1,1,1],[2,2,1,1],
                          [2,2,2,2],[2,2,2,2],[4,4,4,4]]
    for t in ts:
        print(t)
        clustSnapCC = cl.ContactClusterSnapshot(t,traj,atsCC,molno)
        clustSnapCC.setClusterID(contactcut)
        clustSizeCC = clustSnapCC.idsToSizes()
        assert (clustSizeCC == clustSizesActualCC[t]).all()
        assert clustSnapCC.nclusts == nclustsActualCC[t]
        
        atype = 'AB'
        compairs = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]
        clustSnapOC = cl.OpticalClusterSnapshot(t,traj,atsOC,molno,
                                                atype=atype)
        clustSnapOC.setClusterID(opticalcut)
        clustSizeOC = clustSnapOC.idsToSizes()
        assert (clustSizeOC == clustSizesActualOC[t]).all()
        assert clustSnapOC.nclusts == nclustsActualOC[t]
        
        clustSnapAC = cl.AlignedClusterSnapshot(t,traj,atsAC,molno,
                                                compairs=compairs,atype=atype)
        clustSnapAC.setClusterID(alignedcut)
        clustSizeAC = clustSnapAC.idsToSizes()
        assert (clustSizeAC == clustSizesActualAC[t]).all()
        assert clustSnapAC.nclusts == nclustsActualAC[t]

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
        mu2 = smol.massAvSize(clustSize)
        
        npt.assert_almost_equal(mu2,mu2s[t],10)
        
def test_get_clusters_serial():
    """
    test the serial implementation of setting all the clusters correctly
    """
    fname = 'dummy8.gsd'
    traj = gsd.hoomd.open(op.join(data_path,fname))
    ats = {'contact':2}
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
def test_get_clusters_serial_full():
    """
    test the serial implementation of setting all the cluster correctly
    for all cluster types
    """ 
    fname = 'dummyfull4.gsd'
    traj = gsd.hoomd.open(op.join(data_path,fname))
    ats = {'contact':17,'optical':12,'aligned':6}
    molno = 4
    cldict = {'contact':1.1*1.1,'optical':0.35*0.35,'aligned':0.35*0.35}
    compairs = np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]])
    atype = 'AB'
    syst = cl.SnapSystem(traj,ats,molno,cldict,compairs=compairs,atype=atype)
    syst.get_clusters_serial('contact')
    syst.get_clusters_serial('optical')
    syst.get_clusters_serial('aligned')
    cSizesActual = dict()
    cSizesActual['contact'] = [[1,1,1,1],[2,2,2,2],[4,4,4,4],
                          [4,4,4,4],[4,4,4,4],[4,4,4,4]]
    cSizesActual['optical']= [[1,1,1,1],[1,1,2,2],[2,2,2,2],
                          [2,2,2,2],[4,4,4,4],[4,4,4,4]]
    cSizesActual['aligned'] = [[1,1,1,1],[1,1,1,1],[2,2,1,1],
                          [2,2,2,2],[2,2,2,2],[4,4,4,4]]
    for key in ats.keys():
        cid = 0
        clsnaps = syst.clsnaps[key]
        for clsnap in clsnaps:
            csizes = clsnap.idsToSizes()
            
            assert (cSizesActual[key][cid] == csizes).all()
            
            cid +=1
    
def test_mu2vtime():
    """
    test the (serial) version of getting the mass-averaged cluster size v time
    """
    fname = 'mols8.gsd'
    traj = gsd.hoomd.open(op.join(data_path, fname))
    
    ats = {'contact':17}
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
    
    ats = {'contact':17}
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
    
    ats = {'contact':17}
    cutoff= 1.1*1.1
    molno = 8
    cldict = {'contact':cutoff}
    syst = cl.SnapSystem(traj,ats,molno,cldict)
    syst.get_clusters_serial('contact')
    syst.writeSizes('contact',op.join(data_path,'mols8sizes.dat'))
    
def test_alignedDistance():
    """
    test that the aligned distance metric works correctly on several cases
    """
    apos1 = np.array([-1.,0.,0.,-1.,-0.5,0.,
                      -1.,-1.,0.,0.,-1.,0.,0.,-0.5,0.,0.,0.,0.])
    apos2 = np.array([1.5,0.,0.,1.5,-0.5,0.,
                      1.5,-1.,0.,0.5,-1.,0.,0.5,-0.5,0.,0.5,0.,0.])
    d0 = cl.alignedDistance(apos1,apos2)
    npt.assert_almost_equal(d0,0.25,10)
    
    apos1 = np.array([-1.,0.,0.,-1.25,-0.5,0.,
                      -1.5,-1.,0.,-0.5,-1.,0.,
                      -0.25,-0.5,0.,0.,0.,0.])
    apos2 = np.array([0.5,0.,0.,1.5,0.,0.,
                      0.75,-0.5,0.,1.75,-0.5,0.,
                      1.,-1.,0.,2.,-1.,0.])
    d1 = cl.alignedDistance(apos1,apos2)
    npt.assert_almost_equal(d1,1.5 * 1.5, 10)
    
def test_alignedDistanceC():
    """
    test that the aligned distance metric with a C kernel
    works correctly on several cases
    """
    apos1 = np.array([-1.,0.,0.,-1.,-0.5,0.,
                      -1.,-1.,0.,0.,-1.,0.,0.,-0.5,0.,0.,0.,0.])
    apos2 = np.array([1.5,0.,0.,1.5,-0.5,0.,
                      1.5,-1.,0.,0.5,-1.,0.,0.5,-0.5,0.,0.5,0.,0.])
    d0 = cl.alignedDistanceC(apos1,apos2)
    npt.assert_almost_equal(d0,0.25,10)
    
    apos1 = np.array([-1.,0.,0.,-1.25,-0.5,0.,
                      -1.5,-1.,0.,-0.5,-1.,0.,
                      -0.25,-0.5,0.,0.,0.,0.])
    apos2 = np.array([0.5,0.,0.,1.5,0.,0.,
                      0.75,-0.5,0.,1.75,-0.5,0.,
                      1.,-1.,0.,2.,-1.,0.])
    d1 = cl.alignedDistanceC(apos1,apos2)
    npt.assert_almost_equal(d1,1.5 * 1.5, 10)
    
if __name__ == "__main__":
    test_get_clusters_serial_full()
    '''
    test_dummyfull()
      
    test_getComs()
    
    test_OpticalClusterSnapshot_init()
    
    test_pyCEquality()
    
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
    
    test_alignedDistance()
    '''