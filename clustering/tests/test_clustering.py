from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import numpy.testing as npt
import pdb
import gsd.hoomd
import sys
import clustering as cl
import random
import scipy
import pytest
#from context import clustering as cl
#from context import smoluchowski as smol
from cdistances import conOptDistanceCython,alignDistancesCython
from scipy.sparse import csr_matrix,lil_matrix,coo_matrix
#import imp
#cl = imp.load_source('cl','/home/rachael/Analysis_and_run_code/analysis/cluster_analysis/clustering/clustering.py')
data_path = op.join(cl.__path__[0], 'data')
#data_path = '/home/rachael/Analysis_and_run_code/analysis/cluster_analysis/clustering/data'\
'''
def test_coms_missing_arom():
    """
    test the COM for the one molecule with the missing aromatic
    """
    fname = 'missingarom1.gsd'
    traj = gsd.hoomd.open(op.join(data_path,fname))
    ats = 6
    molno = 1
    compairs = np.array([[1,7],[0,6],[2,8],[3,9],[4,10],[5,11]])
    clustSnap = cl.AlignedClusterSnapshot(0,traj,ats,molno,compairs=compairs)
    pos = clustSnap.pos
    actPos = np.array([[0.,-0.393923,0.0,0.,0.393923,0.,0.5,0.393923,0.,0.5,
                        -0.393923,0.0,-0.5,0.393923,0.,-0.5,-0.393923,0.0]])
    #pdb.set_trace()
    npt.assert_array_almost_equal(pos,actPos,decimal=5)
''' 
def test_fixed_arom():
    """
    test making a fixed version of the "missing" aromatic
    """
    fname = op.join(data_path,'missingarom1.gsd')
    outname = op.join(data_path,'fixedmissingarom1.gsd')
    idMiss = 10
    idPartner = 11
    idNotMiss = 4
    idNotPartner = 5
    molno = 1
    ats = 17
    ts = np.array([0])
    cl.fixMisplacedArom(fname,outname,idMiss,idPartner,idNotMiss,idNotPartner
                    ,molno,ats,ts)
    
def test_coms_missing_arom2():
    """
    test the COM function 
    """
    fname = 'mols2.gsd'
    #pdb.set_trace()
    traj = gsd.hoomd.open(op.join(data_path,fname))
    tstep = 0
    box = traj[0].configuration.box
    ats = 6
    molno = 2
    compairs = np.array([[1,7],[0,6],[2,8],[3,9],[4,10],[5,11]])
    clustSnap = cl.AlignedClusterSnapshot(tstep,traj,ats,molno,
                                          compairs=compairs)
    clustSnap.writeCOMsGSD(op.join(data_path,'mol2acoms.gsd'))
    
def test_coms_missing_arom8():
    """
    test the new COM function
    """
    fname = 'mols8.gsd'
    traj = gsd.hoomd.open(op.join(data_path,fname))
    for tstep in range(1000):
        box = traj[0].configuration.box
        ats = 6
        molno = 8
        compairs = np.array([[1,7],[0,6],[2,8],[3,9],[4,10],[5,11]])
        clustSnap = cl.AlignedClusterSnapshot(tstep,traj,ats,molno,
                                          compairs=compairs)
        clustSnap.writeCOMsGSD(op.join(data_path,'mol8acoms.gsd'))
def test_MPI():
    """
    Dummy function reminder to run the MPI tests as well
    """
    raise NotImplementedError("Please run mpiexec -n 5 python MPI_tests.py. \
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

    inputfname = "mols2.gsd"
    traj = gsd.hoomd.open(op.join(data_path,inputfname))
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
    
    r1 = conOptDistanceCython(molA1,molB1)
    r2 = conOptDistanceCython(molA2,molB2)
    r3 = conOptDistanceCython(apos1,apos2)
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
            mol1 = clustSnap.pos[i,:].astype(np.double)
            mol2 = clustSnap.pos[j,:].astype(np.double)
           
            r = cl.conOptDistance(mol1,mol2) 
            rc = conOptDistanceCython(mol1,mol2)
            ar = cl.alignedDistance(mol1,mol2)
            arc = alignDistancesCython(mol1,mol2)
 
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
    mu2s = np.array([1,2,5/2,11/4,17/4,17/4,8,8])
    for t in ts:
        clustSnap = cl.ContactClusterSnapshot(t,traj,ats,molno)
        clustSnap.setClusterID(cutoff)
        clustSize = clustSnap.idsToSizes()
        mu2 = cl.massAvSize(clustSize)
        
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
    box = traj[0].configuration.box
    syst.get_clusters_serial('contact',box)
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
    box = traj[0].configuration.box
    ats = {'contact':17,'optical':12,'aligned':6}
    molno = 4
    cldict = {'contact':1.1*1.1,'optical':0.35*0.35,'aligned':0.35*0.35}
    compairs = np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]])
    atype = 'AB'
    syst = cl.SnapSystem(traj,ats,molno,cldict,compairs=compairs,atype=atype)
    syst.get_clusters_serial('contact',box)
    syst.get_clusters_serial('optical',box)
    syst.get_clusters_serial('aligned',box)
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
    box = traj[0].configuration.box
    syst = cl.SnapSystem(traj,ats,molno,cldict)
    syst.get_clusters_serial('contact',box)
    mu2vtime = syst.getMassAvVsTime('contact')
    assert np.shape(mu2vtime)[1] == len(traj)
    assert not (np.isnan(mu2vtime)).all()

def test_writeCID():
    """
    test writing out the clusterIDs
    """
    fname = 'mols8.gsd'
    traj = gsd.hoomd.open(op.join(data_path, fname))
    box = traj[0].configuration.box
    ats = {'contact':17}
    cutoff= 1.1*1.1
    molno = 8
    cldict = {'contact':cutoff}
    syst = cl.SnapSystem(traj,ats,molno,cldict)
    syst.get_clusters_serial('contact',box)
    syst.writeCIDs('contact',op.join(data_path,'mols8cIDs.dat'))
    
def test_readCID():
    """
    test reading the CID from a file
    """
    fname = 'mols8.gsd'
    traj = gsd.hoomd.open(op.join(data_path, fname))
    box = traj[0].configuration.box
    ats = {'contact':17}
    cutoff= 1.1*1.1
    molno = 8
    cldict = {'contact':cutoff}
    syst = cl.SnapSystem(traj,ats,molno,cldict)
    syst.get_clusters_serial('contact',box)
    syst.writeCIDs('contact',op.join(data_path,'mols8cIDs.dat'))
    
    systwrit = cl.SnapSystem(traj,ats,molno,cldict)
    for i in range(len(systwrit.clsnaps['contact'])):
        snapf = systwrit.clsnaps['contact'][i]
        snapf.setClusterIDFromFile(op.join(data_path,'mols8cIDs.dat'))
        snap = syst.clsnaps['contact'][i]
       
        assert (snapf.clusterIDs == snap.clusterIDs).all()

def test_fixPBC():
    """
    test fixing clusters across PBCs
    """
    fname = 'dummy2PBC.gsd'
    cutoff = 0.6
    traj = gsd.hoomd.open(op.join(data_path, fname))
    clustSnap = cl.ContactClusterSnapshot(0,traj,3,2)
    clustSnap.setClusterID(cutoff)
    fixedXYZ = clustSnap.fixPBC(0,cutoff,
                     writegsd=op.join(data_path, 'dummy2PBCunwrapped.gsd'))
    npt.assert_array_almost_equal(fixedXYZ,np.array([[0.,0.,4.75,0.,0.,5.25,
                                                      0.,0.,5.75],
                                                      [-0.5,0.,5.25,-0.5,0.,
                                                       5.75,-0.5,0.,6.25]]))
                                                       
def test_getLengthDistribution():
    """
    test getting the length distribution from a trajectory
    """
    fname = 'dummyfull4.gsd'     
    traj = gsd.hoomd.open(op.join(data_path,fname))
    box = traj[0].configuration.box
    ats = {'contact':17,'optical':12,'aligned':6}
    molno = 4
    cldict = {'contact':1.1*1.1,'optical':0.35*0.35,'aligned':0.35*0.35}
    compairs = np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]])
    atype = 'AB'
    syst = cl.SnapSystem(traj,ats,molno,cldict,compairs=compairs,atype=atype)
    syst.get_clusters_serial('contact',box)
    syst.get_clusters_serial('optical',box)
    syst.get_clusters_serial('aligned',box)
    box = traj[0].configuration.box[0:3]
    ldistribt = syst.getLengthDistribution('contact',cldict['contact'],box,
                               conOptDistanceCython)
    npt.assert_array_almost_equal(ldistribt,np.array([[0.,0.,0.,0.],
                                                      [2.92,2.92,1.22,1.22],
                                                      [4.92,4.92,4.92,4.92],
                                                      [4.12,4.12,4.12,4.12],
                                                      [3.08,3.08,3.08,3.08],
                                                      [1.93,1.93,1.93,1.93]]),
                                                      decimal=2)

def test_clusters_and_length_serial():
    """
    test reusing the BallTree from cluster computation for the length
    distribution computation
    """
    fname = 'mols8.gsd'
    traj = gsd.hoomd.open(op.join(data_path, fname))
    snap = traj[0]
    box = snap.configuration.box[0:3]
    ats = {'contact':17,'optical':12,'aligned':6}
    cldict = {'contact':1.1*1.1,'optical':0.35*0.35,'aligned':0.35*0.35}
    molno = 8

    syst = cl.SnapSystem(traj,ats,molno,cldict)
    syst.get_clusters_serial('contact',box,
                             lcompute=op.join(data_path,'mols8CLdistrib.dat'))
    syst.get_clusters_serial('optical',box,
                             lcompute=op.join(data_path,'mols8OLdistrib.dat'))
    syst.get_clusters_serial('aligned',box,
                             lcompute=op.join(data_path,'mols8ALdistrib.dat'))
    
    filecidbase = op.join(data_path,'mols8CIDs_')
    fileszbase = op.join(data_path,'mols8sizes_')
    syst.writeCIDs('contact',filecidbase+'C.dat')    
    syst.writeSizes('contact',fileszbase+'C.dat')    
    syst.writeCIDs('optical',filecidbase+'O.dat')    
    syst.writeSizes('optical',fileszbase+'O.dat')    
    syst.writeCIDs('aligned',filecidbase+'A.dat')    
    syst.writeSizes('aligned',fileszbase+'A.dat')   
    
    lC = open(op.join(data_path,'mols8CLdistrib.dat'))
    lO = open(op.join(data_path,'mols8OLdistrib.dat'))
    lA = open(op.join(data_path,'mols8ALdistrib.dat'))
    linesC = lC.readlines()
    linesO = lO.readlines()
    linesA = lA.readlines()
    lC.close()
    lO.close()
    lA.close()

    
    flinesC = np.array([float(f) for f in linesC[len(linesC)-1].split()])
    flinesO = np.array([float(f) for f in linesO[len(linesO)-1].split()])
    flinesA = np.array([float(f) for f in linesA[len(linesA)-1].split()])
    
    npt.assert_array_almost_equal(flinesC,np.array([6.25,6.25,6.25,6.25,6.25,
                                                     6.25,6.25,6.25]),
                                                     decimal=2)
    npt.assert_array_almost_equal(flinesO,np.array([0.,4.18,0.,4.18,4.18,
                                                     4.18,4.18,4.18]),
                                                     decimal=2)
    npt.assert_array_almost_equal(flinesA,np.array([0.,0.,0.,1.03,0.98,0.,1.03,
                                                     0.98]),
                                                     decimal=2)
def test_writeSizes():
    """
    test writing out the clusterIDs
    """
    fname = 'mols8.gsd'
    traj = gsd.hoomd.open(op.join(data_path, fname))
    box = traj[0].configuration.box
    ats = {'contact':17}
    cutoff= 1.1*1.1
    molno = 8
    cldict = {'contact':cutoff}
    syst = cl.SnapSystem(traj,ats,molno,cldict)
    syst.get_clusters_serial('contact',box)
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
    d0 = alignDistancesCython(apos1,apos2)
    npt.assert_almost_equal(d0,0.25,10)
    
    apos1 = np.array([-1.,0.,0.,-1.25,-0.5,0.,
                      -1.5,-1.,0.,-0.5,-1.,0.,
                      -0.25,-0.5,0.,0.,0.,0.])
    apos2 = np.array([0.5,0.,0.,1.5,0.,0.,
                      0.75,-0.5,0.,1.75,-0.5,0.,
                      1.,-1.,0.,2.,-1.,0.])
    d1 = alignDistancesCython(apos1,apos2)
    npt.assert_almost_equal(d1,1.5 * 1.5, 10)

def test_squashRNG():
    """
    test that squashing an RNG works correctly
    """
    rng = np.array([[1,1,1,0,0,0,1,1,0],
                    [1,1,1,0,0,0,1,0,0],
                    [1,1,1,0,0,0,0,0,0],
                    [0,0,0,1,1,1,0,0,0],
                    [0,0,0,1,1,1,0,0,0],
                    [0,0,0,1,1,1,0,0,0],
                    [1,1,0,0,0,0,1,1,1],
                    [1,0,0,0,0,0,1,1,1],
                    [0,0,0,0,0,0,1,1,1]])
    rng = scipy.sparse.csr_matrix(rng)
    rngS = cl.squashRNG(rng,3)
    rngMini = np.array([[0,0,1],[0,0,0],[1,0,0]])
    npt.assert_array_equal(rngMini,rngS.toarray())

def test_squashRNGCython():
    """
    test that squashing an RNG works correctly
    """
    rng = np.array([[1,1,1,0,0,0,1,1,0],
                    [1,1,1,0,0,0,1,0,0],
                    [1,1,1,0,0,0,0,0,0],
                    [0,0,0,1,1,1,0,0,0],
                    [0,0,0,1,1,1,0,0,0],
                    [0,0,0,1,1,1,0,0,0],
                    [1,1,0,0,0,0,1,1,1],
                    [1,0,0,0,0,0,1,1,1],
                    [0,0,0,0,0,0,1,1,1]]).astype(float)
    rng = scipy.sparse.csr_matrix(rng)
    rngS = cl.squashRNGCython(rng,3)
    rngMini = np.array([[0,0,1],[0,0,0],[1,0,0]])
    npt.assert_array_equal(rngMini,rngS.toarray())
    
def test_squashRNGPython():
    """
    test that squashing an RNG works correctly
    """
    rng = np.array([[1,1,1,0,0,0,1,1,0],
                    [1,1,1,0,0,0,1,0,0],
                    [1,1,1,0,0,0,0,0,0],
                    [0,0,0,1,1,1,0,0,0],
                    [0,0,0,1,1,1,0,0,0],
                    [0,0,0,1,1,1,0,0,0],
                    [1,1,0,0,0,0,1,1,1],
                    [1,0,0,0,0,0,1,1,1],
                    [0,0,0,0,0,0,1,1,1]]).astype(float)
    rng = scipy.sparse.csr_matrix(rng)
    rngS = cl.squashRNGPy(rng,3)
    rngMini = np.array([[0,0,1],[0,0,0],[1,0,0]])
    npt.assert_array_equal(rngMini,rngS.toarray())
    
def test_squashRNGCOO():
    """
    test that squashing an RNG works correctly
    """
    rng = np.array([[1,1,1,0,0,0,1,1,0],
                    [1,1,1,0,0,0,1,0,0],
                    [1,1,1,0,0,0,0,0,0],
                    [0,0,0,1,1,1,0,0,0],
                    [0,0,0,1,1,1,0,0,0],
                    [0,0,0,1,1,1,0,0,0],
                    [1,1,0,0,0,0,1,1,1],
                    [1,0,0,0,0,0,1,1,1],
                    [0,0,0,0,0,0,1,1,1]]).astype(float)
    rng = scipy.sparse.csr_matrix(rng)
    rngS = cl.squashRNGCOO(rng,3)
    rngMini = np.array([[0,0,1],[0,0,0],[0,0,0]])
    npt.assert_array_equal(rngMini,rngS.toarray())

def test_squashRNGCOOCython():
    """
    test that squashing an RNG works correctly
    """
    rng = np.array([[1,1,1,0,0,0,1,1,0],
                    [1,1,1,0,0,0,1,0,0],
                    [1,1,1,0,0,0,0,0,0],
                    [0,0,0,1,1,1,0,0,0],
                    [0,0,0,1,1,1,0,0,0],
                    [0,0,0,1,1,1,0,0,0],
                    [1,1,0,0,0,0,1,1,1],
                    [1,0,0,0,0,0,1,1,1],
                    [0,0,0,0,0,0,1,1,1]]).astype(float)
    rng = scipy.sparse.csr_matrix(rng)
    rngS = cl.squashRNGCOOCython(rng,3)
    rngMini = np.array([[0,0,1],[0,0,0],[0,0,0]])
    npt.assert_array_equal(rngMini,rngS.toarray())
    
""" Testing the getting of indices for a CSR matrix"""
def test_get_csr_inds_lil():
    mat = np.array([[0,0,0,0],[5,8,0,0],[0,0,3,0],[0,6,0,0]])
    matcsr = lil_matrix(mat)
    with pytest.raises(TypeError):
        cl.getIndsCsr(matcsr)
        
def test_get_csr_inds_dense():
    matcsr = np.array([[0,0,0,0],[5,8,0,0],[0,0,3,0],[0,6,0,0]])
    with pytest.raises(TypeError):
        cl.getIndsCsr(matcsr)
        
def test_get_csr_inds():
    mat = np.array([[0,0,0,0],[5,8,0,0],[0,0,3,0],[0,6,0,0]])
    matcsr = csr_matrix(mat)
    bonds = cl.getIndsCsr(matcsr)
    abonds = np.array([[1,0],[1,1],[2,2],[3,1]])
    npt.assert_array_equal(bonds,abonds)
    
def test_get_csr_inds2():
    mat = np.array([[10,20,0,0,0,0],
                    [0,30,0,40,0,0],
                    [0,0,50,60,70,0],
                    [0,0,0,0,0,80]])
    matcsr = csr_matrix(mat)
    abonds = np.array([[0,0],
                       [0,1],
                       [1,1],
                       [1,3],
                       [2,2],
                       [2,3],
                       [2,4],
                       [3,5]])
    bonds = cl.getIndsCsr(matcsr)
    npt.assert_array_equal(bonds,abonds)     

"""test intermix function"""

def test_intermixsetup():

    fname = 'dummyfull2type_run1.gsd'
    traj = gsd.hoomd.open(op.join(data_path,fname))
    compairs = np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]])
    molno = 4
    contactcut = 1.1*1.1
    opticalcut = 0.35*0.35
    ats = {'contact':17,'optical':12} 
    cutoff = {'contact':contactcut,'optical':opticalcut}
    Syst = cl.SnapSystem(traj, ats, molno, cutoff, 
                 compairs=compairs,
                 atype='AB',ttotal=6,tstart=0,
                 het=True,typelist=[u'EA',u'EB'])
    
    boxL = traj[0].configuration.box[0]
    Syst.get_clusters_serial('contact',boxL)
    Syst.get_clusters_serial('optical',boxL)
    intermixlistC = Syst.writeIntermix('contact',
                                      'dummyfull2type_run1_cImix.dat',
                                      returnmat=True)
    intermixlistO = Syst.writeIntermix('optical',
                                      'dummyfull2type_run1_oImix.dat',
                                      returnmat=True)       
    iC0 = np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]])
    iC1 = np.array([[2,0,1,1],[2,0,1,1]])
    iC2 = np.array([[4,1,2,3]])
    iC3 = np.array([[4,0,4,4]])
    iC4 = np.array([[4,1,4,5]])
    iC5 = np.array([[4,2,4,6]])
    iCs = [iC0,iC1,iC2,iC3,iC4,iC5]
    iO0 = iC0
    iO1 = np.array([[1,0,0,0],[1,0,0,0],[2,0,1,1]])
    iO2 = np.array([[2,0,1,1],[2,0,1,1]])
    iO3 = np.array([[2,0,1,1],[2,0,1,1]])
    iO4 = np.array([[4,1,2,3]])
    iO5 = np.array([[4,2,2,4]])
    iOs = [iO0,iO1,iO2,iO3,iO4,iO5]
    return (intermixlistC,intermixlistO,iCs,iOs)
    
def test_intermixC0():
    (intermixlistC,intermixlistO,iCs,iOs) = test_intermixsetup()
    npt.assert_array_equal(iCs[0],intermixlistC[0])
    
def test_intermixO0():
    (intermixlistC,intermixlistO,iCs,iOs) = test_intermixsetup()
    npt.assert_array_equal(iOs[0],intermixlistO[0])
    
def test_intermixC1():
    (intermixlistC,intermixlistO,iCs,iOs) = test_intermixsetup()
    npt.assert_array_equal(iCs[1],intermixlistC[1])
    
def test_intermixO1():
    (intermixlistC,intermixlistO,iCs,iOs) = test_intermixsetup()
    npt.assert_array_equal(iOs[1],intermixlistO[1])
    
def test_intermixC2():
    (intermixlistC,intermixlistO,iCs,iOs) = test_intermixsetup()
    npt.assert_array_equal(iCs[2],intermixlistC[2])
    
def test_intermixO2():
    (intermixlistC,intermixlistO,iCs,iOs) = test_intermixsetup()
    npt.assert_array_equal(iOs[2],intermixlistO[2])
    
def test_intermixC3():
    (intermixlistC,intermixlistO,iCs,iOs) = test_intermixsetup()
    npt.assert_array_equal(iCs[3],intermixlistC[3])
    
def test_intermixO3():
    (intermixlistC,intermixlistO,iCs,iOs) = test_intermixsetup()
    npt.assert_array_equal(iOs[3],intermixlistO[3])

def test_intermixC4():
    (intermixlistC,intermixlistO,iCs,iOs) = test_intermixsetup()
    npt.assert_array_equal(iCs[4],intermixlistC[4])
    
def test_intermixO4():
    (intermixlistC,intermixlistO,iCs,iOs) = test_intermixsetup()
    npt.assert_array_equal(iOs[4],intermixlistO[4])
    
def test_intermixC5():
    (intermixlistC,intermixlistO,iCs,iOs) = test_intermixsetup()
    npt.assert_array_equal(iCs[5],intermixlistC[5])
    
def test_intermixO5():
    (intermixlistC,intermixlistO,iCs,iOs) = test_intermixsetup()
    npt.assert_array_equal(iOs[5],intermixlistO[5])
    
"""
Test angle spread function(s)
"""
def test_ang_spread_1mol():
    """
    For one molecule, the output should be a 1 x 3 matrix with 0 for the 
    spread and 1 for the mean
    """
    trj = op.join(data_path,'DFAG.gro')    
    ats = 29
    molno = 1
    t = 0
    cutoff = 0.5
    ainds = np.arange(9,20)
    snap = cl.ContactClusterSnapshotXTC(t,trj,ats,molno)
    snap.setClusterID(cutoff)
    angMat = snap.angSpread(cutoff,ainds)
    expangMat = np.array([[1,0.,1.]])
    #pdb.set_trace()
    npt.assert_array_almost_equal(angMat,expangMat)
    
def test_ang_spread_2mols_no_rotate_short():
    """
    two molecules parallel to one another should have an angle spread of 0
    the mean projection should depend on how far apart they are (because it
    depends on the direction of the principal gyration tensor)
    """
    #initialize dummy snapshot
    trj = op.join(data_path,'DFAG.gro')    
    ats = 29
    molno = 1
    t = 0
    cutoff = 2.
    ainds = np.arange(0,3)
    snap = cl.ContactClusterSnapshotXTC(t,trj,ats,molno)
    snap.setClusterID(cutoff)
    #reset all snapshot properties
    snap.nclusts = 1
    snap.box = np.array([20,20,20])
    snap.clusterIDs = [0,0]
    snap.pos = np.array([[0.,0.,0.,1.,0.,0.,2.,0.,0.],
                         [0.,1.,0.,1.,1.,0.,2.,1.,0.]])
    angMat = snap.angSpread(cutoff,ainds)
    expangMat = np.array([[2,0.,1.]])
    npt.assert_array_almost_equal(angMat,expangMat)
    
def test_ang_spread_2mols_no_rotate_long():
    #initialize dummy snapshot
    trj = op.join(data_path,'DFAG.gro')    
    ats = 29
    molno = 1
    t = 0
    cutoff = 20.*20.
    ainds = np.arange(0,3)
    snap = cl.ContactClusterSnapshotXTC(t,trj,ats,molno)
    snap.setClusterID(cutoff)
    #reset all snapshot properties
    snap.nclusts = 1
    snap.box = np.array([20,20,20])
    snap.clusterIDs = [0,0]
    snap.pos = np.array([[0.,0.,0.,1.,0.,0.,2.,0.,0.],
                         [0.,10.,0.,1.,10.,0.,2.,10.,0.]])
    angMat = snap.angSpread(cutoff,ainds)
    expangMat = np.array([[2,0.,0.]])
    npt.assert_array_almost_equal(angMat,expangMat)
 
def test_ang_spread_2mols_rotate180():
 
    #initialize dummy snapshot
    trj = op.join(data_path,'DFAG.gro')    
    ats = 29
    molno = 1
    t = 0
    cutoff = 2.
    ainds = np.arange(0,3)
    snap = cl.ContactClusterSnapshotXTC(t,trj,ats,molno)
    snap.setClusterID(cutoff)
    #reset all snapshot properties
    snap.nclusts = 1
    snap.box = np.array([20,20,20])
    snap.clusterIDs = [0,0]
    snap.pos = np.array([[0.,0.,0.,1.,0.,0.,2.,0.,0.],
                         [2.,1.,0.,1.,1.,0.,0.,1.,0.]])
    angMat = snap.angSpread(cutoff,ainds)
    expangMat = np.array([[2,0.,1.]])
    npt.assert_array_almost_equal(angMat,expangMat)   

def test_ang_spread_2mols_rotate_long90():
    #initialize dummy snapshot
    trj = op.join(data_path,'DFAG.gro')    
    ats = 29
    molno = 1
    t = 0
    cutoff = 20.*20.
    ainds = np.arange(0,3)
    snap = cl.ContactClusterSnapshotXTC(t,trj,ats,molno)
    snap.setClusterID(cutoff)
    #reset all snapshot properties
    snap.nclusts = 1
    snap.box = np.array([20,20,20])
    snap.clusterIDs = [0,0]
    snap.pos = np.array([[0.,0.,0.,1.,0.,0.,2.,0.,0.],
                         [1.,10.,1.,1.,10.,0.,1.,10.,-1.]])
    angMat = snap.angSpread(cutoff,ainds)
    expangMat = np.array([[2,1.1107,0.]])
    npt.assert_array_almost_equal(angMat,expangMat,decimal=4)
    
def test_ang_spread_2mols_rotate_long45():
    #initialize dummy snapshot
    trj = op.join(data_path,'DFAG.gro')    
    ats = 29
    molno = 1
    t = 0
    cutoff = 20.*20.
    ainds = np.arange(0,3)
    snap = cl.ContactClusterSnapshotXTC(t,trj,ats,molno)
    snap.setClusterID(cutoff)
    #reset all snapshot properties
    snap.nclusts = 1
    snap.box = np.array([20,20,20])
    snap.clusterIDs = [0,0]
    snap.pos = np.array([[0.,0.,0.,1.,0.,0.,2.,0.,0.],
                         [1-np.sin(np.pi/4),10.,0.-np.cos(np.pi/4),1.,10.,0.,
                          1.+np.sin(np.pi/4),10.,np.cos(np.pi/4)]])
    angMat = snap.angSpread(cutoff,ainds)
    expangMat = np.array([[2,0.5554,0.]])
    npt.assert_array_almost_equal(angMat,expangMat,decimal=4)
    
def test_ang_spread_10mols45():
    #initialize dummy snapshot
    trj = op.join(data_path,'DFAG.gro')    
    ats = 29
    molno = 1
    t = 0
    cutoff = 2.*2
    ainds = np.arange(0,3)
    snap = cl.ContactClusterSnapshotXTC(t,trj,ats,molno)
    snap.setClusterID(cutoff)
    #reset all snapshot properties
    snap.nclusts = 1
    snap.box = np.array([30,30,30])
    snap.clusterIDs = [0,0,0,0,0,0,0,0,0,0]
    snap.pos = np.array([[0.,0.,0.,1.,0.,0.,2.,0.,0.],
                         [0.,1.,0.,1.,1.,0.,2.,1.,0.],
                         [0.,2.,0.,1.,2.,0.,2.,2.,0.],
                         [0.,3.,0.,1.,3.,0.,2.,3.,0.],
                         [0.,4.,0.,1.,4.,0.,2.,4.,0.],
                         [1-np.sin(np.pi/4),5.,0.-np.cos(np.pi/4),1.,5.,0.,
                          1.+np.sin(np.pi/4),5.,np.cos(np.pi/4)],
                         [0.,6.,0.,1.,6.,0.,2.,6.,0.],
                         [0.,7.,0.,1.,7.,0.,2.,7.,0.],
                         [0.,8.,0.,1.,8.,0.,2.,8.,0.],
                         [1.,9.,0.,1.,10.,0.,1.,11.,0.]])
    angMat = snap.angSpread(cutoff,ainds)
    expangMat = np.array([[10,0.2484,0.1]])
    npt.assert_array_almost_equal(angMat,expangMat,decimal=4)
    
"""
Test nearest-neighbor angle spread
"""
def test_nnang_spread_1mol():
    """
    For one molecule, the output should be an empty matrix 
    """
    trj = op.join(data_path,'DFAG.gro')    
    ats = 29
    molno = 1
    t = 0
    cutoff = 0.5
    ainds = np.arange(9,20)
    snap = cl.ContactClusterSnapshotXTC(t,trj,ats,molno)
    snap.setClusterID(cutoff)
    angMat = snap.nnangSpread(ainds)
    expangMat = np.array([])
    #pdb.set_trace()
    npt.assert_array_almost_equal(angMat,expangMat)
    
def test_nnang_spread_2mols_no_rotate_short():
    """
    two molecules parallel to one another should make an angle of 0 and
    by definition stddev should be 0
    """
    #initialize dummy snapshot
    trj = op.join(data_path,'DFAG.gro')    
    ats = 29
    molno = 1
    t = 0
    cutoff = 2.
    ainds = np.arange(0,3)
    snap = cl.ContactClusterSnapshotXTC(t,trj,ats,molno)
   
    #reset all snapshot properties
    snap.nclusts = 1
    snap.box = np.array([20,20,20])
    snap.clusterIDs = [0,0]
    snap.pos = np.array([[0.,0.,0.,1.,0.,0.,2.,0.,0.],
                         [0.,1.,0.,1.,1.,0.,2.,1.,0.]])
    snap.setClusterID(cutoff)
    angMat = snap.nnangSpread(ainds)
    expangMat = np.array([[2,0.,0.]])
    npt.assert_array_almost_equal(angMat,expangMat)
    
def test_nnang_spread_2mols_rotate180():
 
    #initialize dummy snapshot
    trj = op.join(data_path,'DFAG.gro')    
    ats = 29
    molno = 1
    t = 0
    cutoff = 2.
    ainds = np.arange(0,3)
    snap = cl.ContactClusterSnapshotXTC(t,trj,ats,molno)
    
    #reset all snapshot properties
    snap.nclusts = 1
    snap.box = np.array([20,20,20])
    snap.clusterIDs = [0,0]
    snap.pos = np.array([[0.,0.,0.,1.,0.,0.,2.,0.,0.],
                         [2.,1.,0.,1.,1.,0.,0.,1.,0.]])
    snap.setClusterID(cutoff)
    angMat = snap.nnangSpread(ainds)
    expangMat = np.array([[2,0.,0.]])
    npt.assert_array_almost_equal(angMat,expangMat)  
    
def test_nnang_spread_2mols_rotate_long90():
    #initialize dummy snapshot
    trj = op.join(data_path,'DFAG.gro')    
    ats = 29
    molno = 1
    t = 0
    cutoff = 2.
    ainds = np.arange(0,3)
    snap = cl.ContactClusterSnapshotXTC(t,trj,ats,molno)
    
    #reset all snapshot properties
    snap.nclusts = 1
    snap.box = np.array([20,20,20])
    snap.clusterIDs = [0,0]
    snap.pos = np.array([[0.,0.,0.,1.,0.,0.,2.,0.,0.],
                         [1.,1.,1.,1.,1.,0.,1.,1.,-1.]])
    snap.setClusterID(cutoff)
    angMat = snap.nnangSpread(ainds)
    expangMat = np.array([[2,np.pi/2,0]])
    npt.assert_array_almost_equal(angMat,expangMat)
    
def test_nnang_spread_2mols_rotate_long45():
    #initialize dummy snapshot
    trj = op.join(data_path,'DFAG.gro')    
    ats = 29
    molno = 1
    t = 0
    cutoff = 2
    ainds = np.arange(0,3)
    snap = cl.ContactClusterSnapshotXTC(t,trj,ats,molno)
    
    #reset all snapshot properties
    snap.nclusts = 1
    snap.box = np.array([20,20,20])
    snap.clusterIDs = [0,0]
    snap.pos = np.array([[0.,0.,0.,1.,0.,0.,2.,0.,0.],
                         [1-np.sin(np.pi/4),1.,0.-np.cos(np.pi/4),1.,1.,0.,
                          1.+np.sin(np.pi/4),1.,np.cos(np.pi/4)]])
    snap.setClusterID(cutoff)
    angMat = snap.nnangSpread(ainds)
    expangMat = np.array([[2,np.pi/4,0.]])
    npt.assert_array_almost_equal(angMat,expangMat,decimal=4)
    
def test_nnang_spread_10mols45():
    #initialize dummy snapshot
    trj = op.join(data_path,'DFAG.gro')    
    ats = 29
    molno = 1
    t = 0
    cutoff = 1.
    ainds = np.arange(0,3)
    snap = cl.ContactClusterSnapshotXTC(t,trj,ats,molno)
    #reset all snapshot properties
    snap.nclusts = 1
    snap.box = np.array([30,30,30])
    snap.clusterIDs = [0,0,0,0,0,0,0,0,0,0]
    snap.pos = np.array([[0.,0.,0.,1.,0.,0.,2.,0.,0.],
                         [0.,1.,0.,1.,1.,0.,2.,1.,0.],
                         [0.,2.,0.,1.,2.,0.,2.,2.,0.],
                         [0.,3.,0.,1.,3.,0.,2.,3.,0.],
                         [0.,4.,0.,1.,4.,0.,2.,4.,0.],
                         [1-np.sin(np.pi/4),5.,0.-np.cos(np.pi/4),1.,5.,0.,
                          1.+np.sin(np.pi/4),5.,np.cos(np.pi/4)],
                         [0.,6.,0.,1.,6.,0.,2.,6.,0.],
                         [0.,7.,0.,1.,7.,0.,2.,7.,0.],
                         [0.,8.,0.,1.,8.,0.,2.,8.,0.],
                         [0.,9.,0.,1.,9.,0.,2.,9.,0.]])
   
    snap.setClusterID(cutoff)
    angMat = snap.nnangSpread(ainds)
    expangMat = np.array([[10,0.17453292519943295,0.34632803675275337]])
    npt.assert_array_almost_equal(angMat,expangMat)
"""
def test_valid_metric():
    loop = 10
    for j in range(loop):
        ln = 5#random.randint(3,50)
        ln = 3*ln
        x = np.zeros(ln)
        y = np.zeros(ln)
        z = np.zeros(ln)
        a = -100
        b = 100
        for i in range(ln):
            x[i] = random.uniform(a,b)
            y[i] = random.uniform(a,b)
            z[i] = random.uniform(a,b)
        try:
            assert cl.conOptDistance(x,y) >= 0
            assert cl.conOptDistance(x,x) == 0
            assert cl.conOptDistance(y,y) == 0
            if (x != y).any():
                assert cl.conOptDistance(x,y) != 0
            else:
                assert cl.conOptDistance(x,y) == 0
            assert cl.conOptDistance(x,y) + cl.conOptDistance(y,z) >= cl.conOptDistance(x,z)
        except AssertionError:
            pdb.set_trace()
"""
if __name__ == "__main__":
    test_intermix1()
