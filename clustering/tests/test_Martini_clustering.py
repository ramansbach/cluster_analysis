# -*- coding: utf-8 -*-
"""
Created on Tue May  1 13:20:59 2018

@author: rachael

Test XTC clustering stuff on a single pull run
"""
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
import time
#from context import clustering as cl
#from context import smoluchowski as smol
from cdistances import conOptDistanceCython,alignDistancesCython
#import imp
#cl = imp.load_source('cl','/home/rachael/Analysis_and_run_code/analysis/cluster_analysis/clustering/clustering.py')
data_path = op.join(cl.__path__[0], 'data')

def test_contactXTC_init_contact():
    """
    Test initializing a contact cluster from xtc/gro
    """
    trj = op.join(data_path,'contact_test.gro')    
    ats = 5
    molno = 2
    t = 0
    snap = cl.ContactClusterSnapshotXTC(t,trj,ats,molno)
    pos = np.array([[0.,0.,0.,0.354,0.354,0.0,0.604,0.8,0.0,0.854,0.354,0.0,
                     1.207,0.0,0.0],
                    [0.0,-0.8,0.0,0.354,-1.154,0.0,0.604,-1.6,0.0,0.854,
                     -1.154,0.0,1.207,-0.8,0.0]])
    npt.assert_array_equal(snap.pos,pos)
    
def test_contactXTC_init_optical():
    """
    Test initializing an optical cluster from xtc/gro
    """
    trj = op.join(data_path,'optical_test.gro')    
    ats = 5
    molno = 2
    t = 0
    snap = cl.ContactClusterSnapshotXTC(t,trj,ats,molno)
    pos = np.array([[0.,0.,0.,0.5,0.,0.,0.75,0.443,0.,1.,0.,0.,1.5,0.,0.],
                    [0.,0.,0.5,0.5,0.,0.5,0.75,0.443,0.5,1.,0.,0.5,1.5,0.,
                     0.5]])
    npt.assert_array_equal(snap.pos,pos)
    
def test_opticalXTC_init_contact():
    """
    Test initializing an optical cluster from xtc/gro
    """
    trj = op.join(data_path,'contact_test.gro')    
    ats = 5
    molno = 2
    t = 0
    comIDs = np.array([[1,2,3]])
    snap = cl.OpticalClusterSnapshotXTC(t,trj,ats,molno,comIDs)
    pos = np.array([[(0.354+0.604+0.854)/3.,(0.354+0.8+0.354)/3.,0.],
                    [(0.354+0.604+0.854)/3.,(-1.154-1.6-1.154)/3.,0.]])
    npt.assert_array_equal(snap.pos,pos)
    
def test_opticalXTC_init_optical():
    """
    Test initializing an optical cluster from xtc/gro
    """
    trj = op.join(data_path,'optical_test.gro')    
    ats = 5
    molno = 2
    t = 0
    comIDs = np.array([[1,2,3]])
    snap = cl.OpticalClusterSnapshotXTC(t,trj,ats,molno,comIDs)
    pos = np.array([[(0.5+0.75+1.)/3.,(0.+0.443+0.)/3.,0.],
                    [(0.5+0.75+1.)/3.,(0.+0.443+0.)/3.,0.5]])
    npt.assert_array_equal(snap.pos,pos)
    
def test_contactXTC_clustering_contact():
    """
    Make sure we cluster correctly for contact cluster from xtc/gro
    """
    trj = op.join(data_path,'contact_test.gro')    
    ats = 5
    molno = 2
    t = 0
    snap = cl.ContactClusterSnapshotXTC(t,trj,ats,molno)
    snap.setClusterID(1.0)
    assert(snap.nclusts==1)
    
def test_contactXTC_clustering_optical():
    """
    Make sure we cluster correctly for contact cluster from xtc/gro
    """
    trj = op.join(data_path,'contact_test.gro')    
    ats = 5
    molno = 2
    t = 0
    comIDs = np.array([[1,2,3]])
    snap = cl.OpticalClusterSnapshotXTC(t,trj,ats,molno,comIDs)
    snap.setClusterID(1.0)
    assert(snap.nclusts==2)

def test_opticalXTC_clustering_contact():
    """
    Make sure we cluster correctly for contact cluster from xtc/gro
    """
    trj = op.join(data_path,'optical_test.gro')    
    ats = 5
    molno = 2
    t = 0
    snap = cl.ContactClusterSnapshotXTC(t,trj,ats,molno)
    snap.setClusterID(0.6)
    assert(snap.nclusts==1)
    
def test_opticalXTC_clustering_optical():
    """
    Make sure we cluster correctly for contact cluster from xtc/gro
    """
    trj = op.join(data_path,'optical_test.gro')    
    ats = 5
    molno = 2
    t = 0
    comIDs = np.array([[1,2,3]])
    snap = cl.OpticalClusterSnapshotXTC(t,trj,ats,molno,comIDs)
    snap.setClusterID(0.6)
    assert(snap.nclusts==1)
    
def test_trajectory():
    """
    Try running on an xtc trajectory (from a pull simulation)
    """
    trj = op.join(data_path,'pull_whole.xtc')
    tpr = op.join(data_path,'pull.tpr')
    molno = 2
    ats = 29
    tstart = 0
    ttotal = 498
    comIDs = np.array([[9,10,11],[13,14,15],[18,19,20]])
    cldict = {'contact':0.5*0.5,'optical':0.7*0.7}
    start = time.time()
    syst = cl.SnapSystem(trj,ats,molno,cldict,compairs=comIDs, 
                ttotal=ttotal,tstart=tstart,tpr=tpr)
    end = time.time()
    print("Time to setup: "+str(end-start)+"\n")
    start = time.time()
    syst.get_clusters_serial('contact',np.array([[24.25935],[24.25935],
                                                 [24.25935]]))
    end = time.time()
    print("Time to get contact: "+str(end-start)+"\n")
    start = time.time()
    syst.get_clusters_serial('optical',np.array([[24.25935],[24.25935],
                                                 [24.25935]]))
    end = time.time()
    print("Time to get optical: "+str(end-start)+"\n")
    syst.writeCIDs('contact','pull-test-contact-CIDs.dat')
    syst.writeCIDs('optical','pull-test-optical-CIDs.dat')
    syst.writeSizes('contact','pull-test-contact-sizes.dat')
    syst.writeSizes('optical','pull-test-optical-sizes.dat')
    
if __name__ == "__main__":
    test_trajectory()
    
