from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import numpy.testing as npt
import pdb
import gsd.hoomd
import sys
#from context import fractald as fd
#from context import clustering as cl
import clustering as cl
from cdistances import conOptDistanceCython,alignDistancesCython
from cfractald import getCOMs
data_path = op.join(cl.__path__[0],'data')
data_path2 = op.join(cl.__path__[0],'tests/test_fractalD')
#data_path = '/home/rachael/Analysis_and_run_code/analysis/cluster_analysis/clustering/tests/test_fractalD/'
#data_path2 = '/home/rachael/Analysis_and_run_code/analysis/cluster_analysis/clustering/data'


def test_comparepy():
    """
    This reads in a set of test data, computes the correlation integral
    and saves it out.  It should then be tested against the Matlab function
    comparepy.m
    """
    for i in range(1,11):
        f = open(data_path+'coords_'+str(i)+'.dat')
        lines = f.readlines()
        f.close()
        coords = np.zeros([len(lines),3])
        for j in range(len(lines)):
            coords[j,:] = [float(s) for s in lines[j].split()]
        cl.corrcalc(coords,22,0.1,fname=data_path+'cepy_'+str(i)+'.dat')
'''
def test_coms():
    """
    Attempting to sanity check/fix c-kernel computation of COM
    """
    t = 400000
    xtc = 'md_test.xtc'
    tpr = 'md_dummy.tpr'
    trj = op.join(data_path2,xtc)
    tpr = op.join(data_path2,tpr)
    outGro = 'temp.gro'
    outGro = op.join(data_path2,outGro)
    ats = 29
    molno = 378
    masslist = np.ones(ats)*45.
    m72 = [0,1,2,6,7,21,22,23,27,28]
    masslist[m72] = 72.
    xtcSnap = cl.ContactClusterSnapshotXTC(t, trj, tpr, outGro, ats, molno)
    comspy = cl.getCOMsPy(np.reshape(xtcSnap.pos,[1,3*ats*molno])[0],masslist,ats)
    comspy = np.reshape(comspy,[molno,3])
    coms = getCOMs(xtcSnap.pos,masslist)
    comscpy = np.zeros([molno,3])
    comscpy = cl.getcomsPy(xtcSnap.pos,comscpy,masslist,ats,molno)
    npt.assert_array_almost_equal(comscpy,comspy)
'''
def test_compareMartini():
    """
    Checks that new code produces the same thing as the older Markov code
    version
    """
    emax = 42
    estep = 0.1
    t = 400000
   

    xtc = 'md_test.xtc'
    tpr = 'md_dummy.tpr'
    trj = op.join(data_path2,xtc)
    tpr = op.join(data_path2,tpr)
    outGro = 'temp.gro'
    outGro = op.join(data_path2,outGro)
    ats = 29
    molno = 378
    masslist = np.ones(ats)*45.
    m72 = [0,1,2,6,7,21,22,23,27,28]
    masslist[m72] = 72.
    xtcSnap = cl.ContactClusterSnapshotXTC(t, trj, tpr, outGro, ats, molno)
    comspy = cl.getCOMsPy(np.reshape(xtcSnap.pos,[1,3*ats*molno])[0],masslist,ats)
    comspy = np.reshape(comspy,[molno,3])
    coms = getCOMs(xtcSnap.pos,masslist)
    #pdb.set_trace()
    #npt.assert_array_almost_equal(coms,comspy)
    cemat = cl.corrcalc(coms,emax,estep)
    olddata = 'test_cdim_400_snap.dat'
    olddata = op.join(data_path2,olddata)
    f = open(olddata)
    olddata = f.readlines()
    f.close()
    olddatamat = np.zeros([2,len(olddata)])
    
    for i in range(len(olddata)):
        olddatamat[0,i] = olddata[i].split()[0]
        olddatamat[1,i] = olddata[i].split()[1]
    npt.assert_array_almost_equal(cemat,olddatamat)
    
