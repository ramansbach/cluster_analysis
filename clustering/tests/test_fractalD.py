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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
font = {'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
#data_path = '/home/rachael/Analysis_and_run_code/analysis/cluster_analysis/clustering/tests/test_fractalD/'
#data_path2 = '/home/rachael/Analysis_and_run_code/analysis/cluster_analysis/clustering/data'

def test_fit2():
    """
    Test the fitting of two lines with data generated from two lines
    stitched together
    """
    x = np.arange(0,10,0.1)
    as1 = 3
    ai1 = -4.7
    as2 = 0.5
    y = np.arange(0,10,0.1)
    w = np.ones(np.shape(y))
    ajunct = 38
    y[0:ajunct+1] = x[0:ajunct+1]*as1 + ai1
    y[(ajunct):len(y)] = as2*x[(ajunct):len(y)] - as2*x[ajunct]\
                           + y[ajunct]
    (r,l1,l2) = cl.fit2(x,y,w,ajunct)
    npt.assert_almost_equal(l1.slope,as1)
    npt.assert_almost_equal(l2.slope,as2)
    npt.assert_almost_equal(l1.intercept,ai1)
    npt.assert_almost_equal(l2.intercept,ai1+as1*x[ajunct] - as2 * x[ajunct])
    npt.assert_almost_equal(r,0)

def test_methodL():
    """
    Test the L method simply
    """
    x = np.arange(0,10,0.1)
    as1 = 3
    ai1 = -4.7
    as2 = 0.5
    y = np.arange(0,10,0.1)
    w = np.ones(np.shape(y))
    ajunct = 38
    y[0:ajunct+1] = x[0:ajunct+1]*as1 + ai1
    y[(ajunct):len(y)] = as2*x[(ajunct):len(y)] - as2*x[ajunct]\
                           + y[ajunct]
    (xjunct,l1,l2,terr) = cl.methodL(x,y,w)
    assert xjunct == ajunct
    npt.assert_almost_equal(l1.slope,as1)
    npt.assert_almost_equal(l2.slope,as2)
    npt.assert_almost_equal(l1.intercept,ai1)
    npt.assert_almost_equal(l2.intercept,ai1+as1*x[ajunct] - as2 * x[ajunct])
    npt.assert_almost_equal(terr,0)

def test_noisyL():
    """
    Test the L method on the mean of noisy data (with plot and weights)
    """
    x = np.arange(0,10,0.1)
    as1 = 3
    ai1 = -4.7
    as2 = 0.5
    y = np.arange(0,10,0.1)
    figname = op.join(data_path2,'noisyLmethod')
    markers = ['x','s','^','v','o']
    ajunct = 38
    y[0:ajunct+1] = x[0:ajunct+1]*as1 + ai1
    y[(ajunct):len(y)] = as2*x[(ajunct):len(y)] - as2*x[ajunct]\
                           + y[ajunct]
    runs = 5
    ys = np.zeros([len(x),runs])
    for run in range(runs):
        noise = np.random.randn(len(x))
        ys[:,run] = y + noise
    meanY = np.mean(ys,axis=1)
    w = 1/np.std(ys,axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for run in range(runs):
        ax.plot(x,ys[:,run],markers[run])
    (xjunct,l1,l2,terr) = cl.methodL(x,meanY,w)
    ax.plot(x[0:xjunct],l1.slope*x[0:xjunct]+l1.intercept,color='black')
    ax.plot(x[xjunct:len(x)],l2.slope*x[xjunct:len(x)]+l2.intercept,color='black')
    fig.savefig(figname)
    plt.close()
    npt.assert_approx_equal(xjunct,ajunct,1)
    npt.assert_approx_equal(l1.slope,as1,1)
    npt.assert_approx_equal(l2.slope,as2,1)
    npt.assert_approx_equal(l1.intercept,ai1,1)
    npt.assert_approx_equal(l2.intercept,ai1+as1*x[ajunct] - as2 * x[ajunct],1)


def test_comparepy():
    """
    This reads in a set of test data, computes the correlation integral
    and saves it out.  It should then be tested against the Matlab function
    comparepy.m 
    """
    for i in range(1,11):
        f = open(data_path2+'/coords_'+str(i)+'.dat')
        lines = f.readlines()
        f.close()
        coords = np.zeros([len(lines),3])
        for j in range(len(lines)):
            coords[j,:] = [float(s) for s in lines[j].split()]
        cl.corrcalc(coords,22,0.1,fname=data_path2+'/cepy_'+str(i)+'.dat')
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

def test_compareMartini():
    """
    Checks that new code produces the same thing as the older Markov code
    version
    """
    emax = 42
    estep = 0.1
    t = 400000
   
    #pdb.set_trace()
    xtc = 'md_final.xtc'
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
    #coms = getCOMs(xtcSnap.pos,masslist)
    #pdb.set_trace()
    #npt.assert_array_almost_equal(coms,comspy)
    cemat = cl.corrcalc(comspy,emax,estep)
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
    '''
