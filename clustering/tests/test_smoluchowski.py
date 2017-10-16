from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import numpy.testing as npt
import scipy.constants as sconst
import gsd.hoomd
import sys
import pdb
import clustering as cl
#from context import clustering as cl
#from context import smoluchowski as smol

data_path = op.join(cl.__path__[0], 'data')
#data_path = '/home/rachael/Analysis_and_run_code/analysis/cluster_analysis/clustering/data'

def test_getSizesFromFile():
    """test reading sizes out of a written file"""
    tname = "dummyfull4"
    fname = 'molsdf4sizesAC'
    for i in range(1,6):
        traj = gsd.hoomd.open(op.join(data_path, tname+'_run'+str(i)+'.gsd'))
        ats = {'aligned':6}
        cldict={'aligned':0.35*0.35}
        molno = 4
        atype = 'AB'
        compairs = np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]])
        syst = cl.SnapSystem(traj,ats,molno,cldict,compairs=compairs,
                             atype=atype)
        syst.get_clusters_serial('aligned')
        syst.writeSizes('aligned',op.join(data_path,fname + '_run' \
                        + str(i) + '.dat'))
    fnames = [op.join(data_path,fname + '_run1.dat'),
              op.join(data_path,fname+'_run2.dat'),
              op.join(data_path,fname+'_run3.dat'),
              op.join(data_path,fname+'_run4.dat'),
              op.join(data_path,fname+'_run5.dat')]
   
    actualSizes1 = np.array([[1,1,1,1],[1,1,1,1],[2,2,1,1],
                    [2,2,2,2],[2,2,2,2],[4,4,4,4]])
    actualSizes2 = np.array([[1,1,1,1],[1,1,1,1],[2,2,1,1],
                    [2,2,2,2],[4,4,4,4],[4,4,4,4]])
    actualSizes3 = np.array([[1,1,1,1],[1,1,1,1],[2,2,2,2],
                    [2,2,2,2],[2,2,2,2],[2,2,2,2]])
    actualSizes4 = np.array([[1,1,1,1],[2,2,1,1],[2,2,2,2],
                    [2,2,2,2],[4,4,4,4],[4,4,4,4]])
    actualSizes5 = np.array([[1,1,1,1],[1,1,1,1],[2,2,1,1],
                    [2,2,2,2],[2,2,2,2],[2,2,2,2]])
    
    csizes1 = cl.getSizesFromFile([fnames[0]],6)
    assert (csizes1 == actualSizes1).all()
    csizes2 = cl.getSizesFromFile([fnames[1]],6)
    assert (csizes2 == actualSizes2).all()
    csizes3 = cl.getSizesFromFile([fnames[2]],6)
    assert (csizes3 == actualSizes3).all()
    csizes4 = cl.getSizesFromFile([fnames[3]],6)
    assert (csizes4 == actualSizes4).all()
    csizes5 = cl.getSizesFromFile([fnames[4]],6)
    assert (csizes5 == actualSizes5).all()
    
    csizes = cl.getSizesFromFile(fnames,6)
    print(csizes)
    assert (csizes == np.concatenate((csizes1,csizes2,csizes3,csizes4,
                                      csizes5),axis=1)).all()

    

def test_linearFit():
    """test Smol linear fitting on dummy runs"""
    tname = "dummyfull4"
    fname = 'molsdf4sizesAC'
    for i in range(1,6):
        traj = gsd.hoomd.open(op.join(data_path, tname+'_run'+str(i)+'.gsd'))
        ats = {'aligned':6}
        cldict={'aligned':0.35*0.35}
        molno = 4
        atype = 'AB'
        compairs = np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]])
        syst = cl.SnapSystem(traj,ats,molno,cldict,compairs=compairs,
                             atype=atype)
        syst.get_clusters_serial('aligned')
        syst.writeSizes('aligned',op.join(data_path,fname + '_run' \
                        + str(i) + '.dat'))
    fnames = [op.join(data_path,fname + '_run1.dat'),
              op.join(data_path,fname+'_run2.dat'),
              op.join(data_path,fname+'_run3.dat'),
              op.join(data_path,fname+'_run4.dat'),
              op.join(data_path,fname+'_run5.dat')]
    (tc,sigtc,mu2s) = cl.linearWithErrors(fnames,6,dt=0.5,fullreturn=True,
                                            plotstats=[op.join(data_path,
                                            fname+'dummyfull4'),'t (0.5)',
                                            '$\mu_2$',['o','x','^','v','s']])
    (tct,R2t) = cl.linearFit(0.5*np.arange(6),mu2s[:,2])
    npt.assert_almost_equal(2/tct,0.5091,4)
    npt.assert_almost_equal(R2t,0.6727,4)
    actualMu2s = np.array([[1,1,1,1,1,1],[1,1,1,3/2,1,11/10],
                           [3/2,3/2,2,2,3/2,17/10],[2,2,2,2,2,2],
                           [2,4,2,4,2,14/5],[4,4,2,4,2,16/5]])
    npt.assert_array_almost_equal(actualMu2s,mu2s)
    
def test_linearFit_martini():
    """ test on already analyzed data """
    fbase = 'martiniSizes'
    fnames = [op.join(data_path,fbase+'1.dat'),
              op.join(data_path,fbase+'2.dat'),
              op.join(data_path,fbase+'3.dat'),
              op.join(data_path,fbase+'4.dat'),
              op.join(data_path,fbase+'5.dat')]
    (tc,sigtc,mu2s) = cl.linearWithErrors(fnames,8001,dt=0.05,
                                            fullreturn=True,
                                            plotstats=[op.join(data_path,fbase)
                                            ,'t (0.05 ns)','$\mu_2$',
                                            ['o','x','^','v','s']])
    c = 44 * (10**-3) / (1*10**24)#44 mMol concentration
    K = 2/(tc*c*sconst.Avogadro)
    fid = open(op.join(data_path,fbase+'.dat'),'w')
    fid.write("tc is: {0} +/- {1}\n".format(tc,sigtc))
    fid.write("K is: {0}".format(K))
    fid.close()
    assert (K >= 10) and (K <= 24)

def test_nlFit():
    """ Sanity check to make sure that the nonlinear fit does what's expected
    Make sure it regurgitates the correct parameters for data drawn directly
    from a function of the correct form and that it does pretty well on
    artificially noisy data for the same function
    """
    lmbda = -1.
    mu2_0 = 2.
    tc = 10.
    ts = 1.*np.arange(100)
    mu2s = mu2_0*(1+2*ts/tc)**(1/(1-lmbda))
    mu2s = np.reshape(mu2s,[len(mu2s),1])
    (tca,stca,sse,lmbdaa,slmbdaa) = cl.nonlinearFit(ts,mu2s)
    npt.assert_almost_equal(tc,tca)
    npt.assert_almost_equal(lmbda,lmbdaa)

def test_nlFit_noise():
    """ Sanity check to make sure that the nonlinear fit does what's expected
    Make sure it regurgitates the correct parameters for artificially noisy
    data
    """
    lmbda = -1.
    mu2_0 = 2.
    tc = 10.
    ts = 1.*np.arange(100)
    runs = 5
    mu2s = np.zeros([len(ts),runs])
    fbase = 'nlfitNoise'
    print(op.join(data_path,fbase))
    #pdb.set_trace()
    for i in range(runs):
        noise = np.random.randn(len(ts))
        mu2s[:,i] = mu2_0*(1+2*ts/tc)**(1/(1-lmbda)) + noise
    (tca,stca,sse,lmbdaa,slmbdaa) = cl.nonlinearFit(ts,mu2s,plotstats=\
                                                [op.join(data_path,fbase),
                                                't','$\mu_2$',
                                                ['o','x','v','^','s']])
    print("found tc: {0}".format(tca))
    print("found lambda: {0} +/- {1}".format(lmbdaa,slmbdaa))
    assert (lmbdaa - slmbdaa < lmbda) and (lmbdaa + slmbdaa > lmbda)
    assert (tca > 1) and (tca < 100)
    #npt.assert_approx_equal(tca,tc)
    #npt.assert_approx_equal(lmbdaa,lmbda)

def test_nlFit_martini():
    """ Run nl fit on 378 molecule Martini and produce plots and predictions
    to check
    """
    fbase = 'martiniSizes'
    fnames = [op.join(data_path,fbase+'1.dat'),
              op.join(data_path,fbase+'2.dat'),
              op.join(data_path,fbase+'3.dat'),
              op.join(data_path,fbase+'4.dat'),
              op.join(data_path,fbase+'5.dat')]
    (tc,sigtc,sse,lmbda,siglmbda) = \
    cl.nonlinearWithErrorsFromFile(fnames,8001,dt=0.05,
                                   plotstats=[op.join(data_path,fbase+'NL')
                                   ,'t (0.05 ns)','$\mu_2$',
                                   ['o','x','^','v','s']])
    fid = open(op.join(data_path,fbase+'NL.dat'),'w')
    fid.write('tc: {0} +/- {1}\n'.format(tc,sigtc))
    fid.write('lmbda: {0} +/- {1}\n'.format(lmbda,siglmbda))
    fid.close()
    lint = [lmbda-siglmbda,lmbda+siglmbda]
    alint = [-0.1956-0.1189,-0.1956+0.1189]
    assert (tc < 10) and (tc > 0.1)
    assert min(max(lint),max(alint)) > max(min(lint),min(alint))
if __name__ == '__main__':
    #test_getSizesFromFile() 
    test_linearFit()      
    #test_linearFit_martini()    
    #test_nlFit()                           
