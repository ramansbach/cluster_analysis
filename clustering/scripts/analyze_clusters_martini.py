"""
Example of running optical and contact cluster stuff on gromacs file
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
data_path ='/home/rachael/coarsegraining/CG/active_learning/martini-assembly/dfmi/4_production'  #folder where trajectory is
#trajectory should not have any water
#this can be done as follows:
#gmx trjconv -f after_eq.gro -o after_eq_whole.gro -pbc whole -s md.tpr  
#choose protein
#gmx trjconv -f md.xtc -o md_whole.xtc -pbc whole -s md.tpr
#choose protein
#grompp -f md_dummy.mdp -c after_eq_whole.gro -p CG_dfmi_prot.top -o md_dummy.tpr
#where md_dummy is the same as the mdp file except with water removed, same
#for the topology file 

def test_trajectory():
    """
    Try running on an xtc trajectory (from a pull simulation)
    """
    trj = op.join(data_path,'md_whole.xtc')
    tpr = op.join(data_path,'md_dummy.tpr')
    molno = 100
    ats = 33
    tstart = 0
    ttotal = 4000
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