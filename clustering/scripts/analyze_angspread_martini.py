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

def run_ang_spread():
    """
    Try running on an xtc trajectory (from a pull simulation)
    """
    trj = op.join(data_path,'md_whole.xtc')
    tpr = op.join(data_path,'md_dummy.tpr')
    molno = 100
    ats = 33
    tstart = 0
    ttotal = 4000
    cainds = range(12,23)
    oainds = range(0,3)
    cfname = op.join(data_path,'angle-spread-contact.dat')
    ofname = op.join(data_path,'angle-spread-optical.dat')
    comIDs = np.array([[12,13,14],[16,17,18],[20,21,22]])
    cldict = {'contact':0.5*0.5,'optical':0.7*0.7}
    

    cfname = op.join(data_path,'pull-test-contact-CIDs.dat')
    ofname = op.join(data_path,'pull-test-optical-CIDs.dat')    
    
    start = time.time()
    syst = cl.SnapSystem(trj,ats,molno,cldict,compairs=comIDs, 
                ttotal=ttotal,tstart=tstart,tpr=tpr)
    end = time.time()
    print("Time to setup: "+str(end-start)+"\n")
    start = time.time()
    syst.get_clusters_from_file('contact',cfname)
    end = time.time()
    print("Time to get contact: "+str(end-start)+"\n")
    start = time.time()
    syst.get_clusters_from_file('optical',ofname)
    end = time.time()
    print("Time to get optical: "+str(end-start)+"\n")
    
    start = time.time()
    syst.writeAngSpread('contact',cfname,cainds)
    syst.writeAngSpread('optical',ofname,oainds)
    end = time.time()
    print("Time to get angle spread: "+str(end-start))
    
    
if __name__ == "__main__":
    run_ang_spread()