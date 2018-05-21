# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:10:34 2018

@author: rachael

Script for XTC size trajectory stuff

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


def analyze_clusters():
    data_path = '/home/rachael/coarsegraining/CG/active_learning/martini-assembly/dfmi/4_production'
    #where to look for data
    
    trj = op.join(data_path,'md_whole.xtc')
    tpr = op.join(data_path,'md_dummy.tpr')
    molno = 100 #number of molecules
    ats = 33 #number of atoms per molecule
    tstart = 0
    ttotal = 4000
    comIDs = np.array([[9,10,11],[13,14,15],[18,19,20]]) #index of beads comprising aromatics
    cldict = {'contact':0.5*0.5,'optical':0.7*0.7} #cutoffs for different cluster types
    start = time.time()
    outGro = op.join(data_path,'conf')
    syst = cl.SnapSystem(trj,ats,molno,cldict,compairs=comIDs, 
                ttotal=ttotal,tstart=tstart,tpr=tpr,outGro=outGro)
    end = time.time()
    print("Time to setup: "+str(end-start)+"\n")
    start = time.time()
    syst.get_clusters_serial('contact',np.array([[24.25935],[24.25935],
                                                 [24.25935]]))
    end = time.time()
    print("Time to get contact: "+str(end-start)+"\n")
    start = time.time()
    syst.get_clusters_serial('optical',np.array([[19.97738],[19.97738],
                                                 [19.97738]]))
    end = time.time()
    print("Time to get optical: "+str(end-start)+"\n")
    syst.writeCIDs('contact','pull-test-contact-CIDs-dfmi.dat')
    syst.writeCIDs('optical','pull-test-optical-CIDs-dfmi.dat')
    syst.writeSizes('contact','pull-test-contact-sizes-dfmi.dat')
    syst.writeSizes('optical','pull-test-optical-sizes-dfmi.dat')
    
if __name__ == "__main__":
    analyze_clusters()

