# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 07:55:15 2017

@author: Rachael Mansbach

Script to do a full (MPI) data analysis run using the cluster module

Description of necessary events:

1) Find all cluster IDs
2) Write out cluster sizes and cluster IDs at each time step
3) Plot Mass-averaged cluster size of contact, aligned, and optical clusters
both separately and in the same plot, including standard deviation over runs
and save raw mu2 data
4) Compute linear and nonlinear Smoluchowski fits & plot for contact, optical,
and aligned clusters
5) Compute, plot, and save data for the correlation integral of the final 
snapshot
"""
from __future__ import absolute_import, division, print_function
from mpi4py import MPI
from time import time
from shutil import move
from os import remove
import clustering as cl
import gsd.hoomd
import os.path as op
import numpy as np
save_path = SSS
data_path=save_path

runs = range(5)

ttotal = 399
tstart = 0
ts = np.arange(tstart,ttotal)
ats = 17
molno = 10648
molnolabel = 10000
AAdlabel = AAA
SCdlabel = SCSCSC
BBdlabel = BBB
idMiss = 10
idPartner = 11
idNotMiss = 4
idNotPartner = 5
fbase = 'mols'+str(molnolabel)+'_' + str(AAdlabel)+'-'\
        +str(SCdlabel)+'-'+str(BBdlabel)+'_run'

start = time()    
for i in runs:
    fname = op.join(data_path,fbase + str(i+1) + '.gsd')
    foutname = op.join(data_path,'temp.gsd')
    cl.fixMisplacedArom(fname,foutname,idMiss,idPartner,idNotMiss,idNotPartner,molno,ats,ts)
    remove(fname)
    move(foutname,fname)
end = time()
print("Time to rewrite files with missing aromatics: ",end-start)
