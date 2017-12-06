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
import clustering as cl
import gsd.hoomd
import os.path as op
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
#data_path = '/home/rachael/Analysis_and_run_code/analysis/cluster_analysis/clustering/data'
save_path = SSS
data_path=save_path
#Matlab setup

plt.ioff()
font = {'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

runs = 1

ttotal = 199
tstart = 0
ats = {'contact':17,'optical':12,'aligned':6}
#molno = 4
molno = 10648
c1=0.7
c2=0.35
c3=0.35
cs={'contact':c1,'optical':c2,'aligned':c3}
cutoff = {'contact':c1*c1,'optical':c2*c2,'aligned':c3*c3}
compairs = np.array([[0,6],[1,7],[2,8],[3,9],[4,10],[5,11]])
#compairs = np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]])
molnolabel = 10000
AAdlabel = AAA
SCdlabel = SCSCSC
BBdlabel = BBB
dt = 1.0
emax = 73.5
estep = 0.147
atype = u'LS'
#atype = 'AB'
combeadtype = 'E'
colors = {'contact':'red','optical':'blue','aligned':'olive'}

fbase = 'mols'+str(molnolabel)+'_' + str(AAdlabel)+'-'\
        +str(SCdlabel)+'-'+str(BBdlabel)+'_short_run'

#fbase = 'dummyfull4_run'
fnames = []
for i in range(runs):
    fname = op.join(data_path,fbase + str(i+1) + '.gsd')
    fnames.append(fname)
start = time()    
Systs = [cl.SnapSystem(gsd.hoomd.open(fname),ats,molno,cutoff,
                       compairs=compairs,ttotal=ttotal,tstart=tstart,
                       atype=atype) for fname in fnames]
end = time()
if rank == 0:
    print("Time to setup clusters: ",end-start)
    

#Find all cluster IDs     
start = time()             
for Syst in Systs:
    Syst.get_clusters_mpi('contact',ttotal=ttotal)
    Syst.get_clusters_mpi('optical',ttotal=ttotal)
    Syst.get_clusters_mpi('aligned',ttotal=ttotal)
end = time()
if rank == 0:
    print("Time to get clusters: ",end-start)

if rank == 0:
    run = 0
    mu2s = {'contact':np.zeros([ttotal,runs]),
            'optical':np.zeros([ttotal,runs]),
            'aligned':np.zeros([ttotal,runs])}
    start = time()
    for Syst in Systs:
        for ctype in ['contact','optical','aligned']:
            #write out cluster sizes and cluster IDs at each time step
            cidName = fbase + 'cut'+str(cs[ctype]) + str(run+1) + ctype + '-CIDs.dat'
            cszName = fbase + 'cut' + str(cs[ctype]) + str(run+1) + ctype + '-sizes.dat'
            Syst.writeCIDs(ctype,op.join(save_path,cidName))
            Syst.writeSizes(ctype,op.join(save_path,cszName))
            #compute mass-averaged cluster size versus time
            #pdb.set_trace()
            
        run += 1
    end = time()
    print("Time to write out clusters: ",end-start)
