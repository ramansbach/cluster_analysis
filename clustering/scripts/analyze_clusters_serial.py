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
from time import time
import clustering as cl
import gsd.hoomd
import os.path as op
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
save_path = SSS
data_path=save_path
#Matlab setup

plt.ioff()
font = {'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)


runs = 1

ttotal = 799
tstart = 0
ats = {'contact':17,'optical':12,'aligned':6}
#molno = 4
molno = 10648
c1=float(BBB)
c2=0.35
c3=0.35
c1 = max(0.5,(c1/100.)*1.1225)
cs={'contact':c1,'optical':c2,'aligned':c3}
cutoff = {'contact':c1*c1,'optical':c2*c2,'aligned':c3*c3}
compairs = np.array([[0,6],[1,7],[2,8],[3,9],[4,10],[5,11]])
molnolabel = 10000
AAdlabel = AAA
SCdlabel = SCSCSC
BBdlabel = BBB
dt = 1.0
rank=0
emax = 73.5
estep = 0.147
atype = u'LS'
combeadtype = 'E'
colors = {'contact':'red','optical':'blue','aligned':'olive'}

fbase = 'mols'+str(molnolabel)+'_' + str(AAdlabel)+'-'\
        +str(SCdlabel)+'-'+str(BBdlabel)+'_short_run'

fnames = []
ldfnames = []
for i in range(runs):
    fname = op.join(data_path,fbase + str(i+1) + '.gsd')
    fnames.append(fname)
    ldfname = op.join(data_path,fbase+'ldistrib'+str(i+1))
    ldfnames.append(ldfname)
start = time()    
traj = gsd.hoomd.open(fname)
box = traj[0].configuration.box[0:3]
Systs = [cl.SnapSystem(traj,ats,molno,cutoff,
                       compairs=compairs,ttotal=ttotal,tstart=tstart,
                       atype=atype) for fname in fnames]
end = time()
if rank == 0:
    print("Time to setup clusters: ",end-start)
    

#Find all cluster IDs     
start = time()
lind = 0             
for Syst in Systs:
    ldfnameC = ldfnames[lind]+'_C.dat'
    ldfnameO = ldfnames[lind]+'_O.dat'
    ldfnameA = ldfnames[lind]+'_A.dat'
    Syst.get_clusters_serial('contact',box,lcompute=ldfnameC)
    Syst.get_clusters_serial('optical',box,lcompute=ldfnameO)
    Syst.get_clusters_serial('aligned',box,lcompute=ldfnameA)
    lind += 1
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
