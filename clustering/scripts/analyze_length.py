"""
Created on Fri Oct 13 07:55:15 2017

@author: Rachael Mansbach

Script to run after analyze_clusters_serial.py, which computes the distribution
of lengths of contact and optical clusters, where "length" means the longest
distance between the COMs of two molecules in a cluster unwrapped over the
periodic boundary conditions.

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
c1 = max(1.1,(c1/100.)*1.1225+0.1)
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
    cidNameC = op.join(data_path,fbase + 'cut'+str(cs['contact']) + str(lind+1) + 'contact' + '-CIDs.dat')
    cidNameO = op.join(data_path,fbase + 'cut'+str(cs['optical']) + str(lind+1) + 'optical' + '-CIDs.dat')
    for Clsnap in Syst.clsnaps['contact']:
        Clsnap.setClusterIDFromFile(cidNameC)
    for Clsnap in Syst.clsnaps['optical']:
        Clsnap.setClusterIDFromFile(cidNameO)
    ldfname = op.join(data_path,fbase+'ldistrib'+str(lind+1))
    Syst.getLengthDistribution('contact',cutoff['contact'],box,writegsd=None,
                              writeldistrib=ldfname+'_C.dat')
    Syst.getLengthDistribution('optical',cutoff['optical'],box,writegsd=None,
                              writeldistrib=ldfname+'_O.dat')

    lind += 1
end = time()
if rank == 0:
    print("Time to get length distributions: ",end-start)
