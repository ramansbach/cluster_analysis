# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 07:55:15 2017

@author: Rachael Mansbach

Script to do (serial) data analysis run using the cluster module

Works for HOOMD type and should be used with setupscripts.sh for correct
population of variables such as SSS, BBB, AAA, SCSCSC

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


runs = 1 #number of independent simulations

ttotal = 1 #total number of timesteps
tstart = 0 #what time to start at, can start t > 0 to eliminate equilibration
ats = {'contact':17,'optical':12} #number of molecules associated with analysis of a monomer
molno = 10240 #total number of molecules in the system
c1=float(BBB) #cutoff for contact clusters (in nm)
c2=0.35 #cutoff for optical clusters (in nm)
c1 = max(1.1,(c1/100.)*1.1225+0.1)
cs={'contact':c1,'optical':c2}
cutoff = {'contact':c1*c1,'optical':c2*c2}
compairs = np.array([[0,6],[1,7],[2,8],[3,9],[4,10],[5,11]])#indices of optical beads
molnolabel = 10000 #label of beginning of file name
AAdlabel = AAA #A bead parameter label
SCdlabel = SCSCSC #SC bead parameter label
BBdlabel = BBB #BB bead parameter label
dt = 1.0 #assumed timestep
atype = u'LS' #optical/aligned bead type
colors = {'contact':'red','optical':'blue'}

typelist = [u'EA',u'EB']

fbase = 'mols'+str(molnolabel)+'_' + str(AAdlabel)+'-'\
        +str(SCdlabel)+'-'+str(BBdlabel)+'_short_run' #template of .gsd file name

fnames = [fbase + r + '.gsd' for r in runs]
start = time()    
traj = gsd.hoomd.open(fnames[0])
box = traj[0].configuration.box[0:3]
Systs = [cl.SnapSystem(fname,ats,molno,cutoff,
                       compairs=compairs,ttotal=ttotal,tstart=tstart,
                       atype=atype,het=True,typelist=typelist) for fname in fnames]
end = time()
print("Time to setup clusters: ",end-start)
    

#Find all cluster IDs     
start = time()
lind = 0             
for Syst in Systs:
    Syst.get_clusters_serial('contact',box,lcompute=None)
    Syst.get_clusters_serial('optical',box,lcompute=None)
    lind += 1
end = time()
print("Time to get clusters: ",end-start)

run = 0
mu2s = {'contact':np.zeros([ttotal,runs]),
            'optical':np.zeros([ttotal,runs])}
start = time()
for Syst in Systs:
    for ctype in ['contact','optical']:
        #write out cluster sizes and cluster IDs at each time step
        cidName = fbase + 'cut'+str(cs[ctype]) + str(run+1) + ctype + '-CIDs.dat'
        cszName = fbase + 'cut' + str(cs[ctype]) + str(run+1) + ctype + '-sizes.dat'
        imixName = fbase + 'cut' + str(cs[ctype]) + str(run+1) + ctype + '-imix.dat'
        Syst.writeCIDs(ctype,op.join(save_path,cidName))
        Syst.writeSizes(ctype,op.join(save_path,cszName))
        #compute intermix
        Syst.writeIntermix(ctype,op.join(save_path,imixName))
            
    run += 1
end = time()
print("Time to write out clusters and computer intermix: ",end-start)
