# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 08:18:15 2017

@author: rachael

Compute the correlation integral over the COMs of peptides.

"""
from __future__ import absolute_import, division, print_function
from time import time
import clustering as cl
import os
import os.path as op
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mdtraj as md
#compute corrdims for final frame for each run
import pdb
save_path=SSS
data_path=save_path
#Matlab setup

plt.ioff()
font = {'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
runs = 1

ats = {'contact':17,'optical':12}
#molno = 4
dt = 1.0
emax = 294 #maximum length scale to compute correlation integral on
estep = 0.147 #distance steps to compute correlation integral at
skip = 100 #compute correlation integral at every skip steps
totalsteps = 4000
combeadtype = 'E'
markers = ['o','x','^','v','s']
fname = 'md_whole.xtc'
tpr = 'md_dummy.tpr'
topfile = 'after_eq_whole.gro'
start = time()

cemats = np.zeros([int(emax/estep),1+runs])
corrfig = plt.figure()
corrax = corrfig.add_subplot(111)
apermol = 33 #number of atoms or beads per molecule


for t in range(int(totalsteps/skip)):
    frame = md.load_frame(fname,t,top=topfile)
    na = frame.top.n_atoms
    nmols = na / apermol
    if not (na % apermol):
        raise(ValueError,"Does not divide cleanly into molecules.")
    comlist = np.zeros((int(nmols),3))
    for moli in range(len(nmols)):
        molframe = frame.atom_slice(range((moli*apermol),((moli+1)*apermol)))
        comlist[moli,:] = md.compute_center_of_mass(molframe)[0]
    cemat = cl.corrcalc(comlist,emax,estep)
    corrax.plot(np.log(cemat[0,:]),np.log(cemat[1,:]),markers[0])
    cemats[:,0] = cemat[0,:]
    cemats[:,1] = cemat[1,:]
    corrax.grid('on')
    corrax.set_xlabel(r'$\log(\epsilon/\epsilon_0)$ $(d^*)$')
    corrax.set_ylabel(r'$ \log(C(\epsilon))$')
    corrfig.savefig(op.join(save_path,'corrcalc'+str(t)), 
                bbox_inches='tight')
    corrfi = open(op.join(save_path,'corrcalc'+str(t)+'.dat'),'w')
    for e in range(np.shape(cemats)[0]):
    	for runi in range(np.shape(cemats)[1]):
            corrfi.write('{0} '.format(cemats[e,runi]))
        corrfi.write('\n')
    corrfi.close()
end = time()

print("Time to compute correlation integral: ",end-start)
