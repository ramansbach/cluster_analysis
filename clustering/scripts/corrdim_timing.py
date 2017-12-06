# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 08:18:15 2017

@author: rachael
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

ats = {'contact':17,'optical':12,'aligned':6}
#molno = 4
molnolabel = 10000
AAdlabel = AAA
SCdlabel = SCSCSC
BBdlabel = BBB
dt = 1.0
emax = 294
estep = 0.147
#atype = 'AB'
combeadtype = 'E'
markers = ['o','x','^','v','s']
fbase = 'mols'+str(molnolabel)+'_' + str(AAdlabel)+'-'\
        +str(SCdlabel)+'-'+str(BBdlabel)+'_short_run'

fnames = []
for i in range(runs):
    fname = op.join(data_path,fbase + str(i+1) + '.gsd')
    fnames.append(fname)

start = time()

cemats = np.zeros([int(emax/estep),1+runs])
corrfig = plt.figure()
corrax = corrfig.add_subplot(111)

for runi in range(runs):
    traj = gsd.hoomd.open(fnames[runi])
    finalFrame = traj[len(traj)-1]
    tind = finalFrame.particles.types.index(combeadtype)
    comlist = finalFrame.particles.position[np.where(finalFrame.particles.typeid==tind)[0]]
    cemat = cl.corrcalc(comlist,emax,estep)
    corrax.plot(np.log(cemat[0,:]),np.log(cemat[1,:]),markers[runi])
    cemats[:,0] = cemat[0,:]
    cemats[:,runi+1] = cemat[1,:]
corrax.grid('on')
corrax.set_xlabel(r'$\log(\epsilon/\epsilon_0)$ $(d^*)$')
corrax.set_ylabel(r'$ \log(C(\epsilon))$')
corrfig.savefig(op.join(save_path,fbase+'-corrcalc'), 
                bbox_inches='tight')
corrfi = open(op.join(save_path,fbase+'-corrcalc.dat'),'w')
for e in range(np.shape(cemats)[0]):
    for runi in range(np.shape(cemats)[1]):
        corrfi.write('{0} '.format(cemats[e,runi]))
    corrfi.write('\n')
corrfi.close()
end = time()

print("Time to compute correlation integral: ",end-start)
