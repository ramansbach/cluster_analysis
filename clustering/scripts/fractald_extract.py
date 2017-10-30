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
runs = 5

#molno = 4
molnolabel = 10000
AAdlabel = AAA
SCdlabel = SCSCSC
BBdlabel = BBB

xstart = 0.7
xend = 4.0

markers = ['o','x','^','v','s']
fbase = op.join(data_path, 'mols'+str(molnolabel)+'_' + str(AAdlabel)+'-'\
        +str(SCdlabel)+'-'+str(BBdlabel)+'_run-corrcalc.dat')
outbase = op.join(data_path, 'mols'+str(molnolabel)+'_' + str(AAdlabel)+'-'\
        +str(SCdlabel)+'-'+str(BBdlabel)+'_run-fractaldim.dat')
pngname = op.join(data_path, 'mols'+str(molnolabel)+'_' + str(AAdlabel)+'-'\
        +str(SCdlabel)+'-'+str(BBdlabel)+'_corrcalcdimfit')
fid = open(fbase)
clines = fid.readlines()
fid.close()

start = time()

cemats = np.zeros([len(clines),1+runs])
cind = 0
for cline in clines:
    cemats[cind,:] = [float(centry) for centry in cline.split()]
    cind += 1

corrfig = plt.figure()
corrax = corrfig.add_subplot(111)
meanY = np.mean(np.log(cemats[:,2:(runs+1)]),axis=1)
#w = 1/np.std(cemats[:,2:(runs+1)],axis=1)
w = np.ones(len(clines))
(ijunct,l1,l2,terr) = cl.methodL(np.log(cemats[:,0]),meanY,w,xstart=xstart,xend=xend)
for runi in range(runs):

    corrax.plot(np.log(cemats[:,0]),np.log(cemats[:,runi+1]),markers[runi])
    
corrax.grid('on')
istart = np.abs(np.log(cemats[:,0])-xstart).argmin()
iend = np.abs(np.log(cemats[:,0])-xend).argmin()
x = np.log(cemats[:,0])
line1 = x*l1.slope + l1.intercept
line2 = x*l2.slope + l2.intercept
line1min = x*(l1.slope-l1.slopeVar) + l1.intercept - l1.interVar
line1max = x*(l1.slope+l1.slopeVar) + l1.intercept + l1.interVar
line2min = x*(l2.slope-l2.slopeVar) + l2.intercept - l2.interVar
line2max = x*(l2.slope+l2.slopeVar) + l2.intercept + l2.interVar

corrax.plot(x[istart:ijunct],line1[istart:ijunct],'--',color='blue',
            linewidth=2)
corrax.plot(x[ijunct:iend],line2[ijunct:iend],'--',color='red',
            linewidth=2)
#corrax.fill_between(x[istart:ijunct],line1min[istart:ijunct],
#                    line1max[istart:ijunct],color='black',alpha=0.3) 
#corrax.fill_between(x[ijunct:iend],line2min[ijunct:iend],
#                    line2max[ijunct:iend],color='black',alpha=0.3)              
corrax.set_xlabel(r'$\log(\epsilon/\epsilon_0)$ $(d^*)$')
corrax.set_ylabel(r'$ \log(C(\epsilon))$')
corrfig.savefig(pngname, 
                bbox_inches='tight')
corrfi = open(outbase,'w')
corrfi.write('#xbreak slope1 error slope2 error intercept1 error intercept2 error totalError\n')
corrfi.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n'.format(cemats[ijunct,0],
             l1.slope,l1.slopeVar,l2.slope,l2.slopeVar,l1.intercept,
             l1.interVar,l2.intercept,l2.interVar,terr))

corrfi.close()
end = time()

print("Time to compute fractal dimensions: ",end-start)
