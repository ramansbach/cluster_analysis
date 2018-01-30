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
save_path=SSS
data_path=save_path
#Matlab setup

plt.ioff()
font = {'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)


runs = 5

ttotal = 999
ttotals = {'contact':ttotal,'optical':ttotal,'aligned':ttotal}
tstart = 10

ats = {'contact':17,'optical':12,'aligned':6}
#molno = 4
molno = 10648
cut=BBB
c1 = float(cut)
c1 = max(1.1,(c1/100.)*1.1225+0.1)
cs={'contact':c1,'optical':0.35,'aligned':0.35}
cutoff = {'contact':1.0*1.0,'optical':0.35*0.35,'aligned':0.35*0.35}
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



run = 0
mu2s = {'contact':np.zeros([ttotal,runs]),
        'optical':np.zeros([ttotal,runs]),
        'aligned':np.zeros([ttotal,runs])}
start = time()



for ctype in ['contact','optical']:
    #pdb.set_trace()
    cszNames = [op.join(save_path,fbase + 'cut'+str(cs[ctype])+ str(runi+1) + ctype + '-sizes.dat') \
                for runi in range(runs)]
    for run in range(runs):
       # pdb.set_trace()
	ccsizes = cl.getSizesFromFile([cszNames[run]],ttotal)
        mu2 = np.zeros(np.shape(ccsizes)[0])
	for m in range(np.shape(ccsizes)[0]):
	    mu2[m] = cl.massAvSize(ccsizes[m,:])
	mu2s[ctype][:,run] = mu2 

end = time()
print("Time to get mu2 from file: ",end-start)
figall = plt.figure()
axall = figall.add_subplot(111)     
figeach = plt.figure()
axseach = {}
cid = 0
start = time()
finsize = dict()
sfinsize = dict()
for ctype in ['contact','optical']:
   axseach[ctype] = figeach.add_subplot(3,1,cid+1)
   mu2sc = mu2s[ctype]
   ymax = np.max(mu2sc[10:len(mu2sc),:])
   mu2smean = np.mean(mu2s[ctype],axis=1)
   mu2sstd = np.std(mu2s[ctype],axis=1)
   finsize[ctype] = mu2smean[len(mu2smean)-1]
   sfinsize[ctype] = mu2sstd[len(mu2sstd)-1]
   runl, = axall.plot(dt*np.arange(0,ttotal),mu2smean,linewidth=2,
                      color=colors[ctype])
   
   mu2final = mu2smean[len(mu2smean)-1]
   if ctype == 'contact':
       axall.set_ylim([0,ymax])
       tend = max(np.argmin(abs(mu2smean[tstart:len(mu2smean)]-0.97*10648))+tstart,tstart+10)
       #pdb.set_trace()
   axseach[ctype].set_ylim([0,ymax])
   runl.set_label(ctype)
   axseach[ctype].plot(dt*np.arange(ttotal),mu2smean,linewidth=2,
                       color=colors[ctype])
   axall.fill_between(np.arange(ttotal),mu2smean-mu2sstd,mu2smean+mu2sstd,
                      facecolor=colors[ctype],alpha=0.3)
   axseach[ctype].fill_between(np.arange(ttotal),mu2smean-mu2sstd,
                               mu2smean+mu2sstd,facecolor=colors[ctype],
                               alpha=0.3)
   axseach[ctype].set_xlabel(r'$t/\tau^*$')
   axseach[ctype].set_ylabel(r'$ \langle \mu_2 \rangle$')
   axseach[ctype].grid('on')
   fmu2s = open(op.join(save_path,fbase + '-' + ctype + '-mu2s.dat'),'w')
   #pdb.set_trace()
   for i in range(np.shape(mu2s[ctype])[0]):
       for j in range(np.shape(mu2s[ctype])[1]):
           fmu2s.write('{0} '.format(mu2s[ctype][i][j]))
       fmu2s.write('\n')   
   fmu2s.close()
   cid += 1
axall.set_xlabel(r'$t$/$\tau^*$')
axall.set_ylabel(r'$ \langle \mu_2  \rangle$')
handles, labels = axall.get_legend_handles_labels()
lgd = axall.legend(handles, labels, loc='upper center', 
                    bbox_to_anchor=(0.5,-0.2))
axall.grid('on')
figall.savefig(op.join(save_path,fbase+'-mu2s-all'), 
               bbox_extra_artists=(lgd,), 
                bbox_inches='tight')

figeach.savefig(op.join(save_path,fbase+'-mu2s-each'), 
                bbox_extra_artists=(lgd,), 
                bbox_inches='tight')
plt.close('all')
end = time()
print("Time to plot mu2s: ",end-start)
#Smoluchowski fitting
start = time()
for ctype in ['contact','optical']:
    cszNames = [op.join(save_path,fbase + 'cut'+str(cs[ctype])+str(runi+1) + ctype + '-sizes.dat') \
            for runi in range(runs)]
    (tc,sigtc) = cl.linearWithErrors(cszNames,ttotals[ctype],tstart=tstart,dt=dt,
                                        plotstats=[op.join(save_path,
                                        fbase+'smol-linear-'+ctype),
                                        r'$t/\tau^*$',
                                        '$\mu_2$',['o','x','^','v','s']],
					tend=tend)
    (nltc,nlsigtc,sse,lmbda,siglmbda) = \
    cl.nonlinearWithErrorsFromFile(cszNames,ttotals[ctype],dt=dt,tstart=tstart,
                               plotstats=[op.join(save_path,
                                                  fbase+'smol-NL-'+ctype)
                               ,'t (0.05 ns)','$\mu_2$',
                               ['o','x','^','v','s']],tend=tend)
    ftcs = open(op.join(save_path,fbase+'smol-data-'+ctype+'.dat'),'w')
    ftcs.write('#tc sigtc nltc nlsigtc lmbda siglmbda sse finsize sigfinsize\n')
    ftcs.write('{0} {1} {2} {3} {4} {5} {6} {7} {8}\n'.format(tc,sigtc,nltc[0],nlsigtc,
                                                  lmbda,siglmbda,sse,finsize[ctype],
						  sfinsize[ctype]))
                                                  
    ftcs.close()
end = time()
print("Time to fit Smol: ",end-start)
