from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import numpy.testing as npt
import pdb
import gsd.hoomd
import sys
import clustering as cl

#from context import clustering as cl
#from context import smoluchowski as smol
from cdistances import conOptDistanceCython,alignDistancesCython
#import imp
#cl = imp.load_source('cl','/home/rachael/Analysis_and_run_code/analysis/cluster_analysis/clustering/clustering.py')
data_path = op.join(cl.__path__[0], 'data')

def test_write_out_frame():
    fname = 'mols8.gsd'
    traj = gsd.hoomd.open(op.join(data_path, fname))
    box = traj[0].configuration.box
    ats = {'contact':17}
    cutoff= 1.1*1.1
    molno = 8
    cldict = {'contact':cutoff}
    syst = cl.SnapSystem(traj,ats,molno,cldict)
    syst.get_clusters_serial('contact',box)
    syst.writeCIDs('contact',op.join(data_path,'mols8cIDs.dat'))
    cIDfile = op.join(data_path,'mols8cIDs.dat')
    cIDfile = open(cIDfile)
    lines = cIDfile.readlines()
    cIDfile.close()
    line = lines[35]
    cIDsf = [float(c) for c in line.split()]
    cIDs = [int(c) for c in cIDsf]
    cl.writeFrameForVMD(cIDs,molno,ats['contact'],
                        op.join(data_path,'testframe35.dat'))