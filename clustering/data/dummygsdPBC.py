# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:19:06 2017

@author: rachael

Create a dummy gsd file for PBC testing, which is just a single frame with
two 3-molecule rigid bodies in it
"""
import gsd.hoomd
import numpy as np
pN = 6
ptypes = ['A']
ptypeid = [0,0,0,0,0,0]
pbody = [0,0,0,1,1,1]
pbox = [10,10,10,0,0,0]
pdiameter = [0.5,0.5,0.5,0.5,0.5,0.5]

def createFrame(step,position,pN=pN,ptypes=ptypes,ptypeid=ptypeid,pbody=pbody,
                pbox=pbox,pdiameter=pdiameter):
    

    s = gsd.hoomd.Snapshot()
    s.particles.N = pN
    s.configuration.step = step
    s.particles.types = ptypes
    s.particles.typeid = ptypeid
    s.particles.body = pbody 
    s.configuration.box = pbox
    s.particles.position = position
    s.particles.diameter = pdiameter
    return s


spos0 = np.array([0.,0.,4.75,0.,0.,-4.76,0.,0.,-4.25,-0.5,0.,-4.75,
                 -0.5,0.,-4.25,-0.5,0.,-3.75])
tdummy = gsd.hoomd.open(name='dummy2PBC.gsd',mode='wb')
tdummy.append(createFrame(0,spos0))
