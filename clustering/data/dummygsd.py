# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:19:06 2017

@author: rachael

Create a dummy gsd file with 8 molecules of 2 beads each in 8 snapshots
Used to check cluster analysis
"""
import gsd.hoomd

pN = 16
ptypes = ['A']
ptypeid = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
pbody = [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
pbox = [10,10,10,0,0,0]
pdiameter = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
                        0.5,0.5,0.5]

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


p01 = [2.25,0,2]
p02 = [1.75,0,2]      
p11 = [0.25,0,2]
p12 = [-0.25,0,2]
p21 = [2.25,2,2]
p22 = [1.75,2,2]
p31 = [0.25,2,2]
p32 = [-0.25,2,2]
p41 = [2.25,0,0]
p42 = [1.75,0,0]
p51 = [0.25,0,0]
p52 = [-0.25,0,0]
p61 = [2.25,2,0]
p62 = [1.75,2,0]
p71 = [0.25,2,0]
p72 = [-0.25,2,0]
spos0 = [p01,p02,p11,p12,p21,p22,p31,p32,p41,p42,p51,p52,p61,p62,p71,p72]    

p11 = [0.25,-0.25,2]
p12 = [-0.25,-0.25,2]
p21 = [0.25,0.25,2]
p22 = [-0.25,0.25,2]
p41 = [-0.75,0,0]
p42 = [-0.75,0.5,0]
p71 = [-0.75,1.0,0]
p72 = [-0.75,1.5,0]    
spos1 = [p01,p02,p11,p12,p21,p22,p31,p32,p41,p42,p51,p52,p61,p62,p71,p72] 

p31 = [0,0.75,2]
p32 = [0,1.25,2]
p41 = [-0.75,0,0]
p42 = [-1.25,0,0]
p71 = [-0.25,0.5,0]
p72 = [-0.25,1,0]
spos2 = [p01,p02,p11,p12,p21,p22,p31,p32,p41,p42,p51,p52,p61,p62,p71,p72] 

p61 = [2.25,0,1.5]
p62 = [2.25,0,1.0]
spos3 = [p01,p02,p11,p12,p21,p22,p31,p32,p41,p42,p51,p52,p61,p62,p71,p72] 

p01 = [0.75,0.25,2]
p02 = [0.75,-0.25,2]
p61 = [1.25,-0.25,2]
p62 = [1.75,-0.25,2]
spos4 = [p01,p02,p11,p12,p21,p22,p31,p32,p41,p42,p51,p52,p61,p62,p71,p72] 
spos5 = [p01,p02,p11,p12,p21,p22,p31,p32,p41,p42,p51,p52,p61,p62,p71,p72] 

p01 = [1,2,0]
p02 = [1,2,0.5]
p11 = [1,2,1]
p12 = [1,1.5,1]
p21 = [1,1,1]
p22 = [1,0.5,1]
p31 = [1,0,1]
p32 = [1,0,1.5]
p41 = [1,-0.125,2]
p42 = [1,0.125,2]
p51 = [1,0,2.5]
p52 = [1,0,3]
p61 = [1,0.5,2.5]
p62 = [1,0.5,3]
p71 = [1,1,2.75]
p72 = [1,1.4,2.75]
spos6 = [p01,p02,p11,p12,p21,p22,p31,p32,p41,p42,p51,p52,p61,p62,p71,p72] 
spos7 = [p01,p02,p11,p12,p21,p22,p31,p32,p41,p42,p51,p52,p61,p62,p71,p72] 

tdummy = gsd.hoomd.open(name='dummy8.gsd',mode='wb')
tdummy.append(createFrame(0,spos0))
tdummy.append(createFrame(1,spos1))
tdummy.append(createFrame(2,spos2))
tdummy.append(createFrame(3,spos3))
tdummy.append(createFrame(4,spos4))
tdummy.append(createFrame(5,spos5))
tdummy.append(createFrame(6,spos6))
tdummy.append(createFrame(7,spos7))
