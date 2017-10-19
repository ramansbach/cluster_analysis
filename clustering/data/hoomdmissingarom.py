# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:28:04 2017

@author: rachael
HOOMD single molecule with one aromatic bead misplaced.
"""

import hoomd,imp
from hoomd import *
from hoomd import md
import numpy as np

import gsd.hoomd

beadMass = 1
beadR = 0.5
lbeadR = 0.125
sticky_theta = 10.*(np.pi/180.) #angle down the main sphere of the little spheres
seed = 123451
sigslj = 2*beadR #shifted LJ sigma is basically particle diameter
rcutslj = sigslj
rcutlb = 2.0
siglb = 2*beadR / 2.**(1./6.) #diameter of particle is roughly the LJ min, rm = 2^(1/6) * sigma
sigls = 2*lbeadR / 2.**(1./6.)
siglbb = siglb
#initial rough estimates from Martini oriented dimer PMF's, lseps = 10.0, lbeps = 4.5
lseps=1.85
lbeps=2.0
lbb=0.0
offset=0.025
context.initialize()
system = init.create_lattice(unitcell = hoomd.lattice.sc(a=8, type_name='E'), n=[1,1,1]) 
rigid = md.constrain.rigid()
lbeadR-=offset
system.particles.types.add('E2')
system.particles.types.add('LB')
system.particles.types.add('LS')
rigid.set_param('E',positions=[(-beadR,0,0),(beadR,0,0),(-3*beadR,0,0),
                               (3*beadR,0,0),
(0,(beadR-lbeadR)*np.cos(sticky_theta),(beadR-lbeadR)*np.sin(sticky_theta)),
(0,-(beadR-lbeadR)*np.cos(sticky_theta),(beadR-lbeadR)*np.sin(sticky_theta)),
(beadR,(beadR-lbeadR)*np.cos(sticky_theta),(beadR-lbeadR)*np.sin(sticky_theta)),
(beadR,-(beadR-lbeadR)*np.cos(sticky_theta),(beadR-lbeadR)*np.sin(sticky_theta)),
(-beadR,(beadR-lbeadR)*np.cos(sticky_theta),(beadR-lbeadR)*np.sin(sticky_theta)),
(-beadR,-(beadR-lbeadR)*np.cos(sticky_theta),(beadR-lbeadR)*np.sin(sticky_theta)),
(0,(lbeadR-beadR)*np.cos(sticky_theta),-(beadR-lbeadR)*np.sin(sticky_theta)),
(0,-(beadR-lbeadR)*np.cos(sticky_theta),-(beadR-lbeadR)*np.sin(sticky_theta)),
(beadR,(beadR-lbeadR)*np.cos(sticky_theta),-(beadR-lbeadR)*np.sin(sticky_theta)),
(beadR,-(beadR-lbeadR)*np.cos(sticky_theta),-(beadR-lbeadR)*np.sin(sticky_theta)),
(-beadR,(beadR-lbeadR)*np.cos(sticky_theta),-(beadR-lbeadR)*np.sin(sticky_theta)),
(-beadR,-(beadR-lbeadR)*np.cos(sticky_theta),-(beadR-lbeadR)*np.sin(sticky_theta))],
types=['E2','E2','LB','LB','LS','LS','LS','LS','LS','LS','LS','LS','LS','LS',
       'LS','LS']) 

lbeadR+=offset
rigid.create_bodies()

groupR = group.rigid_center()
groupLB = group.type('LB')
groupLS = group.type('LS')
groupBB = (group.type('E') or group.type('E2'))


 #set masses and diameters of beads
for p in groupBB:
    p.mass = beadMass
    p.diameter = 2*beadR

for p in groupLB:
    p.mass = 3.75*beadMass
    p.diameter = 2*beadR

for p in groupLS:
    p.mass = 0.0
    p.diameter = 2*lbeadR

dump.gsd(filename='missingarom1.gsd',period=None,group=group.all())
