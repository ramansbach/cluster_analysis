# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:03:20 2017

@author: rachael
Create a dummy gsd file with 4 molecules of 17 beads each in 6 snapshots
Used to check cluster analysis
"""

from __future__ import absolute_import, division, print_function
import gsd.hoomd
import numpy as np

def quatMultiply(q1,q2):
    """Returns a quaternion that is a composition of two quaternions
    
    Parameters
    ----------
    q1: 1 x 4 numpy array
        representing a quaternion
    q2: 1 x 4 numpy array
        representing a quatnernion
        
    Returns
    -------
    qM: 1 x 4 numpy array
        representing a quaternion that is the rotation of q1 followed by
        the rotation of q2
        
    Notes
    -----
    q2 * q1 is the correct order for applying rotation q1 and then
    rotation q2
    """
    
    Q2 = np.array([[q2[0],-q2[1],-q2[2],-q2[3]],[q2[1],q2[0],-q2[3],q2[2]],
                  [q2[2],q2[3],q2[0],-q2[1]],[q2[3],-q2[2],q2[1],q2[0]]])
    qM = np.dot(Q2,q1)
    return qM

def createOneMol(comPos,qrot):
    """Returns a molecule, which is a list of typeids and positions
    
    Parameters
    ----------
    comPos: 1 x 3 numpy array
        position of the center of mass
    qrot: 1 x 4 numpy array
        quaternion representing the orientation of the molecule
        always rotate about the x axis for consistency
        
    Returns
    -------
    pos: 17 x 3 numpy array
        represents the positions of all the beads in the molecule
    typeinds: 1 x 17 numpy array
        represents the molecule types of all the beads in the molecule
        large beads, LB = 0
        aromatic beads, AB = 1
    diams: 1 x 17 numpy array
        gives the diameters of all the beads
    
    Notes
    -----
    For consistency, track the pairs of indices going into the aromatics in
    the order
    [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]
    small beads are at a radius of 0.4 and an angle of theta = 10 degrees
    """
    sRad = 0.475
    th = 10 *(np.pi/180)
    pos = np.zeros([17,3])
    typeinds = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1])
    diams = np.zeros(17)
    for i in range(len(diams)):
        if typeinds[i] == 0:
            diams[i] = 1.0
        else:
            diams[i] = 0.125
    baseLocations = np.array([[0.,0.,0.],[0.5,0.,0.],[-0.5,0.,0.],
                              [1.5,0.,0.],[-1.5,0.,0.],
                              [-0.5,sRad*np.cos(th),sRad*np.sin(th)],
                              [-0.5,sRad*np.cos(th),-sRad*np.sin(th)],
                              [0.,sRad*np.cos(th),sRad*np.sin(th)],
                              [0.,sRad*np.cos(th),-sRad*np.sin(th)],
                              [0.5,sRad*np.cos(th),sRad*np.sin(th)],
                              [0.5,sRad*np.cos(th),-sRad*np.sin(th)],
                              [-0.5,-sRad*np.cos(th),sRad*np.sin(th)],
                              [-0.5,-sRad*np.cos(th),-sRad*np.sin(th)],
                              [0.,-sRad*np.cos(th),sRad*np.sin(th)],
                              [0.,-sRad*np.cos(th),-sRad*np.sin(th)],
                              [0.5,-sRad*np.cos(th),sRad*np.sin(th)],
                              [0.5,-sRad*np.cos(th),-sRad*np.sin(th)]])
    pos = np.zeros(np.shape(baseLocations))
    for rind in range(np.shape(baseLocations)[0]):
        r = baseLocations[rind,:]
        q = qrot[0]
        qvec = qrot[1:4]
        rp = r + 2. * q * np.cross(qvec,r) \
             + 2. * np.cross(qvec,np.cross(qvec,r))
        pos[rind,:] = rp
    pos += comPos
    return(pos,typeinds,diams)

def createSnapshot(coms,qrots,step):
    """Create HOOMD snapshot with the given molecules
    
    Parameters
    ----------
    coms: N x 3 numpy array
        the positions of the centers of masses of the N molecules in the system
    qrots: N x 4 numpy array
        the orientations of the N molecules in the system
    step: int
        timestep
        
    Returns
    -------
    snap: HOOMD snapshot
    """
    snap = gsd.hoomd.Snapshot()
    molno = np.shape(coms)[0]
    beadsPerMol = 17
    snap.particles.N = molno * beadsPerMol
    snap.configuration.step = step
    snap.configuration.box = [20,20,20,0,0,0]
    snap.particles.types = ['LB','AB']
    snap.particles.position = np.zeros([molno * beadsPerMol,3])
    snap.particles.body = np.zeros(molno * beadsPerMol)
    snap.particles.typeid = np.zeros(molno * beadsPerMol)
    snap.particles.diameter = np.zeros(molno * beadsPerMol)
    for moli in range(molno):
        snap.particles.body[(moli * beadsPerMol): \
                            (moli * beadsPerMol + beadsPerMol)] \
                            = moli * np.ones(beadsPerMol)
        (pos,typeinds,diams) = createOneMol(coms[moli,:],qrots[moli,:])
        snap.particles.position[(moli * beadsPerMol): \
                                (moli * beadsPerMol + beadsPerMol)] = pos
        snap.particles.typeid[(moli * beadsPerMol): \
                              (moli * beadsPerMol + beadsPerMol)] = typeinds
        snap.particles.diameter[(moli * beadsPerMol): \
                                (moli * beadsPerMol + beadsPerMol)] = diams
    return snap

if __name__ == "__main__":
    #quaternion = (cos(th/2),sin(th/2) omhat) => rotation of th about omhat
    molno = 4
    ats = 17
    pN = molno * ats
    df4 = gsd.hoomd.open('dummyfull4_run1.gsd','wb')
    """Snapshot 1"""
    coms1 = np.array([[0.,0.,0.],[0.,3.,0.],[0.,0.,-3],[0.,3.,-3.]])
    qrot1 = np.array([[1.,0.,0.,0.],[1.,0.,0.,0.],[1.,0.,0.,0],[1.,0.,0.,0.]])
    snap1 = createSnapshot(coms1,qrot1,0)
    df4.append(snap1)
    
    """Snapshot 2"""
    coms2 = np.array([[0.,0.,0.],[-1.5,2.5,0.],[0.,3.,-3.],[1.,3.5,-3.5]])
    qrot2 = np.array([[1.,0.,0.,0.],[np.cos(np.pi/4),0.,0.,np.sin(np.pi/4)],
                      [np.cos(np.pi/4),0.,0.,np.sin(np.pi/4)],
                      quatMultiply(np.array([np.cos(np.pi/4),0.,
                                             np.sin(np.pi/4),0.]),
                                   np.array([np.cos(np.pi/4),0.,0.,
                                             np.sin(np.pi/4)]))])
    snap2 = createSnapshot(coms2,qrot2,1)
    df4.append(snap2)
    """Snapshot 3"""
    coms3 = np.array([[0.,0.,0.],[0.,1.,0.],[-4.5,-1.0,0.],[-4.,0.,0.]])
    qrot3 = np.array([[1.,0.,0.,0.],[-1.,0.,0.,0.],
                      [1.,0.,0.,0.],[1.,0.,0.,0.]])
    snap3 = createSnapshot(coms3,qrot3,2)
    df4.append(snap3)
    """Snapshot 4"""
    coms4 = np.array([[0.,0.,0.],[0.,1.,0.],[-4.,0.,0.],[-4.,1.,0.]])
    qrot4 = qrot3
    snap4 = createSnapshot(coms4,qrot4,3)
    df4.append(snap4)
    """Snapshot 5"""
    coms5 = np.array([[0.,0.,0.],[0.,1.,0.],[0.5,2.,-0.5],[0.5,3.,-0.5]])
    qrot5 = np.array([[1.,0.,0.,0.],[-1.,0.,0.,0.],
                      [np.cos(np.pi/4),0.,np.sin(np.pi/4),0.],
                      [np.cos(np.pi/4),0.,-np.sin(np.pi/4),0.]])
    snap5 = createSnapshot(coms5,qrot5,4)
    df4.append(snap5)
    """Snapshot 6"""
    coms6 = np.array([[0.,0.,0.],[0.,-0.5,np.sqrt(3)/2],
                      [0.,0.5,np.sqrt(3)/2],[0.,0.,-1.]])
    qrot6 = np.array([[np.cos(np.pi/4),np.sin(np.pi/4),0.,0.],
                      [np.cos(np.pi/12),-np.sin(np.pi/12),0.,0.],
                      [np.cos(np.pi/12),np.sin(np.pi/12),0.,0.],
                      [np.cos(np.pi/4),np.sin(np.pi/4),0.,0.]])
    snap6 = createSnapshot(coms6,qrot6,5)
    df4.append(snap6)
    df4_2 = gsd.hoomd.open('dummyfull4_run2.gsd','wb')
    df4_2.append(snap2)
    df4_2.append(snap2)
    df4_2.append(snap3)
    df4_2.append(snap5)
    df4_2.append(snap6)
    df4_2.append(snap6)
    df4_3 = gsd.hoomd.open('dummyfull4_run3.gsd','wb')
    df4_3.append(snap2)
    df4_3.append(snap2)
    df4_3.append(snap4)
    df4_3.append(snap4)
    df4_3.append(snap5)
    df4_3.append(snap5)
    df4_4 = gsd.hoomd.open('dummyfull4_run4.gsd','wb')
    df4_4.append(snap1)
    df4_4.append(snap3)
    df4_4.append(snap4)
    df4_4.append(snap4)
    df4_4.append(snap6)
    df4_4.append(snap6)
    df4_5 = gsd.hoomd.open('dummyfull4_run5.gsd','wb')
    df4_5.append(snap2)
    df4_5.append(snap2)
    df4_5.append(snap3)
    df4_5.append(snap5)
    df4_5.append(snap5)
    df4_5.append(snap5)
    
