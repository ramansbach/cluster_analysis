from __future__ import absolute_import, division, print_function
import numpy as np

import pdb
from cfractald import corrDim, getCOMs
__all__ = ['corrcalc','getCOMsPy','getCOMs','getCOMnumpy','methodL','fit2']

def getCOMnumpy(poslist,masslist):
    #return com coordinates of a single molecule, written in python using numpy
    poslist3 = poslist.reshape([len(poslist)/3,3])
    com = np.dot(masslist,poslist3)/masslist.sum()
    return com
    
def getCOMsPy(poslist,masslist,ats):
    #return the center of mass of each molecule in a position list of peptides
    N = int(len(poslist)/3/ats) #total number of molecules
    comlocs = np.zeros(3*N)
    for i in range(N):
        Rcom = getCOMnumpy(poslist[3*ats*i:3*ats*i+3*ats],masslist)
        comlocs[3*i:3*i+3] = Rcom
    return comlocs

def getcomsPy( pos, coms, masslist, beads, mols): 
    pos = np.reshape(pos,[1,3 * beads * mols])[0]
    coms = np.reshape(coms,[1,3*mols])[0]
    for i in range(mols):
        
            X = 0;
            Y = 0;
            Z = 0;
            M = 0;
            for j in range(beads):
                    #if i == 1:
                    #    pdb.set_trace()
                    x = masslist[j] * pos[beads*i + 3*j];
                    y = masslist[j] * pos[beads*i + 3*j+1];
                    z = masslist[j] * pos[beads*i + 3*j+2];
                    X+=x;
                    Y+=y;
                    Z+=z;
                    M += masslist[j];
                
            X/=M;
            Y/=M;
            Z/=M;
            coms[3*i] = X;
            coms[3*i + 1] = Y;
            coms[3*i + 2] = Z;
    coms = np.reshape(coms,[mols,3]) 
    return coms
    
class LineFit(object):
    """ A simple container for slope, intercept, and variance of a line fit
    
    Attributes
    ----------
    slope: slope of line fit
    intercept: intercept of line fit
    slopeVar: variance of the slope
    interVar: variance of the intercept    
    
    """
    def __init__(self,s,i,sV,iV):
        self.slope = s
        self.intercept = i
        self.slopeVar = sV
        self.interVar = iV


def fit2(x,y,w,i):
    """ Helper function for L method. Takes independent and dependent data
    and performs two linear fits, splitting the data at index i.
    
    Parameters
    ----------
    x: numpy vector
        the independent variable
    y: numpy vector
        the dependent variable
    w: numpy vector
        weights for the fit
    i: int
        the index to split on
        
    Returns
    -------
    rmset: float
        total root mean square error
    line1: LineFit object
        first line fit
    line2: LineFit object
        second line fit
    
    """
    pf1 = np.polyfit(x[0:i+1],y[0:i+1],1,w=w[0:i+1],full=True)
    p1 = pf1[0]
    sse1 = pf1[1][0]
    v1 = i - 2
    mse1 = sse1/v1
    rmse1 = np.sqrt(mse1)

    pf2 = np.polyfit(x[i:len(x)],y[i:len(x)],1,w=w[i:len(x)],full=True)
    p2 = pf2[0]
    sse2 = pf2[1][0]
    v2 = len(x) - i - 2
    mse2 = sse2/v2
    rmse2 = np.sqrt(mse2)

    (p1,cov1) = np.polyfit(x[0:i+1],y[0:i+1],1,w=w[0:i+1],cov=True)
    (p2,cov2) = np.polyfit(x[i:len(x)],y[i:len(x)],1,w=w[i:len(x)],cov=True)
    
    line1 = LineFit(p1[0],p1[1],cov1[0][0],cov1[1][1])
    line2 = LineFit(p2[0],p2[1],cov2[0][0],cov2[1][1])
    
    #pdb.set_trace()
    rmset = ((i-1.)/(len(x)-1.))*rmse1 + ((len(x)-i)/(len(x)-1.))*rmse2    
    return (rmset,line1,line2)
    
def methodL(x,y,w,xstart=None,xend=None):
    """ Performs the L method on y vs x data
    
    Parameters
    ----------
    x: numpy vector
        the independent variable
    y: numpy vector
        the dependent variable
    w: numpy vector
        weights for the fit
    xstart: float
        the x value at which to start, if not starting from the initial one
        if it is None, defaults to starting from the beginning
    xend: float
        the x value at which to end, if not ending at the final value
        if it is None, defaults to ending at the final value
        
    Returns
    -------
    xjunct: float
        the x value at which to break between the two lines
    line1: a simple struct containing slope and intercept
    line2: a simple struct containing slope and intercept
    totalErr: float
        total root mean squared error
        
    Notes
    -----
    The L method simply finds the spot where to break the data such that two
    lines fitted to the data return the minimum possible root mean square 
    error, which is computed as
    
    RMSET = ((j-1)/(length(vals)-1))*RMSE1 
            + ((length(vals)-j)/(length(vals)-1))*RMSE2,
            
    where RMSE1 is the root mean square error of line 1, RMSE2 is the root 
    mean square error of line 2, j is the point of switching over, and vals
    is the set of dependent variables
    """
    if xstart is None:
        xstart = min(x)
    if xend is None:
        xend = max(x)
    istart = np.abs(x-xstart).argmin()
    iend = np.abs(x-xend).argmin()
    x = x[istart:(iend+1)]
    y = y[istart:(iend+1)]
    rmset = np.inf
    for i in range(3,len(x)-3):
        
        (rmsetj,linei1,linei2) = fit2(x,y,w,i)
        #pdb.set_trace()
        if rmsetj < rmset:
            rmset = rmsetj
            xjunct = i
            line1 = linei1
            line2 = linei2
    return (xjunct,line1,line2,rmset)
            
def corrcalc(coms,emax,estep,fname=None):
    """ Given the locations of the beads in a snapshot, find and write out
    the C(e) function
    
    Parameters
    ----------
    coms: numpy array [M x 3]
        M = molno
        array contains COM positions for all molecules in a snapshot
    emax: double
        distance out to which to compute the correlation integral
    estep: double
        size of steps for correlation integral calculation
    fname: file name to write to
        if None, do not write file
        
    Returns
    -------
    corrInt: numpy array [N x 2]
        matrix representing the correlation integral versus radius (eps)
    
    """
    
    (epsnap,cdsnap) = corrDim(coms,emax,estep) 
    if fname is not None:
        f = open(fname,'w')
        for i in range(len(cdsnap)):
            f.write('{0}\t{1}\n'.format(epsnap[i],cdsnap[i]))
        f.close()
    return np.array([epsnap,cdsnap])