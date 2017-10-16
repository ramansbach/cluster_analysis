from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy.optimize import curve_fit
from scipy.optimize import minimize

plt.ioff()
font = {'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

__all__ = ['linearFit','nonlinearFit','linearWithErrors',
           'nonlinearWithErrorsFromFile','massAvSize',
           'getSizesFromFile']

def linearFit(t,mu2):
    """Perform a linear fit to the mass-averaged cluster size.
    Parameters
    ----------
    t: numpy vector (length T)
        array containing timesteps
    mu2: numpy vector (length T)
        containing mass-averaged cluster size mu2(t)
        
    Returns
    -------
    tc: float
        coagulation time
    R2: float
        R^2 value = 1 - SSres/SStot
    
    Notes
    -----
    Fits mu2(t) = mu2(0) + K * t
    where K = 2/tc
    SSres is returned from linalg.lstsq
    SStot = sum_i(y_i - <y_i>)^2 where y_i is the ith data point
    """
    
    K = np.linalg.lstsq(np.reshape(t,[len(t),1]), 
                         np.reshape(mu2 - mu2[0],[len(mu2),1]))
    tc = 2/K[0][0][0]
    ssres = K[1][0]
    mu2av = np.mean(mu2)
    sstot = ((mu2 - mu2av)*(mu2-mu2av)).sum()
    R2 = 1 - ssres/sstot
    return (tc,R2)
    
def minSSE(t,mu2s,tol=1e-4):
    """ Helper function that finds the tc that minimizes SSE for a fit for a
    set of runs
    
    Parameters
    ----------
    t: numpy vector (length T)
        array containing timesteps
    mu2s: numpy array [T x runs]
        array containing mass-averaged cluster size values for a run
    tol: float 
        tolerance for minimization
    
    Returns
    -------
    tc: float
        the optimal tc
    etc: float
        tolerance of fit (error in tc)
    optsse: float
        total best SSE for tc
    lmbda: float
        the optimal lambda
    elmbda: float
        error in lmbda as the standard deviation of the lmbdas for different
        runs
    """
    sz = np.shape(mu2s)
    def sse(tc):
            """ returns the sum-square-error of the best for lambda to Smol"""
            sse = 0
            def f(t,lmbda):
                y = mu2s[0,run]*(1 + 2*t/tc)**(1/(1-lmbda))
                return y
            for run in range(sz[1]):
                #pdb.set_trace()
                popt,pcov = curve_fit(f,t,mu2s[:,run],
                                      bounds=([-np.inf,2]),
                                      p0=np.array([-1.]))
                
                bestlambda = popt[0]
                ssecurr = (mu2s[:,run] - f(t,bestlambda))\
                     *(mu2s[:,run] - f(t,bestlambda))
                sse += ssecurr.sum()
            return sse
    #pdb.set_trace()
    runresult = minimize(sse,10,tol=1e-4,bounds=[(tol,np.inf)])
    opttc = runresult.x
    optsse = runresult.fun
    lmbdas = np.zeros(sz[1])
    for run in range(sz[1]):
        def newf(t,lmbda):
            """Smoluchowski style fit with set tc"""
            y = mu2s[0,run]*(1 + 2*t/opttc)**(1/(1-lmbda))
            return y
        #pdb.set_trace()
        popt,pcov = curve_fit(newf,t,mu2s[:,run],bounds=([-np.inf,2]),
                              p0=np.array([-1.]))
        lmbda = popt[0]
        lmbdas[run] = lmbda
    return (opttc,tol,optsse,np.mean(lmbdas),np.std(lmbdas))

def nonlinearFit(t,mu2s,plotstats=None,tol=1e-4):
    """ Perform a nonlinear fit to the mass-averaged cluster size.
    
    Parameters
    ----------
    t: numpy vector (length T)
        array containing timesteps
    mu2s: numpy array [T x runs]
        array containing the mass-averaged cluster size at each time step
        for a number of independent runs
    plotstats: array of data for plotting
        None or [plotname,xlabel,ylabel,[marker1,...,markerN]]
        if not none, save a plot of the data with the avg fit and std errors
    tol: float
        tolerance for minimization
        
    Returns
    -------
    tc: float
        coagulation time
    etc: float
        error in the coagulation time
    sse: float
        minimum SSE from searching for best tc
    lmbda: float
        Smoluchowski exponent
    elmbda: float
        error in the Smoluchowski exponent
        
    Notes
    -----
    Performs a fit to mu2(t) = mu2(0)*(1+2t/tc)^(1/(1-lambda))
    Uses scipy.optimize.curve_fit with sigmas in the y data taken from the
    uncertainties computed from the different runs, unless there is only
    one run
    
    Note that by definition, the Smoluchowski exponent must be <=2 
    and the coagulation time must be >=0
    """
    
    sz = np.shape(mu2s)
    if plotstats is not None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(sz[1]):
            #pdb.set_trace()
            runl, = ax.plot(t,mu2s[:,i],plotstats[3][i]
                                        ,fillstyle='none')
        ax.plot(t,np.mean(mu2s,axis=1),linewidth=2)
        
    
    (tc,etc,sse,lmbda,elmbda) = minSSE(t,mu2s,tol)
    def f(t,lmbda):
         y = np.mean(mu2s[0,:])*(1 + 2*t/tc)**(1/(1-lmbda))
         return y
    if plotstats is not None:
        p1 = f(t,lmbda-elmbda)
        p2 = f(t,lmbda+elmbda)
      
        mincurve = np.min(np.array([p1,p2]),axis=0)
        maxcurve = np.max(np.array([p1,p2]),axis=0)
        ax.plot(t,f(t,lmbda),'--',linewidth=2,color="black")
        ax.fill_between(t,mincurve,maxcurve,facecolor='black',alpha=0.3)
        plt.xlabel(plotstats[1])
        plt.ylabel(plotstats[2])
        ax.grid('on')
        fig.savefig(plotstats[0],bbox_inches='tight')
        plt.close()
    return (tc,etc,sse,lmbda,elmbda)

def nonlinearWithErrorsFromFile(fnames,T,dt=1.0,plotstats=None,tstart=0):
    """ Perform a nonlinear fit to a number of different independent sizes
    and find the spread in the fit
    
    Parameters
    ----------
    fnames: list of strings
        all the filenames containing the sizes
    T: int
        number of timesteps
    tstart: int
        timestep to start on, defaults to 0
    dt: float
        how large a timestep is
    fullreturn: bool
        whether to return the full set of mu2s
    plotstats: array of data for plotting
        None or [plotname,xlabel,ylabel,[marker1,...,markerN]]
        if not none, save a plot of the data with the avg fit and std errors
        
    Returns
    -------
    tc: float
        average coagulation time
    etc: float
        error in the average coagulation time
    lmbda: float
        average Smoluchowski exponent
    elmbda: float
        error in Smoluchowski exponent
    """
    mu2s = np.zeros([T,len(fnames)])
    f = 0
    for fname in fnames:
        csizes = getSizesFromFile([fname],T)
        mu2 = [massAvSize(csize) for csize in csizes]
        mu2s[:,f] = mu2
        f+=1
    (tc,etc,sse,lmbda,elmbda) = nonlinearFit(dt*np.arange(tstart,T+tstart),mu2s,plotstats=plotstats)
    return (tc,etc,sse,lmbda,elmbda)
    
def linearWithErrors(fnames,T,dt=1.0,fullreturn=False,plotstats=None,tstart=0):
    """ Perform a linear fit to a number of different independent sizes
    and find the spread in the fit
    
    Parameters
    ----------
    fnames: list of strings
        all the filenames containing the sizes
    T: int
        number of timesteps
    tstart: int
        timestep to start on
    dt: float
        how large a timestep is
    fullreturn: bool
        whether to return the full set of mu2s
    plotstats: array of data for plotting
        None or [plotname,xlabel,ylabel,[marker1,...,markerN]]
        if not none, save a plot of the data with the avg fit and std errors
        
    Returns
    -------
    tc: float
        average coagulation time
    etc: float
        error in the average coagulation time
    """
    tcs = np.zeros(len(fnames))
    i = 0
    if plotstats is not None:
        fig = plt.figure(1)
    if fullreturn:
        mu2s = np.zeros([T,len(fnames)+1])
    for fname in fnames:
        csizes = getSizesFromFile([fname],T)
        mu2 = [massAvSize(csize) for csize in csizes]
        (tc,R2) = linearFit(dt * np.arange(tstart,T+tstart),mu2)
        if plotstats is not None:
            ax = fig.add_subplot(111)
            #pdb.set_trace()
            runl, = ax.plot(dt * np.arange(tstart,T+tstart),mu2,plotstats[3][i],fillstyle='none')
            
            runl.set_label('$R^2$ = {0}'.format(round(R2,2)))
            #ax.legend()
        if fullreturn:
            mu2s[:,i] = mu2
        
        tcs[i] = tc
        i+=1
    csizes = getSizesFromFile(fnames,T)
    mu2 = [massAvSize(csize) for csize in csizes]
    (tc,R2) = linearFit(dt * np.arange(tstart,T+tstart),mu2)
    if fullreturn:
        mu2s[:,len(fnames)] = mu2
    if plotstats is not None:
        ax.plot(dt * np.arange(tstart,T+tstart,0.1),mu2[0] + (2/tc)*dt*np.arange(0,T,0.1),
                 linestyle='--',linewidth=2,color='black')
        #plt.tight_layout()
        Ks = 2/tcs
        sigma = np.std(Ks)
        K = 2/tc
        ax.fill_between(dt*np.arange(tstart,T+tstart,0.1),mu2[0] \
                        + (K-sigma)*dt*np.arange(tstart,T+tstart,0.1),
                         mu2[0]+(K+sigma)*dt*np.arange(tstart,T+tstart,0.1),
                        facecolor='black',alpha=0.3)
        plt.xlabel(plotstats[1])
        plt.ylabel(plotstats[2])
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='upper center', 
                        bbox_to_anchor=(0.5,-0.2))
        ax.grid('on')
        fig.savefig(plotstats[0], bbox_extra_artists=(lgd,), 
                    bbox_inches='tight')
        plt.close()
    if fullreturn:
        if plotstats is not None:
            f = 0
            for fname in fnames:
                mu2curr = mu2s[:,f]
                plt.figure()
                plt.plot(dt*np.arange(tstart,T+tstart,0.1),
                         mu2curr[0]+(2/tcs[f])*dt*np.arange(tstart,T+tstart,0.1),
                         linestyle='--',linewidth=2,color='black')
                plt.plot(dt*np.arange(tstart,T+tstart),mu2curr,linewidth=2)
                plt.plot(dt*np.arange(tstart,T+tstart),mu2curr,'o')
                plt.xlabel(plotstats[1])
                plt.ylabel(plotstats[2])
                plt.title('run '+str(f))
                plt.savefig(plotstats[0]+str(f))
                plt.close()
                f+=1
        return (tc,np.std(tcs),mu2s)
    else:
        return (tc,np.std(tcs))
    
def massAvSize(csizes):
    """
    Given a cluster sizes list, returns the mass averaged cluster size
    of the snapshot
    
    Parameters
    ----------
    csizes: numpy array as returned by idsToSizes
    
    Returns
    -------
    mu2: float, the mass-averaged cluster size
    """
    umass,counts = np.unique(csizes,return_counts=True)
    mu2 = (umass*counts).sum() / (counts).sum()
    return mu2
    
def getSizesFromFile(fnames,T):
    """ create an array from a file or files of csizes data

    Parameters
    ----------
    fnames: list
        can be any number of files containing csizes data at a list of 
        timesteps, but all files must have the same number of rows & columns
    T: total number of timesteps
        
    Returns
    -------
    csizes: numpy array [T x M]
        an array where each number represents another molecule in a cluster
        of the given size
    """
    fname1 = fnames[0]
    fid1 = open(fname1)
    line1 = fid1.readline()
    M = len(line1.split())
    fid1.close()
    csizes = np.zeros([T,M * len(fnames)])
    f = 0
    for fname in fnames:
        fid = open(fname)
        flines = fid.readlines()
        fid.close()
        for t in range(T):
            fline = flines[t]
            spline = [float(m) for m in fline.split()]
            try:
                csizes[t,(f*M):(f*M+M)] = spline
            except:
                print("fug")
        f += 1
    return csizes