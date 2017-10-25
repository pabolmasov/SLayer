import matplotlib
import numpy as np
import shtns
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy.ndimage as nd
import time
from spharmt import Spharmt 
import os
import h5py

#proper LaTeX support and decent fonts in figures 
rc('font',**{'family':'serif','serif':['Times']})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

def plotnth(filename, nstep):

    f = h5py.File(filename,'r')
    params=f["params"]
    nlons=params.attrs["nlons"] ; nlats=params.attrs["nlats"] ; omega=params.attrs["omega"] 
    lons1d = (2.*np.pi/nlons)*np.arange(nlons)
    clats1d = 2.*np.arange(nlats)/np.double(nlats)-1.
    lons,lats = np.meshgrid(lons1d, np.arcsin(clats1d))
    lons*=180./np.pi ; lats*=180./np.pi
    rsphere=params.attrs["rsphere"]
    
    data=f["cycle_"+str(nstep).rjust(6, '0')]
    vortg=data["vortg"][:] ; divg=data["divg"][:] ; ug=data["ug"][:] ; vg=data["vg"][:] ; t=data.attrs["t"]
    sig=data["sig"][:] ; diss=data["diss"][:] ; accflag=data["accflag"][:]
    f.close()
    
    # velocity
    xx=ug-omega*rsphere*np.cos(lats) ; yy=vg
    xxmean=xx.mean(axis=1) ;    yymean=yy.mean(axis=1)
    sigmean=sig.mean(axis=1)
    sig1=np.zeros(sig.shape, dtype=np.double)
    for k in np.arange(nlons):
        sig1[:,k]=sig[:,k]-sigmean[:]
        xx[:,k]-=xxmean[:]
        yy[:,k]-=yymean[:]
    vv=np.sqrt(xx**2+yy**2)
    vvmax=vv.max()
    skx = 8 ; sky=16
    xx = nd.filters.gaussian_filter(xx, skx/2., mode='constant')*500./vvmax
    yy = nd.filters.gaussian_filter(yy, sky/2., mode='constant')*500./vvmax

    wpoles=np.where(np.fabs(lats)>30.)
    s0=sig[wpoles].min() ; s1=sig[wpoles].max()
    #    s0=0.1 ; s1=10. # how to make a smooth estimate?
    nlev=20
    levs=(s1/s0)**(np.arange(nlev)/np.double(nlev-1))*s0
    
    plt.clf()
    fig=plt.figure()
    plt.contourf(lons, lats, sig1,cmap='jet') #,levels=levs)
    plt.colorbar()
    plt.quiver(lons[::skx, ::sky],
        lats[::skx, ::sky],
        xx[::skx, ::sky], yy[::skx, ::sky],
        pivot='mid',
        units='x',
        linewidth=1.0,
        color='k',
        scale=8.0,
    )
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    fig.set_size_inches(8, 5)
    plt.savefig('out/snapshot.png')
    plt.savefig('out/snapshot.eps')
    plt.close()

    # now let us make a polar plot:
    tinyover=1./np.double(nlons)
    theta=90.*(1.+tinyover)-lats
    plt.clf()
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    #    wnorth=np.where(lats>0.)
    tinyover=1./np.double(nlons)
    ax.contourf(lons*np.pi/180.*(tinyover+1.), theta, sig,cmap='jet',levels=levs)
    ax.contour(lons*np.pi/180.*(tinyover+1.), theta, accflag,colors='w',levels=[0.5])
    ax.set_rticks([20., 40.])
    ax.set_rmax(60.)
    plt.title('N') #, t='+str(nstep))
    plt.tight_layout()
    fig.set_size_inches(4, 4)
    plt.savefig('out/northpole.eps')
    plt.savefig('out/northpole.png')
    plt.close()
    plt.clf()
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    #    wnorth=np.where(lats>0.)
    tinyover=1./np.double(nlons)
    ax.contourf(lons*np.pi/180.*(tinyover+1.), 180.*(1.+tinyover)-theta, sig,cmap='jet',levels=levs)
    ax.contour(lons*np.pi/180.*(tinyover+1.), 180.*(1.+tinyover)-theta, accflag,colors='w',levels=[0.5])
    ax.set_rticks([20., 40.])
    ax.set_rmax(60.)
    plt.tight_layout(pad=2)
    fig.set_size_inches(4, 4)
    plt.title('S') #, t='+str(nstep))
    plt.savefig('out/southpole.eps')
    plt.savefig('out/southpole.png')
    plt.close()

def multireader(nmin, nmax):

    ndigits=np.long(np.ceil(np.log10(nmax))) # number of digits
    
    for k in np.arange(nmax-nmin)+nmin:
        plotnth('out/runOLD.hdf5', k)
        os.system('cp out/snapshot.png out/shot'+str(k).rjust(ndigits, '0')+'.png')
        os.system('cp out/northpole.png out/north'+str(k).rjust(ndigits, '0')+'.png')
        os.system('cp out/southpole.png out/south'+str(k).rjust(ndigits, '0')+'.png')
        os.system('cp out/snapshot.eps out/shot'+str(k).rjust(ndigits, '0')+'.eps')
        os.system('cp out/northpole.eps out/north'+str(k).rjust(ndigits, '0')+'.eps')
        os.system('cp out/southpole.eps out/south'+str(k).rjust(ndigits, '0')+'.eps')
        print 'shot'+str(k).rjust(ndigits, '0')+'.png'
        
