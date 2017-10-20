import matplotlib
import numpy as np
import shtns
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy.ndimage as spin
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
    
    keys=f.keys()
    data=f[keys[nstep]]
    vortg=data["vortg"][:] ; divg=data["divg"][:] ; ug=data["ug"][:] ; vg=data["vg"][:]
    sig=data["sig"][:] ; diss=data["diss"][:]
    f.close()
    
    # velocity
    xx=ug-omega*rsphere*np.cos(lats) ; yy=vg
    vv=np.sqrt(xx**2+yy**2)
    vvmax=vv.max()
    skx = 8 ; sky=16
    xx = spin.filters.gaussian_filter(xx, skx/2., mode='constant')*200./vvmax
    yy = spin.filters.gaussian_filter(yy, sky/2., mode='constant')*200./vvmax

    s0=sig.min() ; s1=sig.max()
    #    s0=0.1 ; s1=10. # how to make a smooth estimate?
    nlev=20
    levs=(s1/s0)**(np.arange(nlev)/np.double(nlev-1))*s0
    
    plt.clf()
    fig=plt.figure()
    plt.contourf(lons, lats, sig,cmap='jet',levels=levs)
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

def multireader(nmax):

    ndigits=np.long(np.ceil(np.log10(nmax))) # number of digits
    
    for k in np.arange(nmax):
        plotnth('out/run.hdf5', k)
        os.system('cp out/snapshot.png out/shot'+str(k).rjust(ndigits, '0')+'.png')
        print 'shot'+str(k).rjust(ndigits, '0')+'.png'
        
