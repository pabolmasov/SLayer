import matplotlib
import numpy as np
import shtns
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage as nd
import time
from spharmt import Spharmt 
import os
import h5py
import plots

#proper LaTeX support and decent fonts in figures 
rc('font',**{'family':'serif','serif':['Times']})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

def keyshow(filename):
    f = h5py.File(filename,'r')
    print f.keys()
    f.close()
    
def plotnth(filename, nstep):

    f = h5py.File(filename,'r')
    params=f["params"]
    nlons=params.attrs["nlons"] ; nlats=params.attrs["nlats"] ; omega=params.attrs["omega"] 
    lons1d = (2.*np.pi/np.double(nlons))*np.arange(nlons)
    clats1d = 2.*np.arange(nlats)/np.double(nlats)-1.
    lons,lats = np.meshgrid(lons1d-1., np.arcsin(clats1d))
    lons*=180./np.pi ; lats*=180./np.pi
    rsphere=params.attrs["rsphere"] ; grav=params.attrs["grav"] # ; kappa=params.attrs["kappa"]
    
    data=f["cycle_"+str(nstep).rjust(6, '0')]
    vortg=data["vortg"][:] ; divg=data["divg"][:] ; ug=data["ug"][:] ; vg=data["vg"][:] ; t=data.attrs["t"]
    sig=data["sig"][:] ; energy=data["energy"][:] ; diss=data["diss"][:] ; accflag=data["accflag"][:]
    f.close()
    
    # velocity
    xx=ug-0.*omega*rsphere*np.cos(lats) ; yy=-vg
    xxmean=xx.mean(axis=1) ;    yymean=yy.mean(axis=1)
    sigmean=sig.mean(axis=1)
    sig1=np.zeros(sig.shape, dtype=np.double)
    for k in np.arange(nlons):
        sig1[:,k]=sig[:,k] # -sigmean[:]
#        xx[:,k]-=xxmean[:]
#        yy[:,k]-=yymean[:]
    vv=np.sqrt(xx**2+yy**2)
    vvmax=vv.max()
    skx = 8 ; sky=16
    xx = nd.filters.gaussian_filter(xx, skx/2., mode='constant')*500./vvmax
    yy = nd.filters.gaussian_filter(yy, sky/2., mode='constant')*500./vvmax
    plots.snapplot(lons, lats, sig, accflag, xx, yy, [skx,sky]) # geographic maps

    kappa=0.34
    geff=-grav+(ug**2+vg**2)/rsphere
    radgeff=sig*diss*kappa
    plots.sgeffplot(sig, grav, geff, radgeff) # Eddington violation plot

    # Reynolds's stress (I know the Pythonic way to pronounce thiss!)
    plots.reys(lons, lats, sig, ug, vg, energy,rsphere)
    
def multireader(nmin, nmax):

    ndigits=np.long(np.ceil(np.log10(nmax))) # number of digits
    
    for k in np.arange(nmax-nmin)+nmin:
        plotnth('firstrun/run.hdf5', k)
        os.system('cp out/snapshot.png out/shot'+str(k).rjust(ndigits, '0')+'.png')
        os.system('cp out/northpole.png out/north'+str(k).rjust(ndigits, '0')+'.png')
        os.system('cp out/southpole.png out/south'+str(k).rjust(ndigits, '0')+'.png')
        os.system('cp out/snapshot.eps out/shot'+str(k).rjust(ndigits, '0')+'.eps')
        os.system('cp out/northpole.eps out/north'+str(k).rjust(ndigits, '0')+'.eps')
        os.system('cp out/southpole.eps out/south'+str(k).rjust(ndigits, '0')+'.eps')
        os.system('cp out/sgeff.eps out/sgeff'+str(k).rjust(ndigits, '0')+'.eps')
        print 'shot'+str(k).rjust(ndigits, '0')+'.png'
        
