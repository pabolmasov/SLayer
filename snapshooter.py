from __future__ import print_function
from __future__ import division
from builtins import str
from past.utils import old_div
import numpy as np
import shtns
import scipy.ndimage as nd
import time
from spharmt import Spharmt 
import os
import h5py

'''
Post-processing and various post-factum diagnostic outputs
'''
from conf import ifplot, rsphere

if(ifplot):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import rc
    import pylab
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    from mpl_toolkits.mplot3d import Axes3D
    import plots

    #proper LaTeX support and decent fonts in figures 
    rc('font',**{'family':'serif','serif':['Times']})
    rc('mathtext',fontset='cm')
    rc('mathtext',rm='stix')
    rc('text', usetex=True)
    # #add amsmath to the preamble
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

def keyshow(filename):
    '''
    showing the list of keys (entries) in a given data file
    '''
    f = h5py.File(filename,'r')
    print(list(f.keys()))
    f.close()

def plotnth(filename, nstep):
    '''
    plot a given time step of a given data file. To list the available nsteps (integer values), use keyshow(filename) 
    '''
    global rsphere
    outdir=os.path.dirname(filename)
    f = h5py.File(filename,'r')
    params=f["params"]
    nlons=params.attrs["nlons"] ; nlats=params.attrs["nlats"] ; omega=params.attrs["omega"] 
    x = Spharmt(int(nlons),int(nlats),int(old_div(nlons,3)),rsphere,gridtype='gaussian')
    lons1d = x.lons # (2.*np.pi/np.double(nlons))*np.arange(nlons)
    clats1d = np.sin(x.lats) # 2.*np.arange(nlats)/np.double(nlats)-1.
    lons,lats = np.meshgrid(lons1d, x.lats)
    lons*=old_div(180.,np.pi) ; lats*=old_div(180.,np.pi)
    rsphere=params.attrs["rsphere"] ; grav=params.attrs["grav"] # ; kappa=params.attrs["kappa"]
    omegaNS=params.attrs["omega"]
    data=f["cycle_"+str(nstep).rjust(6, '0')]
    vortg=data["vortg"][:] ; divg=data["divg"][:] ; ug=data["ug"][:] ; vg=data["vg"][:] ; t=data.attrs["t"]
    sig=data["sig"][:] ; energy=data["energy"][:] ; beta=data["beta"][:] ; diss=data["diss"][:] ; accflag=data["accflag"][:]
    f.close()
    
    skx = 8 ; sky=8 # we do not need to output every point; these are the steps for the output in two dimensions
    if(ifplot):
        # velocity
        xx=ug ; yy=-vg
        xxmean=xx.mean(axis=1) ;    yymean=yy.mean(axis=1)
        sigmean=sig.mean(axis=1)
        sig1=np.zeros(sig.shape, dtype=np.double)
        for k in np.arange(nlons):
            sig1[:,k]=sig[:,k] # -sigmean[:]
            #        xx[:,k]-=xxmean[:]
            #        yy[:,k]-=yymean[:]
        vv=np.sqrt(xx**2+yy**2)
        vvmax=vv.max()
        xx = nd.filters.gaussian_filter(xx, old_div(skx,2.), mode='constant')*500./vvmax
        yy = nd.filters.gaussian_filter(yy, old_div(sky,2.), mode='constant')*500./vvmax
        plots.snapplot(lons, lats, sig, accflag, xx, yy, [skx,sky], outdir=outdir) # geographic maps

        kappa=0.34
        geff=-grav+old_div((ug**2+vg**2),rsphere)
        radgeff=sig*diss*kappa
        plots.sgeffplot(sig, grav, geff, radgeff, outdir=outdir) # Eddington violation plot
        plots.vortgraph(lats, lons, vortg, divg, sig, energy, omegaNS, lonrange=[0.,360.], outdir=outdir)
        plots.dissgraph(sig, energy, diss, vg**2+(ug)**2, accflag, outdir=outdir)
        # we need a vorticity-divergence graph
        # Reynolds's stress (I know the Pythonic way to pronounce thiss!)
        #    plots.reys(lons, lats, sig, ug, vg, energy,rsphere)

        # thicknesses and Froude number:
        hthick=-5./geff*energy/sig/3./(1.-old_div(beta,2.))
        fru=old_div(ug,np.sqrt(-hthick*geff)) # azimuthal Froude/Mach
        frv=old_div(vg,np.sqrt(-hthick*geff)) # polar Froude/Mach
        plots.somemap(lons, lats, np.sqrt(frv**2), outdir+'/froude.eps')
        plots.somemap(lons, lats, np.log10(old_div(hthick,rsphere)), outdir+'/htor.eps')
    # ascii output:
    #    fmap=open(filename+'_map.dat', 'w')
    
    
def multireader(nmin, nmax, outdir='out'):

    ndigits=np.long(np.ceil(np.log10(nmax))) # number of digits
    
    for k in np.arange(nmax-nmin)+nmin:
        plotnth('firstrun/run.hdf5', k)
        os.system('cp '+outdir+'/snapshot.png '+outdir+'/shot'+str(k).rjust(ndigits, '0')+'.png')
        os.system('cp '+outdir+'/northpole.png '+outdir+'/north'+str(k).rjust(ndigits, '0')+'.png')
        os.system('cp '+outdir+'/southpole.png '+outdir+'/south'+str(k).rjust(ndigits, '0')+'.png')
        os.system('cp '+outdir+'/snapshot.eps '+outdir+'/shot'+str(k).rjust(ndigits, '0')+'.eps')
        os.system('cp '+outdir+'/northpole.eps '+outdir+'/north'+str(k).rjust(ndigits, '0')+'.eps')
        os.system('cp '+outdir+'/southpole.eps '+outdir+'/south'+str(k).rjust(ndigits, '0')+'.eps')
        os.system('cp '+outdir+'/sgeff.eps '+outdir+'/sgeff'+str(k).rjust(ndigits, '0')+'.eps')
        print('shot'+str(k).rjust(ndigits, '0')+'.png')
        
