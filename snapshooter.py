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
from conf import ifplot, rsphere, sigmascale, mass1

if(ifplot):
    import plots

def keyshow(filename):
    '''
    showing the list of keys (entries) in a given data file
    '''
    f = h5py.File(filename,'r')
    print(list(f.keys()))
    f.close()

def plotnth(filename, nstep):
    '''
    plot a given time step of a given data file. To list the available nsteps (integer values), use keyshow(filename).
    If "ifplot" is off, makes an ascii map instead (useful for remote calculations)
    '''
    global rsphere
    outdir=os.path.dirname(filename)
    f = h5py.File(filename,'r')
    params=f["params"]
    nlons=params.attrs["nlons"] ; nlats=params.attrs["nlats"] ; omega=params.attrs["omega"] 
    x = Spharmt(int(nlons),int(nlats),int(old_div(nlons,3)),rsphere,gridtype='gaussian')
    lons1d = x.lons # (2.*np.pi/np.double(nlons))*np.arange(nlons)
    clats1d = np.sin(x.lats) # 2.*np.arange(nlats)/np.double(nlats)-1.
    slats1d = np.cos(x.lats) # 2.*np.arange(nlats)/np.double(nlats)-1.
    lons,lats = np.meshgrid(lons1d, x.lats)
    lonsDeg=lons*180./np.pi ; latsDeg=lats*180./np.pi
    rsphere=params.attrs["rsphere"] ; grav=params.attrs["grav"] # ; kappa=params.attrs["kappa"]
    omegaNS=params.attrs["omega"]
    data=f["cycle_"+str(nstep).rjust(6, '0')]
    vortg=data["vortg"][:] ; divg=data["divg"][:] ; ug=data["ug"][:] ; vg=data["vg"][:] ; t=data.attrs["t"]
    sig=data["sig"][:] ; energy=data["energy"][:] ; beta=data["beta"][:] ; diss=data["diss"][:] ; accflag=data["accflag"][:]
    qminus=data["qminus"][:]
    f.close()
    press=energy* 3. * (1.-beta/2.)
    # ascii output:
    fmap=open(filename+'_map'+str(nstep)+'.dat', 'w')
    step=3
    nth,nphi=np.shape(lats)
    fmap.write("# map with step = "+str(step)+"\n")
    fmap.write("# t="+str(t)+"\n")
    fmap.write("# format: lats lons sigma ug vg E diss accflag\n")
    for kth in np.arange(0,nth, step):
        for kphi in np.arange(0,nphi, step):
            fmap.write(str(lats[kth,kphi])+" "+str(lons[kth,kphi])+" "+str(sig[kth,kphi])+" "
                       +str(ug[kth,kphi])+" "+str(vg[kth,kphi])+" "
                       +str(energy[kth,kphi])+" "+str(diss[kth,kphi])+" "
                       +str(accflag[kth,kphi])+"\n")
    fmap.flush()
    fmap.close()
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
        skx = 8 ; sky=8 # we do not need to output every point; these are the steps for the output in two dimensions
        xx = nd.filters.gaussian_filter(xx, old_div(skx,2.), mode='constant')*500./vvmax
        yy = nd.filters.gaussian_filter(yy, old_div(sky,2.), mode='constant')*500./vvmax
        tbottom=(50.59*((1.-beta)*energy*sigmascale/mass1)**0.25)
        teff=(qminus*sigmascale/mass1)**0.25*3.64 # effective temperature in keV
        # plots.snapplot(lonsDeg, latsDeg, sig, accflag, vortg-2.*omegaNS*np.sin(lats), xx, yy, [skx,sky], outdir=outdir) # geographic maps
        plots.snapplot(lonsDeg, latsDeg, sig, accflag, qminus, xx, yy, [skx,sky], outdir=outdir) # geographic maps

# multiple diagnostic maps for making movies
def multireader(nmin, nmax, infile):

    outdir=os.path.dirname(infile)
    ndigits=np.long(np.ceil(np.log10(nmax))) # number of digits
    
    for k in np.arange(nmax-nmin)+nmin:
        plotnth(infile, k)
        os.system('cp '+outdir+'/snapshot.png '+outdir+'/shot'+str(k).rjust(ndigits, '0')+'.png')
        os.system('cp '+outdir+'/northpole.png '+outdir+'/north'+str(k).rjust(ndigits, '0')+'.png')
        os.system('cp '+outdir+'/southpole.png '+outdir+'/south'+str(k).rjust(ndigits, '0')+'.png')
        os.system('cp '+outdir+'/snapshot.eps '+outdir+'/shot'+str(k).rjust(ndigits, '0')+'.eps')
        os.system('cp '+outdir+'/northpole.eps '+outdir+'/north'+str(k).rjust(ndigits, '0')+'.eps')
        os.system('cp '+outdir+'/southpole.eps '+outdir+'/south'+str(k).rjust(ndigits, '0')+'.eps')
        os.system('cp '+outdir+'/sgeff.eps '+outdir+'/sgeff'+str(k).rjust(ndigits, '0')+'.eps')
        print('shot'+str(k).rjust(ndigits, '0')+'.png')
        
