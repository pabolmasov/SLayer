from __future__ import print_function
from __future__ import division
from builtins import str
from past.utils import old_div
import os.path
import numpy as np
import time
import h5py
from spharmt import Spharmt 

from scipy.integrate import trapz
from scipy.interpolate import interp1d

# TODO: global parameters should be read from hdf5 rather than taken from conf
from conf import ifplot
if(ifplot):
    import plots

def comparot(infile):
    '''
    compares two snapshots of a single output file assuming rigid-body rotation
    '''
    f = h5py.File(infile,'r')

    params=f["params"]
    # loading global parameters from the hdf5 file
    nlons=params.attrs["nlons"] ; nlats=params.attrs["nlats"]
    omega=params.attrs["omega"] ; rsphere=params.attrs["rsphere"]
    tscale=params.attrs["tscale"] ; mass1=params.attrs["NSmass"]
    x = Spharmt(int(nlons),int(nlats),int(old_div(nlons,3)),rsphere,gridtype='gaussian')
    lons1d = x.lons ; lats1d = x.lats
    clats1d = np.sin(x.lats) # 2.*np.arange(nlats)/np.double(nlats)-1.
    lons,lats = np.meshgrid(lons1d, np.arccos(clats1d))
    dlons=2.*np.pi/np.size(lons1d) ; dlats=old_div(2.,np.double(nlats))

    keys=list(f.keys())
    nsize=np.size(keys)-1 # last key contains parameters
    print(str(nsize)+" points from "+str(keys[0])+" to "+str(keys[-2]))

    # first snapshot:
    data=f[keys[0]]
    sig0=data["sig"][:]
    err=np.zeros(nsize, dtype=double)
    tar=np.zeros(nsize, dtype=double)
    # all the others
    for k in np.arange(nsize-1)+1:
        data=f[keys[k]]
        sig=data["sig"][:]
        t=data.attrs["t"] ; tar[k]=t
        rotang=t*omega
        print("rotation in "+str(rotang)+"rad")
        sigfun=interp1d(lons, sig, axis=0)
        sig1=sigfun((lons+rotang) % (2.*np.pi))
        err[k]=(sig1/sig0-1.).std()
        if(ifplot):
            plots.somemap(lons, lats, sig1-sig0, outdir+"/err_sigma.png")
    if(ifplot):
        plots.sometimes(tar*1e3*tscale, [err], prefix=outdir+'/errplot.png')
    
