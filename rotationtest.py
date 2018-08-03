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
    outdir=os.path.dirname(infile)

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
    err=np.zeros(nsize, dtype=np.double)
    serr=np.zeros(nsize, dtype=np.double)
    tar=np.zeros(nsize, dtype=np.double)
    sig1=np.zeros(np.shape(sig0), dtype=np.double)
    t0=data.attrs["t"] ; tar[0]=t0
    sigfun=interp1d(lons1d, sig0, axis=1, bounds_error=False, kind='nearest')
    fout=open("rtest.dat", "w")
    # all the others
    for k in np.arange(nsize-1)+1:
        data=f[keys[k]]
        sig=data["sig"][:]
        t=data.attrs["t"] ; tar[k]=t
        rotang=(t-t0)*omega
        print("rotation in "+str(rotang)+"rad")
        #        print(np.shape(sig))
        #        print(np.shape(lons1d))
#        print((lons1d+rotang) % (2.*np.pi))
        sig1[:]=sigfun((lons1d-rotang) % (2.*np.pi))
        wfin=np.isfinite(sig1)
        err[k]=(sig/sig1-1.)[wfin].std()
        serr[k]=(sig/sig1-1.)[wfin].mean() # systematic error
        fout.write(str(t-t0)+" "+str(err[k])+" "+str(serr[k])+"\n")
        print("error +-"+str(err[k]))
        print("systematic +-"+str(serr[k]))
        if(ifplot):
            plots.somemap(lons, lats, sig-sig1, outdir+"/err_sigma.png")
        print("entry "+str(keys[k])+" finished")
    if(ifplot):
        plots.sometimes(tar*1e3*tscale, [err, serr], fmt=['r-', 'k-'] , prefix=outdir+'/err')
    fout.close()
    f.close()
