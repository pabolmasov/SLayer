from __future__ import print_function
from __future__ import division
from builtins import str
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

def comparetwoshots(run1, n1, run2, n2):
    '''
    simply computes the difference maps and average residuals between two unrelated snapshots
    dimensions should be identical 
    '''
    f1 = h5py.File(run1,'r')
    keys1=list(f1.keys())
    params=f1["params"]
    data1=f1[keys1[n1]]
    sig1=data1["sig"][:]
    rsphere=params.attrs["rsphere"]; nlons=params.attrs["nlons"] ; nlats=params.attrs["nlats"]
    x = Spharmt(int(nlons),int(nlats),int(np.double(nlons)/3.),rsphere,gridtype='gaussian')
    lons1d = x.lons ; lats1d = x.lats
    clats1d = np.sin(x.lats) # 2.*np.arange(nlats)/np.double(nlats)-1.
    lons,lats = np.meshgrid(lons1d, np.arccos(clats1d))
    dlons = 2.*np.pi/np.size(lons1d) ; dlats = (2. / np.double(nlats))
    f2 = h5py.File(run2,'r')
    keys2=list(f2.keys())
    data2=f2[keys2[n2]]
    sig2=data2["sig"][:]
    if(ifplot):
        plots.somemap(lons, lats, sig2-sig1, "diff_sigma.png")
    drel=((sig1-sig2)/(sig1+sig2)).std()
    sdrel=((sig1-sig2)/(sig1+sig2)).mean()
    return drel, sdrel
    
def comparot(infile, nmax = None):
    '''
    compares all the snapshots of a single output file assuming rigid-body rotation 
    '''
    f = h5py.File(infile,'r')
    outdir=os.path.dirname(infile)

    params=f["params"]
    # loading global parameters from the hdf5 file
    nlons=params.attrs["nlons"] ; nlats=params.attrs["nlats"]
    omega=params.attrs["omega"] ; rsphere=params.attrs["rsphere"]
    tscale=params.attrs["tscale"] ; mass1=params.attrs["NSmass"]
    x = Spharmt(int(nlons),int(nlats),int(np.double(nlons)/3.),rsphere,gridtype='gaussian')
    lons1d = x.lons ; lats1d = x.lats
    clats1d = np.sin(x.lats) # 2.*np.arange(nlats)/np.double(nlats)-1.
    lons,lats = np.meshgrid(lons1d, np.arccos(clats1d))
    dlons=2.*np.pi/np.size(lons1d) ; dlats = 2. / np.double(nlats)

    keys=list(f.keys())
    nsize=np.size(keys)-1 # last key contains parameters
    print(str(nsize)+" points from "+str(keys[0])+" to "+str(keys[-2]))
    if(nmax is not None):
        nsize = np.minimum(nsize, nmax)

    # first snapshot:
    data=f[keys[0]]
    sig0=data["sig"][:]
    err=np.zeros(nsize, dtype=np.double)
    serr=np.zeros(nsize, dtype=np.double)
    merr=np.zeros(nsize, dtype=np.double)
    tar=np.zeros(nsize, dtype=np.double)
    sig1=np.zeros(np.shape(sig0), dtype=np.double)
    t0=data.attrs["t"] ; tar[0]=t0
    sigfun=interp1d(lons1d, sig0, axis=1, bounds_error=False, kind='linear',fill_value='extrapolate')
    fout=open(outdir+"/rtest.dat", "w")
    fout.write("#  time(s) -- std -- systematic error -- maximal relative error")
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
        merr[k]=abs(sig/sig1-1.)[wfin].max()
        err[k]=(sig/sig1-1.)[wfin].std()
        serr[k]=(sig/sig1-1.)[wfin].mean() # systematic error
        fout.write(str((t-t0)*tscale)+" "+str(err[k])+" "+str(serr[k])+" "+str(merr[k])+"\n")
        print(str(k)+"/"+str(nsize))
        print("error +-"+str(err[k]))
        print("systematic +-"+str(serr[k]))
        print("maximal +-"+str(merr[k]))
        if(ifplot):
            plots.somemap(lons, lats, sig-sig1, outdir+"/err_sigma.png")
        print("entry "+str(keys[k])+" finished")
    if(ifplot):
        plots.sometimes(tar*tscale*1e3, [err, merr], fmt=['r-', 'k-'] , prefix=outdir+'/err', yname = '$\Delta \Sigma / \Sigma$')
    fout.close()
    f.close()
# comparot('out/run.hdf5')
