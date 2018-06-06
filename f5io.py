from __future__ import print_function
from __future__ import division
from builtins import str
from past.utils import old_div
import h5py
import numpy as np
import os
import scipy.interpolate as si
import scipy.ndimage as nd
from spharmt import Spharmt 

outdir = "out" #default output directory

# combine several HDF5 files into one
def HDFcombine(f5array):
    n=np.size(f5array)
    if(n<=1):
        print("nothing to combine")
        exit()
    else:
        print("preparing to glue "+str(n)+" files")
    fnew=h5py.File(outdir+'/runcombine.hdf5', "w")
    f = h5py.File(f5array[0],'r')
    #    params=f0['params']
    #    parnew = f.create_group("params")
    f.copy("params", fnew)
    keys=list(f.keys())
    currentkeys=[]
    for k in np.arange(n):
        nkeys=np.size(keys)-1 # the last one is "params", we do not need it anymore
        for kkey in np.arange(nkeys):
#            print keys[kkey]
            if keys[kkey] in currentkeys:
                print(" duplicate entry "+keys[kkey])
            else:
                f.copy(keys[kkey], fnew)
                currentkeys.append(keys[kkey])
        print("file "+f5array[k]+" added")
        if(k<(n-1)): # reading next file
            f.close()
            f = h5py.File(f5array[k+1],'r')
            keys=list(f.keys())
    f.flush() ;    f.close()
    fnew.close() 
    
# save general simulation parameters to the file
def saveParams(f5, conf):
    grp0 = f5.create_group("params")
    
    grp0.attrs['nlons']      = conf.nlons
    grp0.attrs['ntrunc']     = conf.ntrunc
    grp0.attrs['nlats']      = conf.nlats
    grp0.attrs['tscale']     = conf.tscale
#    grp0.attrs['dt_cfl']     = conf.dt_cfl
#    grp0.attrs['itmax']      = conf.itmax
    grp0.attrs['rsphere']    = conf.rsphere
    grp0.attrs['pspin']      = conf.pspin
    grp0.attrs['omega']      = conf.omega
    grp0.attrs['overkepler'] = conf.overkepler
    grp0.attrs['grav']       = conf.grav
    grp0.attrs['sig0']       = conf.sig0
    grp0.attrs['csqmin']         = conf.csqmin
    grp0.attrs['NSmass']         = conf.mass1

    f5.flush()

#Save simulation snapshot
def saveSim(f5, nout, t,
            vortg, divg, ug, vg, sig, energy, beta,
            accflag, dissipation, qminus, qplus,
            conf):
    sarea=4.*np.pi/np.double(conf.nlons*conf.nlats)*conf.rsphere**2
    mass=sig.sum()*sarea
    #        mass_acc=(sig*accflag).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
    #        mass_native=(sig*(1.-accflag)).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
    totenergy=(sig*energy+old_div((ug**2+vg**2),2.)).sum()*sarea

    scycle = str(nout).rjust(6, '0')
    grp = f5.create_group("cycle_"+scycle)
    grp.attrs['t']      = t      # time
    #    grp.attrs['mass']   = mass   # total mass
    #    grp.attrs['energy'] = totenergy # total mechanical energy

    grp.create_dataset("vortg", data=vortg)
    grp.create_dataset("divg",  data=divg)
    grp.create_dataset("ug",    data=ug)
    grp.create_dataset("vg",    data=vg)
    grp.create_dataset("sig",   data=sig)
    grp.create_dataset("energy",   data=energy)
    grp.create_dataset("beta",   data=beta)
    grp.create_dataset("accflag",   data=accflag)
    grp.create_dataset("diss",  data=dissipation)
    grp.create_dataset("qplus",  data=qplus)
    grp.create_dataset("qminus",  data=qminus)

    f5.flush()

# restart from file
def restart(restartfile, nrest, conf):

    f5 = h5py.File(restartfile,'r')

    params=f5['params/']
    nlons1 =int(params.attrs["nlons"])
    nlats1 =int(params.attrs["nlats"])
    ntrunc1 = int(old_div(nlons1,3)) 
    rsphere=params.attrs["rsphere"]
    
    data  = f5["cycle_"+str(nrest).rjust(6, '0')]

    if ((nlons1 != conf.nlons) | (nlats1 != conf.nlats)): # interpolate!
        print("restart: dimensions unequal\n")
        print("restart: interpolating from "+str(nlons1)+", "+str(nlats1))
        print(" to "+str(conf.nlons)+", "+str(conf.nlats))
        x = Spharmt(conf.nlons, conf.nlats, conf.ntrunc, conf.rsphere, gridtype='gaussian') # new grid
        #        lons,lats = np.meshgrid(x.lons, x.lats)
        x1 = Spharmt(nlons1, nlats1, ntrunc1, rsphere, gridtype='gaussian') # old grid
        x1.lats*=-1. # the picture gets upside down otherwise (why?)
        #        lons1,lats1 = np.meshgrid(x1.lons, x1.lats)
        # file contains dimensions nlats1, nlons1,
        vortg1 = data["vortg"][:]  ;     divg1 = data["divg"][:] ;  sig1 = data["sig"][:] ;   accflag1 = data["accflag"][:] ; energy1 = data["energy"][:]
        # interpolation:
        vortfun =  si.interp2d(x1.lons, x1.lats, -vortg1, kind='linear')
        divfun =  si.interp2d(x1.lons, x1.lats, divg1, kind='linear')
        sigfun =  si.interp2d(x1.lons, x1.lats, sig1, kind='linear')
        energyfun =  si.interp2d(x1.lons, x1.lats, energy1, kind='linear')
        accflagfun =  si.interp2d(x1.lons, x1.lats, accflag1, kind='linear')
        vortg = -vortfun(x.lons, x.lats) ; divg = divfun(x.lons, x.lats) ; sig = sigfun(x.lons, x.lats) ; energyg = energyfun(x.lons, x.lats) ; accflag = accflagfun(x.lons, x.lats)
        # accflag may be smoothed without any loss of generality or stability
        dlats=old_div(np.pi,np.double(conf.nlats)) ;  dlons=2.*np.pi/np.double(conf.nlons) # approximate size in latitudinal and longitudinal directions
        print("smoothing accflag")
        accflag = nd.filters.gaussian_filter(accflag, old_div(2.,(old_div(1.,dlats)+old_div(1.,dlons))), mode='constant') # smoothing
        w1=np.where(accflag > 1.) ; w0=np.where(accflag <0.)
        if(np.size(w1)>0):
            accflag[w1]=1.
        if(np.size(w0)>0):
            accflag[w0]=0. 
        print("restart: restore and interpolation finished")
    else:
        vortg = data["vortg"][:]
        divg  = data["divg"][:]
        sig   = data["sig"][:]
        energyg   = data["energy"][:]
        accflag   = data["accflag"][:]
    t0=data.attrs["t"]
    f5.close()

    # if successful, we need to take the file off the way
    #    os.system("mv "+restartfile+" "+restartfile+".backup")

    return vortg, divg, sig, energyg, accflag, t0


