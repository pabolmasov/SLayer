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

# combine several HDF5 files into one
def HDFcombine(f5array, otheroutdir=None):
    n=np.size(f5array)
    if(n<=1):
        print("nothing to combine")
        exit()
    else:
        print("preparing to glue "+str(n)+" files")
    if(otheroutdir == None):
        outdir=os.path.dirname(f5array[0])
    else:
        outdir=otheroutdir
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
    grp0.attrs['nlats']      = conf.nlats
    grp0.attrs['ntrunc']     = conf.ntrunc
    grp0.attrs['tscale']     = conf.tscale
    grp0.attrs['ktrunc']     = conf.ktrunc
    grp0.attrs['ktrunc_diss']     = conf.ktrunc_diss
    grp0.attrs['ndiss']     = conf.ndiss
#    grp0.attrs['itmax']      = conf.itmax
    grp0.attrs['rsphere']    = conf.rsphere
    grp0.attrs['pspin']      = conf.pspin
    grp0.attrs['omega']      = conf.omega # a bit redundant, omega = 2.*np.pi/pspin*tscale
    grp0.attrs['sigmascale']       = conf.sigmascale
    grp0.attrs['overkepler'] = conf.overkepler
    grp0.attrs['grav']       = conf.grav
    grp0.attrs['sig0']       = conf.sig0
    grp0.attrs['sigplus']       = conf.sigplus
    grp0.attrs['mdotfinal']       = conf.mdotfinal
    grp0.attrs['latspread']       = conf.latspread
    grp0.attrs['csqmin']         = conf.csqmin
    grp0.attrs['tfric']         = conf.tfric
    grp0.attrs['tdepl']         = conf.tdepl
    grp0.attrs['NSmass']         = conf.mass1

    f5.flush()

#Save simulation snapshot
def saveSim(f5, nout, t,
            vortg, divg, ug, vg, sig, energy, beta,
            accflag, dissipation, qminus, qplus, sdot,
            conf):
    x = Spharmt(int(conf.nlons),int(conf.nlats),int(old_div(conf.nlons,3)),conf.rsphere,gridtype='gaussian')
    lons1d = x.lons
    clats1d = np.sin(x.lats) # 2.*np.arange(nlats)/np.double(nlats)-1.
    dlons=2.*np.pi/np.double(conf.nlons) ; dlats=2./np.double(conf.nlats)
    mass=np.trapz(sig.sum(axis=1), x=-clats1d)*dlons
    mass_acc=np.trapz((sig*accflag).sum(axis=1), x=-clats1d)*dlons
    mass_native=np.trapz((sig*(1.-accflag)).sum(axis=1), x=-clats1d)*dlons
    lumtot=np.trapz(qminus.sum(axis=1), x=-clats1d)*dlons
    heattot=np.trapz(qplus.sum(axis=1), x=-clats1d)*dlons
    print("f5io: lumtot = "+str(lumtot)+"; heattot = "+str(heattot))
    mdot=np.trapz(sdot.sum(axis=1), x=-clats1d)*dlons
    #    totenergy=(sig*energy+old_div((ug**2+vg**2),2.)).sum()*sarea

    scycle = str(nout).rjust(6, '0')
    grp = f5.create_group("cycle_"+scycle)
    grp.attrs['t']      = t      # time
    grp.attrs['mass']   = mass   # total mass
    grp.attrs['newmass']   = mass_acc   # accreted mass
    grp.attrs['oldmass']   = mass_native   # native mass
    grp.attrs['lumtot']   = lumtot   # total luminosity
    grp.attrs['heattot']   = heattot   # total energy released
    grp.attrs['mdot']   = mdot   # total mass accreted/lost
    print("f5io: mdot = "+str(mdot*conf.rsphere**2*conf.mass1**2*(conf.sigmascale/1e8) * 0.00702374))
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
        sigfun =  si.interp2d(x1.lons, x1.lats, np.log(sig1), kind='linear')
        energyfun =  si.interp2d(x1.lons, x1.lats, np.log(energy1), kind='linear')
        accflagfun =  si.interp2d(x1.lons, x1.lats, accflag1, kind='linear')
        vortg = -vortfun(x.lons, x.lats) ; divg = divfun(x.lons, x.lats) ; sig = np.exp(sigfun(x.lons, x.lats)) ; energyg = np.exp(energyfun(x.lons, x.lats)) ; accflag = accflagfun(x.lons, x.lats)
        # accflag may be smoothed without any loss of generality or stability
        dlats=old_div(np.pi,np.double(conf.nlats)) ;  dlons=2.*np.pi/np.double(conf.nlons) # approximate size in latitudinal and longitudinal directions
        print("smoothing accflag")
        w1=np.where(accflag > 1.) ; w0=np.where(accflag <0.)
        if(np.size(w1)>0):
            accflag[w1]=1.
        if(np.size(w0)>0):
            accflag[w0]=0. 
        accflag = nd.filters.gaussian_filter(accflag,
                                             2./(1./dlats+1./dlons),
                                             mode='constant') # smoothing
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


