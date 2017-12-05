import h5py
import numpy as np
import os
import scipy.interpolate as si
import scipy.ndimage as nd
from spharmt import Spharmt 

outdir = "out" #default output directory

# save general simulation parameters to the file
def saveParams(f5, conf):
    grp0 = f5.create_group("params")
    
    grp0.attrs['nlons']      = conf.nlons
    grp0.attrs['ntrunc']     = conf.ntrunc
    grp0.attrs['nlats']      = conf.nlats
    grp0.attrs['tscale']     = conf.tscale
    grp0.attrs['dt']         = conf.dt
    grp0.attrs['itmax']      = conf.itmax
    grp0.attrs['rsphere']    = conf.rsphere
    grp0.attrs['pspin']      = conf.pspin
    grp0.attrs['omega']      = conf.omega
    grp0.attrs['overkepler'] = conf.overkepler
    grp0.attrs['grav']       = conf.grav
    grp0.attrs['sig0']       = conf.sig0
    grp0.attrs['cs']         = conf.csqmin

    f5.flush()


#Save simulation snapshot
def saveSim(f5, nout, t,
            mass, energy,
            vortg, divg, ug, vg, sig, press, accflag, dissipation
            ):

    scycle = str(nout).rjust(6, '0')
    grp = f5.create_group("cycle_"+scycle)
    grp.attrs['t']      = t      # time
    grp.attrs['mass']   = mass   # total mass
    grp.attrs['energy'] = energy # total mechanical energy

    grp.create_dataset("vortg", data=vortg)
    grp.create_dataset("divg",  data=divg)
    grp.create_dataset("ug",    data=ug)
    grp.create_dataset("vg",    data=vg)
    grp.create_dataset("sig",   data=sig)
    grp.create_dataset("press",   data=press)
    grp.create_dataset("accflag",   data=accflag)
    grp.create_dataset("diss",  data=dissipation)

    f5.flush()

# restart from file
def restart(restartfile, nrest, conf):

    f5 = h5py.File(restartfile,'r')

    params=f5['params/']
    nlons1 =params.attrs["nlons"]
    nlats1 =params.attrs["nlats"]
    ntrunc1 = int(nlons1/3) 
    rsphere=params.attrs["rsphere"]
    
    data  = f5["cycle_"+str(nrest).rjust(6, '0')]

    if ((nlons1 != conf.nlons) | (nlats1 != conf.nlats)): # interpolate!
        print "restart: dimensions unequal\n"
        print "restart: interpolating from "+str(nlons1)+", "+str(nlats1)
        print " to "+str(conf.nlons)+", "+str(conf.nlats)
        x = Spharmt(conf.nlons, conf.nlats, conf.ntrunc, conf.rsphere, gridtype='gaussian') # new grid
        #        lons,lats = np.meshgrid(x.lons, x.lats)
        x1 = Spharmt(nlons1, nlats1, ntrunc1, rsphere, gridtype='gaussian') # old grid
        #        lons1,lats1 = np.meshgrid(x1.lons, x1.lats)
        # file contains dimensions nlats1, nlons1,
        vortg1 = data["vortg"][:]  ;     divg1 = data["divg"][:] ;  sig1 = data["sig"][:] ;   accflag1 = data["accflag"][:] ; press1 = data["press"][:]
        # interpolation:
        vortfun =  si.interp2d(x1.lons, x1.lats, vortg1, kind='linear')
        divfun =  si.interp2d(x1.lons, x1.lats, divg1, kind='linear')
        sigfun =  si.interp2d(x1.lons, x1.lats, sig1, kind='linear')
        pressfun =  si.interp2d(x1.lons, x1.lats, sig1, kind='linear')
        accflagfun =  si.interp2d(x1.lons, x1.lats, accflag1, kind='linear')
        vortg = -vortfun(x.lons, x.lats) ; divg = divfun(x.lons, x.lats) ; sig = sigfun(x.lons, x.lats) ; pressg = pressfun(x.lons, x.lats) ; accflag = accflagfun(x.lons, x.lats)
        # accflag may be smoothed without any loss of generality or stability
        dlats=np.pi/np.double(conf.nlats) ;  dlons=2.*np.pi/np.double(conf.nlons) # approximate size in latitudinal and longitudinal directions
        print "smoothing accflag"
        accflag = nd.filters.gaussian_filter(accflag, 2./(1./dlats+1./dlons), mode='constant') # smoothing
        w1=np.where(accflag > 1.) ; w0=np.where(accflag <0.)
        if(np.size(w1)>0):
            accflag[w1]=1.
        if(np.size(w0)>0):
            accflag[w0]=0. 
        print "restart: restore and interpolation finished"
    else:
        vortg = data["vortg"][:]
        divg  = data["divg"][:]
        sig   = data["sig"][:]
        pressg   = data["press"][:]
        accflag   = data["accflag"][:]
    f5.close()

    # if successful, we need to take the file off the way
    #    os.system("mv "+restartfile+" "+restartfile+".backup")

    return vortg, divg, sig, pressg, accflag


