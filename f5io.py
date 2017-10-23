import h5py
import numpy as np
import os


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
    grp0.attrs['cs']         = conf.cs

    f5.flush()


#Save simulation snapshot
def saveSim(f5, nout, t,
            mass, energy,
            vortg, divg, ug, vg, sig, accflag, dissipation
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
    grp.create_dataset("accflag",   data=accflag)
    grp.create_dataset("diss",  data=dissipation)

    f5.flush()

# restart from file
def restart(restartfile, nrest, conf):

    f5 = h5py.File(restartfile,'r')

    params=f5['params/']
    nlons1 =params.attrs["nlons"]
    nlats1 =params.attrs["nlats"]
    rsphere=params.attrs["rsphere"]
    
    if ((nlons1 != conf.nlons) | (nlats1 != conf.nlats)): # interpolate!
        print "restart: dimensions unequal, not supported yet"
        exit(1)
    else:
        keys  = f5.keys()
        ksize = np.size(keys)

        data  = f5["cycle_"+str(nrest).rjust(6, '0')]

        vortg = data["vortg"][:]
        divg  = data["divg"][:]
        sig   = data["sig"][:]
        accflag   = data["accflag"][:]

    f5.close()

    # if successful, we need to take the file off the way
    #    os.system("mv "+restartfile+" "+restartfile+".backup")

    return vortg, divg, sig, accflag


