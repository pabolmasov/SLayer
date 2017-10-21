import h5py
import numpy as np


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



