import numpy as np
from os import *

import timing, snapshooter
from timing import lightcurves
from snapshooter import multireader

'''
if you need to reduce multiple output directories
'''

outlist = ['out_NALR']
#['out_3LR', 'out_3HR', 'out_8LR', 'out_8HR', 'out_NAHR',
#           'out_NALR',  'out_NDHR',  'out_NDLR',  'out_slow3HR',
#           'out_stwistHR',  'out_twistHR']
latlist = [0., np.pi/4., np.pi/2.]

for out in outlist:
    for k in np.arange(np.size(latlist)):
        print("reduction under way in "+out+"; lat = "+str(latlist[k]))
        lightcurves(out+'/run.hdf5', latlist[k], 0.)
        #  system('cp '+out+'/pds_diss.dat '+out+'/pds_diss'+str(latlist[k])+'.dat')
        #  system('cp '+out+'/pds_mass.dat '+out+'/pds_mass'+str(latlist[k])+'.dat')
#    multireader(out+'/run.hdf5', nframes=500)
#    system("tar -cf "+out+"/alldat.tar "+out+"/*.dat")
#    system("ls "+out+"/alldat.tar")
