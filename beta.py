import numpy as np
import scipy.interpolate as si

from conf import betamin
############################
# beta calibration
bmin=betamin ; bmax=1.-betamin ; nb=10000
# hard limits for stability; bmin\sim 1e-7 approximately corresponds to Coulomb coupling G\sim 1,
# hence there is even certain physical meaning in bmin
# "beta" part of the main loop is little-time-consuming independently of nb
b = (bmax-bmin)*((np.arange(nb)+0.5)/np.double(nb))+bmin
bx = b/(1.-b)**0.25
b[0]=0. ; bx[0]=0.0  # ; b[nb-1]=1e3 ; bx[nb-1]=1.
betasolve_p=si.interp1d(bx, b, kind='linear', bounds_error=False, fill_value=1.)
# as a function of pressure
betasolve_e=si.interp1d(bx/(1.-b/2.)/3., b, kind='linear', bounds_error=False,fill_value=1.)
# as a function of energy
######################################
