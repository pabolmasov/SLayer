""" 
    File containing all the simulation setup parameters

    NOTE: Can be imported as module that contains everything defined here.
"""
from __future__ import print_function
from __future__ import division
from builtins import str
from past.utils import old_div
import numpy as np

#################################
# a switch for plotting
ifplot=True

##########################
# a switch for restart
ifrestart=False
nrest=400 # number of output entry for restart
restartfile='out/runOLD.hdf5' 
if(not(ifrestart)):
    nrest=0
##################################################
# grid, time step info
nlons  = 256          # number of longitudes
ntrunc = int(nlons/3) # spectral truncation (to make it alias-free)
nlats  = int(nlons/2) # for gaussian grid #
# dt=1e-9

tscale = 6.89631e-06 # time units are GM/c**3, for M=1.4Msun
# itmax  = 10000000    # number of iterations
outskip= 10000 # how often do we output the snapshots (in dt_CFL)

# basic physical parameters
rsphere    = 6.04606               # neutron star radius, GM/c**2 units
pspin      = 0.003                  # spin period, in seconds
omega      = 2.*np.pi/pspin*tscale # rotation rate
grav       = 1./rsphere**2         # gravity
sigmascale = 1e8 # all the sigmas are normalized to sigmascale, all the energy to sigmascale * c**2
sig0       = 0.01                   # own neutron star atmosphere scale
sigfloor = 1e-5*sig0   # minimal initial surface density
print("rotation is about "+str(omega*np.sqrt(rsphere))+"Keplerian")
print("approximate cell size is dx ~ "+str(rsphere/np.double(nlons)))
tmax=10.*pspin/tscale # we are going to run the simulation for ten(s) of spin periods
csqmin=1e-6 # speed of sound squared (minimal or isothermal)
# 1e-6 is about 1keV...
energyfloor = sigfloor * csqmin
csqinit=1e-4 # initial speed of sound squared
isothermal = False # if we use isothermal or polytropic initial conditions
gammainit = 0. # artificially very stiff EOS, because we want density contrasts to be lower
kinit = 1e-6 # proportionality coefficient in initial EOS, Pi=kinit * Sigma^gammainit; of the order c_s^2

kappa = 0.35*sigmascale # opacity, inverse sigmascale
mu=0.6 # mean molecular weight
mass1=1.4 # accretor mass
# cssqscale = 1.90162e-06/mu/kappa**0.25 # = (4/7) (k/m_p c^2) (0.75 c^5/kappa/sigma_B /GM)^{1/4}
cssqscale = 2.89591e-06 * sigmascale**0.25 / mu * mass1**0.25 # = (4/5) (k/m_p c^2) (0.75 c^5/sigma_B /GM)^{1/4} # cssqscale * (-geff)**0.25 = csq corresponds roughly to an Eddington limit
# if csqmin>cssqscale, we are inevitably super-Eddington
betamin=1e-8 # beta is solved for in the range betamin .. 1-betamin
# there is a singularity near beta=1, not sure about beta=0

print("speed of sound / Keplerian = "+str(np.sqrt(csqmin) / omega / rsphere))
# print("vertical scaleheight is ~ "+str(old_div(csqmin,grav))+" = "+str(csqmin/grav/dx)+"dx")

##################################################

# Hyperdiffusion
##################################################
efold = 0.1 # efolding timescale at ntrunc for hyperdiffusion (in dt units)
efold_diss = 0. # smoothing the dissipation term when used as a heat source
ndiss = 8      # order for hyperdiffusion (2 is normal diffusion)

##################################################
#perturbation parameters
bump_amp  = -0.15     # perturbation amplitude
bump_lat0  = old_div(np.pi,6.) # perturbation latitude
bump_lon0  = old_div(np.pi,3.) # perturbation longitude
bump_dlon = old_div(np.pi,15.) # size of the perturbed region (longitude)
bump_dlat  = old_div(np.pi,15.) # size of the perturbed region (latitude)

##################################################
# source term
sigplus   = 0.0 # mass accretion rate is sigplus * 4. * pi * latspread * rsphere**2
latspread = 0.1   # spread in radians
incle     = latspread*0.1 # inclination of initial rotation, radians
slon0     = 0.1  # longitudinal shift of the source, radians
overkepler = 0.9     # source term rotation with respect to Kepler
# friction time scale with the neutron star:
tfric=0.*pspin/tscale
# depletion of the atmosphere:
tdepl=0.*pspin/tscale

#####################################################
# twist test
iftwist=False
twistscale=latspread

