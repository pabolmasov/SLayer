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
nrest=600 # number of output entry for restart
restartfile='out/runOLD.hdf5' 
if(not(ifrestart)):
    nrest=0
##################################################
# grid, time step info
nlons  = 128          # number of longitudes
ntrunc = int(nlons/3) # spectral truncation (to make it alias-free)
nlats  = int(nlons/2) # for gaussian grid #
# dt=1e-9

logSE = False # if we work with logarithms of Sigma and E instead of the quantities themselves
# (logarithmic treatment does not conserve mass and energy; linear produces strong Gibbs waves)

tscale = 6.89631e-06 # time units are GM/c**3, for M=1.4Msun
# itmax  = 10000000    # number of iterations
outskip= 1000 # how often do we make a simple log output

# basic physical parameters
rsphere    = 6.04606               # neutron star radius, GM/c**2 units
pspin      = 0.03                  # spin period, in seconds
omega      = 2.*np.pi/pspin*tscale # rotation rate
grav       = 1./rsphere**2         # gravity
eps_deformation = omega**2*rsphere**3
print("deformation factor "+str(eps_deformation))
sigmascale = 1.e8 # all the sigmas are normalized to sigmascale, all the energy to sigmascale * c**2

kappa = 0.35*sigmascale # opacity, inverse sigmascale
mu=0.6 # mean molecular weight
mass1=1.4 # accretor mass
# cssqscale = 1.90162e-06/mu/kappa**0.25 # = (4/7) (k/m_p c^2) (0.75 c^5/kappa/sigma_B /GM)^{1/4}
cssqscale = 2.89591e-06 * sigmascale**0.25 / mu * mass1**0.25 # = (4/5) (k/m_p c^2) (0.75 c^5/sigma_B /GM)^{1/4} # cssqscale * (-geff)**0.25 = csq corresponds roughly to an Eddington limit
# if csqmin>cssqscale, we are inevitably super-Eddington
betamin=1e-10 # beta is solved for in the range betamin .. 1-betamin
# there is a singularity near beta=1, not sure about beta=0

sig0       = 1e8/sigmascale             # own neutron star atmosphere scale
print("rotation is about "+str(omega*np.sqrt(rsphere**3))+"Keplerian")
dt_cfl_factor = 0.5 #  Courant-Friedrichs-Levy's multiplier (<~1) for the time step
dt_out_factor = 0.5 # output step, in dynamical times
ifscaledt = True # if we change the value of the time step (including thermal-timescale processes etc. )
tmax=100.*pspin/tscale # we are going to run the simulation for some multiple of spin periods
csqmin=1e-6 # speed of sound squared (minimal or isothermal)
# 1e-6 is about 1keV...
csqinit=1e-6*(sig0*kappa)**0.25 # initial speed of sound squared
isothermal = False # if we use isothermal or polytropic initial conditions
gammainit = 0. # artificially very stiff EOS, because we want density contrasts to be lower
kinit = 1e-8 # proportionality coefficient in initial EOS, Pi=kinit * Sigma^gammainit; of the order c_s^2

# minimal physical surface density and energy density:
sigmafloor = 1./kappa
energyfloor = sigmafloor * csqmin

print("speed of sound / Keplerian = "+str(np.sqrt(csqmin) / omega / rsphere))
# print("vertical scaleheight is ~ "+str(old_div(csqmin,grav))+" = "+str(csqmin/grav/dx)+"dx")

##################################################

# Hyperdiffusion
##################################################
ktrunc = 1. # wavenumber multiplier for spectral cut-off (1 for kmax)
ktrunc_diss = 2. # smoothing the dissipation term when used as a heat source
ndiss = 8.      # order for hyperdiffusion (2 is normal diffusion)

ddivfac = 10. # smoothing enhancement for divergence
##################################################
#perturbation parameters
bump_amp  = -0.05     # perturbation amplitude
bump_lat0  = old_div(np.pi,6.) # perturbation latitude
bump_lon0  = old_div(np.pi,3.) # perturbation longitude
bump_dlon = old_div(np.pi,15.) # size of the perturbed region (longitude)
bump_dlat  = old_div(np.pi,15.) # size of the perturbed region (latitude)

##################################################
# source term
mdotfinal = 1e-3 # Msun/yr, intended mass accretion rate
# sigplus   = 100. # mass accretion rate is sigplus * 4. * pi * latspread * rsphere**2
latspread = 0.2   # spread in radians
sigplus   = 142.374 * (1e8/sigmascale) * mdotfinal / (2.*np.pi*rsphere**2) / mass1 / np.sqrt(4.*np.pi)/np.sin(latspread) # dependence on latspread is approximate and has an accuracy of the order latspread**2
# 6.30322e8*tscale*mdotfinal*(1e8/sigmascale)/np.sqrt(4.*np.pi)/np.sin(latspread)
print("conf: sigplus = "+str(sigplus))
incle     = latspread*0.1 # inclination of initial rotation, radians
slon0     = 0.1  # longitudinal shift of the source, radians
overkepler = 0.9     # source term rotation with respect to Kepler
# friction time scale with the neutron star:
tfric=0.*pspin/tscale
# depletion of the atmosphere:
tdepl=0.*pspin/tscale
# turning on the source smoothly
tturnon=10.*pspin/tscale

#####################################################
# twist test
iftwist=False
twistscale=latspread

