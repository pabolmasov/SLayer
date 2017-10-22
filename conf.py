""" 
    File containing all the simulation setup parameters

    NOTE: Can be imported as module that contains everything defined here.
"""
import numpy as np


##################################################
# grid, time step info
nlons  = 256          # number of longitudes
ntrunc = int(nlons/3) # spectral truncation (to make it alias-free)
nlats  = int(nlons/2) # for gaussian grid


tscale = 6.89631e-06 # time units are GM/c**3 \simeq
dt     = 1.e-8       # time step in seconds
itmax  = 10000000    # number of iterations

dt/=tscale
print "dt = "+str(dt)+"GM/c**3 = "+str(dt*tscale)+"s"


# parameters for test
rsphere    = 6.04606               # earth radius
pspin      = 1e-2                  # spin period, in seconds
omega      = 2.*np.pi/pspin/1.45e5 # rotation rate
overkepler = 0.9                   # source term rotation with respect to Keplerian
grav       = 1./rsphere**2         # gravity
sig0       = 100.                   # own neutron star atmosphere

print "rotation is about "+str(omega*np.sqrt(rsphere))+"Keplerian"

cs=0.01 # speed of sound
sigfloor = 0.1   # auxiliary patameter for EOS; H = cs^2 * log(|sigma| + sigfloor) 
print "speed of sound / Keplerian = "+str(cs / omega / rsphere)


##################################################

# Hyperdiffusion
##################################################
efold = 1000.*dt # efolding timescale at ntrunc for hyperdiffusion
ndiss = 8        # order for hyperdiffusion


##################################################
#perturbation parameters
hamp  = 0.05     # height perturbation amplitude
phi0  = np.pi/3. # perturbation latitude
lon0  = np.pi/3. # perturbation longitude
alpha = 1./3.
beta  = 1./15.


##################################################
# source term
sigplus = 1.
sigmax    = 1.e8
latspread = 0.2   # spread in radians
  







