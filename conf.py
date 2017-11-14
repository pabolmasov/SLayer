""" 
    File containing all the simulation setup parameters

    NOTE: Can be imported as module that contains everything defined here.
"""
import numpy as np

#################################
# a switch for plotting
ifplot=True

##################################################
# grid, time step info
nlons  = 256          # number of longitudes
ntrunc = int(nlons/3) # spectral truncation (to make it alias-free)
nlats  = int(nlons/2) # for gaussian grid


tscale = 6.89631e-06 # time units are GM/c**3 \simeq
dt     = 5.e-9       # time step in seconds
itmax  = 10000000    # number of iterations
outskip= 20000 # how often do we output the snapshots

dt/=tscale
print "dt = "+str(dt)+"GM/c**3 = "+str(dt*tscale)+"s"


# parameters for test
rsphere    = 6.04606               # neutron star radius, GM/c**2 units
pspin      = 1e-2                  # spin period, in seconds
omega      = 2.*np.pi/pspin*tscale # rotation rate
grav       = 1./rsphere**2         # gravity
sig0       = 1e5                   # own neutron star atmosphere


print "rotation is about "+str(omega*np.sqrt(rsphere))+"Keplerian"

cs=0.01 # speed of sound
sigfloor = 0.1   # auxiliary patameter for EOS; H = cs^2 * log(|sigma| + sigfloor) 
print "speed of sound / Keplerian = "+str(cs / omega / rsphere)


##################################################

# Hyperdiffusion
##################################################
efold = 2000.*dt # efolding timescale at ntrunc for hyperdiffusion
ndiss = 4        # order for hyperdiffusion


##################################################
#perturbation parameters
hamp  = 0.25     # height perturbation amplitude
phi0  = np.pi/3. # perturbation latitude
lon0  = np.pi/3. # perturbation longitude
alpha = 1./10. # size of the perturbed region
beta  = 1./25.# size of the perturbed region


##################################################
# source term
sigplus = 1e3
sigmax    = 1.e8
latspread = 0.1   # spread in radians
incle      = np.pi*0.1 # inclination of initial rotation, radians
slon0       = 0.1 # longitudinal shift of the source, radians
overkepler = 0.9     # source term rotation with respect to Kepler







