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


tscale = 6.89631e-06 # time units are GM/c**3, for M=1.4Msun
dt     = 5.e-9       # time step in seconds
itmax  = 10000000    # number of iterations
outskip= 1000 # how often do we output the snapshots

dt/=tscale # dt now in tscales
print "dt = "+str(dt)+"GM/c**3 = "+str(dt*tscale)+"s"

# basic physical parameters
rsphere    = 6.04606               # neutron star radius, GM/c**2 units
pspin      = 1e-2                  # spin period, in seconds
omega      = 2.*np.pi/pspin*tscale # rotation rate
grav       = 1./rsphere**2         # gravity
sig0       = 1e5                   # own neutron star atmosphere

print "rotation is about "+str(omega*np.sqrt(rsphere))+"Keplerian"

# vertical structure parameters:
ifiso = True # if we use isothermal EOS instead
csqmin=1e-4 # speed of sound squared (minimal or isothermal)
sigfloor = 0.1   # auxiliary patameter for EOS; H = cs^2 * log(|sigma| + sigfloor) 
kappa = 0.35 # opacity, cm^2/g
mu=0.6 # mean molecular weight
cssqscale = 1.90162e-06/mu/kappa**0.25 # = (4/7) (k/m_p c^2) (0.75 c^5/kappa/sigma_B /GM)^{1/4}
betamin=1e-5

print "speed of sound / Keplerian = "+str(np.sqrt(csqmin) / omega / rsphere)

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
incle      = np.pi*0.08 # inclination of initial rotation, radians
slon0       = 0.1 # longitudinal shift of the source, radians
overkepler = 0.9     # source term rotation with respect to Kepler


