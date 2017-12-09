""" 
    File containing all the simulation setup parameters

    NOTE: Can be imported as module that contains everything defined here.
"""
import numpy as np

#################################
# a switch for plotting
ifplot=True

##########################
# a switch for restart
ifrestart=True
nrest=5347 # number of output entry for restart
restartfile='out/runOLD.hdf5' 

##################################################
# grid, time step info
nlons  = 256          # number of longitudes
ntrunc = int(nlons/3) # spectral truncation (to make it alias-free)
nlats  = int(nlons/2) # for gaussian grid

tscale = 6.89631e-06 # time units are GM/c**3, for M=1.4Msun
dt     = 2.e-9       # time step in seconds
itmax  = 10000000    # number of iterations
outskip= 1000 # how often do we output the snapshots

dt/=tscale # dt now in tscales
print "dt = "+str(dt)+"GM/c**3 = "+str(dt*tscale)+"s"

# basic physical parameters
rsphere    = 6.04606               # neutron star radius, GM/c**2 units
pspin      = 1e-2                  # spin period, in seconds
omega      = 2.*np.pi/pspin*tscale # rotation rate
grav       = 1./rsphere**2         # gravity
sig0       = 1e5                   # own neutron star atmosphere scale
sigfloor = 1e-5*sig0   # minimal initial surface density
print "rotation is about "+str(omega*np.sqrt(rsphere))+"Keplerian"

# vertical structure parameters:
# ifiso = False # if we use isothermal EOS instead (obsolete)
csqmin=1e-6 # speed of sound squared (minimal or isothermal)
csqinit=1e-3 # initial speed of sound squared

kappa = 0.35 # opacity, cm^2/g
mu=0.6 # mean molecular weight
mass1=1.4 # accretor mass
# cssqscale = 1.90162e-06/mu/kappa**0.25 # = (4/7) (k/m_p c^2) (0.75 c^5/kappa/sigma_B /GM)^{1/4}
cssqscale = 1.90162e-06 / mu / mass1**0.25 # = (4/7) (k/m_p c^2) (0.75 c^5/sigma_B /GM)^{1/4} # cssqscale * (-geff)**0.25 = csq corresponds roughly to an Eddington limit
betamin=1e-5

print "speed of sound / Keplerian = "+str(np.sqrt(csqmin) / omega / rsphere)

##################################################

# Hyperdiffusion
##################################################
efold = 1000.*dt # efolding timescale at ntrunc for hyperdiffusion
efold_diss = efold/10. # smoothing the dissipation term when used as a heat source
ndiss = 4        # order for hyperdiffusion

##################################################
#perturbation parameters
bump_amp  = 0.25     # height perturbation amplitude
bump_phi0  = np.pi/3. # perturbation latitude
bump_lon0  = np.pi/3. # perturbation longitude
bump_alpha = 1./10. # size of the perturbed region (longitude)
bump_beta  = 1./25.# size of the perturbed region (latitude)

##################################################
# source term
sigplus = 1000. # mass accretion rate is sigplus * 4. * pi * latspread * rsphere**2
sigmax    = 1.e8
latspread = 0.1   # spread in radians
incle      = np.pi*0.08 # inclination of initial rotation, radians
slon0       = 0.1 # longitudinal shift of the source, radians
overkepler = 0.9     # source term rotation with respect to Kepler


