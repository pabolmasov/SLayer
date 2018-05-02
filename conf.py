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
nrest=762 # number of output entry for restart
restartfile='out/runOLD.hdf5' 
if(not(ifrestart)):
    nrest=0
##################################################
# grid, time step info
nlons  = 256          # number of longitudes
ntrunc = int(old_div(nlons,3)) # spectral truncation (to make it alias-free)
nlats  = int(old_div(nlons,2)) # for gaussian grid
# dt=1e-9

tscale = 6.89631e-06 # time units are GM/c**3, for M=1.4Msun
# itmax  = 10000000    # number of iterations
outskip= 10000 # how often do we output the snapshots (in dt_CFL)

# basic physical parameters
rsphere    = 6.04606               # neutron star radius, GM/c**2 units
pspin      = 0.1                  # spin period, in seconds
omega      = 2.*np.pi/pspin*tscale # rotation rate
grav       = old_div(1.,rsphere**2)         # gravity
sig0       = 1e8                   # own neutron star atmosphere scale
sigfloor = 1e-5*sig0   # minimal initial surface density
print("rotation is about "+str(omega*np.sqrt(rsphere))+"Keplerian")
print("approximate cell size is dx ~ "+str(rsphere/np.double(nlons)))
tmax=10.*pspin/tscale # we are going to run the simulation for ten(s) of spin periods

# dt/=tscale # dt now in tscales
# dx = old_div(rsphere,np.double(nlons))
# dy = old_div(rsphere,np.double(nlats))
# minimal physical size of the cell is smaller than R/N_phi because of the variable Lame multiplier np.cos(lats)
# dx=np.fabs((lons[1:-1,1:]-lons[1:-1,:-1])*np.cos(lats[1:-1,:-1])).min()
# dy=np.fabs(x.lats[1:]-x.lats[:-1]).min()
# dt_cfl = 0.5 * (1./dx + 1./dy) # basic CFL limit for light velocity
# print("dt(CFL) = "+str(dt_cfl)+"GM/c**3 = "+str(dt_cfl*tscale)+"s")
# dtout=np.double(outskip)*dt_cfl # time step for output
# print("dtout = "+str(dtout)+"GM/c**3 = "+str(dtout*tscale)+"s")
# itmax  = np.int(np.ceil(tmax/dt_cfl))    # number of iterations   
# print("preparing to run for "+str(itmax)+"iterations, making "+str(np.int(np.floor(tmax/dtout)))+" outputs")
# ii=raw_input("x")
# vertical structure parameters:
# ifiso = False # if we use isothermal EOS instead (obsolete)
csqmin=1e-6 # speed of sound squared (minimal or isothermal)
# 1e-6 is about 1keV...
csqinit=1e-4 # initial speed of sound squared

kappa = 0.35 # opacity, cm^2/g
mu=0.6 # mean molecular weight
mass1=1.4 # accretor mass
# cssqscale = 1.90162e-06/mu/kappa**0.25 # = (4/7) (k/m_p c^2) (0.75 c^5/kappa/sigma_B /GM)^{1/4}
cssqscale = 2.89591e-06 / mu * mass1**0.25 # = (4/5) (k/m_p c^2) (0.75 c^5/sigma_B /GM)^{1/4} # cssqscale * (-geff)**0.25 = csq corresponds roughly to an Eddington limit
# if csqmin>cssqscale, we are inevitably super-Eddington
betamin=1e-7 # beta is solved for in the range betamin .. 1-betamin
# there is a singularity near beta=1, not sure about beta=0

print("speed of sound / Keplerian = "+str(np.sqrt(csqmin) / omega / rsphere))
# print("vertical scaleheight is ~ "+str(old_div(csqmin,grav))+" = "+str(csqmin/grav/dx)+"dx")

##################################################

# Hyperdiffusion
##################################################
efold = 1000. # efolding timescale at ntrunc for hyperdiffusion (in dt units)
efold_diss = .1*efold # smoothing the dissipation term when used as a heat source
ndiss = 4        # order for hyperdiffusion (4 is normal diffusion)

##################################################
#perturbation parameters
bump_amp  = -0.05     # perturbation amplitude
bump_lat0  = old_div(np.pi,6.) # perturbation latitude
bump_lon0  = old_div(np.pi,3.) # perturbation longitude
bump_dlon = old_div(np.pi,15.) # size of the perturbed region (longitude)
bump_dlat  = old_div(np.pi,15.) # size of the perturbed region (latitude)

##################################################
# source term
sigplus = 1e10 # mass accretion rate is sigplus * 4. * pi * latspread * rsphere**2
sigmax    = 1.e8
latspread = 0.2   # spread in radians
incle     = np.pi/6. # inclination of initial rotation, radians
slon0     = 0.1  # longitudinal shift of the source, radians
overkepler = 0.9     # source term rotation with respect to Kepler
# friction time scale with the neutron star
tfric=1000.*pspin/tscale

#####################################################
# twist test
iftwist=True
twistscale=latspread

