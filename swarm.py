# Simple spherical harmonic shallow water model toy code based on shtns library.
#
# 
# Refs:
#  "non-linear barotropically unstable shallow water test case"
#  example provided by Jeffrey Whitaker
#  https://gist.github.com/jswhit/3845307
#
#  Galewsky et al (2004, Tellus, 56A, 429-440)
#  "An initial-value problem for testing numerical models of the global
#  shallow-water equations" DOI: 10.1111/j.1600-0870.2004.00071.x
#  http://www-vortex.mcs.st-and.ac.uk/~rks/reprints/galewsky_etal_tellus_2004.pdf
#  
#  shtns/examples/shallow_water.py
#
#  Jakob-Chien et al. 1995:
#  "Spectral Transform Solutions to the Shallow Water Test Set"
#

import numpy as np
import scipy.ndimage as spin
import shtns
import matplotlib.pyplot as plt
import time
import os
import h5py


#Code modules:
from spharmt import Spharmt 
import f5io as f5io #module for file I/O
import conf as conf #importing simulation setup module
from plots import visualize

##################################################
# setup code environment
f5io.outdir = 'out'
if not os.path.exists(f5io.outdir):
    os.makedirs(f5io.outdir)

f5 = h5py.File(f5io.outdir+'/run.hdf5', "w")

#import simulation parameters to global scope
from conf import nlons, nlats
from conf import dt, omega, rsphere, sig0, overkepler, tscale
from conf import hamp, phi0, lon0, alpha, beta  #initial height perturbation parameters
from conf import efold, ndiss
from conf import cs
from conf import itmax
from conf import sigfloor
from conf import sigplus, sigmax, latspread #source and sink terms




##################################################
# setup up spherical harmonic instance, set lats/lons of grid
x = Spharmt(conf.nlons, conf.nlats, conf.ntrunc, conf.rsphere, gridtype='gaussian')
lons,lats = np.meshgrid(x.lons, x.lats)


# guide grids for plotting
lons1d = (180./np.pi)*x.lons-180.
lats1d = (180./np.pi)*x.lats


# zonal jet
vg = np.zeros((nlats, nlons))
ug = np.ones((nlats, nlons))*np.cos(lats)*omega*rsphere


# height perturbation.
hbump = hamp*np.cos(lats)*np.exp(-((lons-lon0)/alpha)**2)*np.exp(-(phi0-lats)**2/beta)


# initial vorticity, divergence in spectral space
vortSpec, divSpec =  x.getVortDivSpec(ug,vg)
vortg = x.sph2grid(vortSpec)
divg  = x.sph2grid(divSpec)



# create hyperdiffusion factor
hyperdiff_fact = np.exp((-dt/efold)*(x.lap/x.lap[-1])**(ndiss/2))
sigma_diff=np.exp((-dt/efold)*(x.lap/x.lap[-1])**(ndiss/2))

# sigma is an exact isothermal solution + an unbalanced bump
sig = sig0*(np.exp(-(omega*rsphere/cs)**2/2.*(1.-np.cos(lats))) + hbump) # exact solution + perturbation
signative = sig0*(np.exp(-(omega*rsphere/cs)**2/2.*(1.-np.cos(lats))) + hbump) # this density component will ignore the source contribution
vortg *= (1.-hbump) # some initial condition for vorticity (cyclone?)

# initialize spectral tendency arrays
ddivdtSpec  = np.zeros(vortSpec.shape+(3,), np.complex)
dvortdtSpec = np.zeros(vortSpec.shape+(3,), np.complex)
dsigdtSpec  = np.zeros(vortSpec.shape+(3,), np.complex)
# dsignativedtSpec  = np.zeros(vortSpec.shape+(3,), np.complex)

# Cycling integers for integrator
nnew = 0
nnow = 1
nold = 2

###########################################################
# restart module:
ifrestart=True

if(ifrestart):
    restartfile='out/runOLD.hdf5'
    nrest=3190 # No of the restart output
    #    nrest=5300 # No of the restart output
    vortg, digg, sig, signative = f5io.restart(restartfile, nrest, conf)

else:
    nrest=0
        
sigSpec  = x.grid2sph(sig)
signativeSpec  = x.grid2sph(signative)
divSpec  = x.grid2sph(divg)
vortSpec = x.grid2sph(vortg)


###################################################
# Save simulation setup to file
f5io.saveParams(f5, conf)


    
##################################################
# source/sink term
def sdotsource(lats, lons, latspread):
    y=np.zeros((nlats,nlons), np.float)
    w=np.where(np.fabs(np.sin(lats))>(latspread*5.))
    if(np.size(w)>0):
        y[w]=sigplus*np.exp(-(np.sin(lats[w])/latspread)**2/.2)
    return y

def sdotsink(sigma, sigmax):
    w=np.where(sigma>(sigmax/100.))
    y=0.0*sigma
    if(np.size(w)>0):
        y[w]=1.0*sigma*np.exp(-sigmax/sigma)
    return y

# main loop
time1 = time.clock() # time loop

nout=nrest
outskip=1000 # how often do we output the snapshots

for ncycle in np.arange(itmax+1)+nrest*outskip:
    #for ncycle in range(2): #debug option
    t = ncycle*dt

    # get vort,u,v,sigma on grid
    vortg = x.sph2grid(vortSpec)
    sig  = x.sph2grid(sigSpec)
    signative  = x.sph2grid(signativeSpec)
    divg  = x.sph2grid(divSpec)

    ug,vg = x.getuv(vortSpec,divSpec)

#    print('t=%10.5f ms' % (t*1e3*tscale))

    # compute tendencies
    tmpg1 = ug*vortg
    tmpg2 = vg*vortg
    ddivdtSpec[:,nnew], dvortdtSpec[:,nnew] = x.getVortDivSpec(tmpg1,tmpg2)
    dvortdtSpec[:,nnew] *= -1
    tmpg = x.sph2grid(ddivdtSpec[:,nnew])
    tmpg1 = ug*sig; tmpg2 = vg*sig
    tmpSpec, dsigdtSpec[:,nnew] = x.getVortDivSpec(tmpg1,tmpg2)
    dsigdtSpec[:,nnew] *= -1
    press=cs**2*np.log((sig+np.fabs(sig))/2.+sigfloor) # stabilizing equation of state (in fact, it is enthalpy)
    engy=press+0.5*(ug**2+vg**2) # energy per unit mass
    tmpSpec = x.grid2sph(engy)
    ddivdtSpec[:,nnew] += -x.lap*tmpSpec
    # update vort,div,phiv with third-order adams-bashforth.
    # forward euler, then 2nd-order adams-bashforth time steps to start
    if ncycle == 0:
        dvortdtSpec[:,nnow] = dvortdtSpec[:,nnew]
        dvortdtSpec[:,nold] = dvortdtSpec[:,nnew]
        ddivdtSpec[:,nnow] = ddivdtSpec[:,nnew]
        ddivdtSpec[:,nold] = ddivdtSpec[:,nnew]
        dsigdtSpec[:,nnow] = dsigdtSpec[:,nnew]
        dsigdtSpec[:,nold] = dsigdtSpec[:,nnew]
    elif ncycle == 1:
        dvortdtSpec[:,nold] = dvortdtSpec[:,nnew]
        ddivdtSpec[:,nold] = ddivdtSpec[:,nnew]
        dsigdtSpec[:,nold] = dsigdtSpec[:,nnew]

    vortSpec += dt*( \
    (23./12.)*dvortdtSpec[:,nnew] - (16./12.)*dvortdtSpec[:,nnow]+ \
    (5./12.)*dvortdtSpec[:,nold] )

    divSpec += dt*( \
    (23./12.)*ddivdtSpec[:,nnew] - (16./12.)*ddivdtSpec[:,nnow]+ \
    (5./12.)*ddivdtSpec[:,nold] )

    sigSpec += dt*( \
    (23./12.)*dsigdtSpec[:,nnew] - (16./12.)*dsigdtSpec[:,nnow]+ \
    (5./12.)*dsigdtSpec[:,nold] )

    signativeSpec += dt*( \
    (23./12.)*dsigdtSpec[:,nnew] - (16./12.)*dsigdtSpec[:,nnow]+ \
    (5./12.)*dsigdtSpec[:,nold] )

    sdotplus=sdotsource(lats, lons, latspread)
    sdotminus=sdotsink(sig, sigmax)
    sdotSpec=x.grid2sph(sdotplus-sdotminus)

    vortdot=sdotplus/sig*(2.*overkepler/rsphere**1.5*np.sin(lats)-vortg)
    vortdotSpec=x.grid2sph(vortdot)
    
    sigSpec += dt*sdotSpec # source term for density
    sigSpec += dt*sdotSpec # source term for density
    vortSpec += dt*vortdotSpec # source term for vorticity

    # total kinetic energy loss
    dissSpec=(vortSpec**2+divSpec**2)*(1.-hyperdiff_fact)/x.lap
    wnan=np.where(np.isnan(dissSpec))
    if(np.size(wnan)>0):
#        print str(np.size(wnan))+" nan points"
#        ii=raw_input()
        dissSpec[wnan]=0.
    dissipation=x.sph2grid(dissSpec)/2./dt
    # implicit hyperdiffusion for vort and div
    vortSpec *= hyperdiff_fact
    divSpec *= hyperdiff_fact
    #     sigSpec *= sigma_diff 

    # switch indices, do next time step
    nsav1 = nnew; nsav2 = nnow
    nnew = nold; nnow = nsav1; nold = nsav2


    if(ncycle % 100 ==0):
        print('t=%10.5f ms' % (t*1e3*tscale))

    #plot & save
    if (ncycle % outskip == 0):

        mass=sig.sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
        energy=(sig*engy).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
        
        visualize(t, nout,
                  lats, lons, 
                  vortg, divg, ug, vg, sig, signative, dissipation,
                  mass, energy,
                  engy,
                  hbump,
                  conf)

        #file I/O
        f5io.saveSim(f5, nout, t,
                     energy, mass, 
                     vortg, divg, ug, vg, sig, signative, dissipation
                     )
        nout += 1

        
#end of time cycle loop

time2 = time.clock()
print('CPU time = ',time2-time1)


