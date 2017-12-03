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
from conf import grav, rsphere
from conf import dt, omega, rsphere, sig0, overkepler, tscale
from conf import hamp, phi0, lon0, alpha, beta  #initial height perturbation parameters
from conf import efold, ndiss
from conf import ifiso, csqmin, cssqscale, kappa, mu, betamin, sigfloor # EOS parameters
from conf import itmax, outskip
from conf import ifplot
from conf import sigplus, sigmax, latspread #source and sink terms
from conf import incle, slon0
from conf import ifrestart, nrest, restartfile

##################################################
# setup up spherical harmonic instance, set lats/lons of grid
x = Spharmt(conf.nlons, conf.nlats, conf.ntrunc, conf.rsphere, gridtype='gaussian')
lons,lats = np.meshgrid(x.lons, x.lats)

# guide grids for plotting
lons1d = (180./np.pi)*x.lons-180.
lats1d = (180./np.pi)*x.lats

# initial velocity field 
ug = omega*rsphere*np.cos(lats)
vg = ug*0.

# density perturbation
hbump = hamp*np.cos(lats)*np.exp(-((lons-lon0)/alpha)**2)*np.exp(-(phi0-lats)**2/beta)

# initial vorticity, divergence in spectral space
vortSpec, divSpec =  x.getVortDivSpec(ug,vg)
vortg = x.sph2grid(vortSpec)
divg  = x.sph2grid(divSpec)

# create (hyper)diffusion factor; normal diffusion corresponds to ndiss=4
hyperdiff_fact = np.exp((-dt/efold)*(x.lap/np.abs(x.lap).max())**(ndiss/2))
sigma_diff = hyperdiff_fact
hyperdiff_expanded = (x.lap/np.abs(x.lap).max())**(ndiss/2) / efold

# sigma is an exact isothermal solution + an unbalanced bump
sig = sig0*np.exp(-(omega*rsphere)**2/csqmin/2.*(1.-np.cos(lats))) * (1. + hbump) # exact solution * (1 + perturbation)
# in pressure, there should not be any strong bumps, as we do not want artificial shock waves
pressg = sig * cs**2 / (1. + hbump) 
# vortg *= (1.-hbump*2.) # some initial condition for vorticity (cyclone?)
accflag=hbump*0.
#print accflag.max(axis=1)
#print accflag.max(axis=0)
#ii=raw_input("f")
# initialize spectral tendency arrays
ddivdtSpec  = np.zeros(vortSpec.shape+(3,), np.complex)
dvortdtSpec = np.zeros(vortSpec.shape+(3,), np.complex)
dsigdtSpec  = np.zeros(vortSpec.shape+(3,), np.complex)
dpressdtSpec  = np.zeros(vortSpec.shape+(3,), np.complex)
daccflagdtSpec = np.zeros(vortSpec.shape+(3,), np.complex)

# Cycling integers for integrator
nnew = 0
nnow = 1
nold = 2

###########################################################
# restart module:

if(ifrestart):
    vortg, divg, sig, pressg, accflag = f5io.restart(restartfile, nrest, conf)
else:
    nrest=0
        
sigSpec  = x.grid2sph(sig)
pressSpec  = x.grid2sph(pressg)
accflagSpec  = x.grid2sph(accflag)
divSpec  = x.grid2sph(divg)
vortSpec = x.grid2sph(vortg)

###################################################
# Save simulation setup to file
f5io.saveParams(f5, conf)
    
##################################################
# source/sink term
def sdotsource(lats, lons, latspread):
'''
source term for surface density:
lats -- latitudes, radians
lons -- longitudes, radians
latspread -- width of the accretion belt, radians
outputs: dSigma/dt and cosine of the angle towards the rotation direction
'''
    y=np.zeros((nlats,nlons), np.float)
    devcos=np.sin(lats)*np.cos(incle)+np.cos(lats)*np.sin(incle)*np.cos(lons-slon0)
    
    w=np.where(np.fabs(devcos)<(latspread*5.))
    if(np.size(w)>0):
        y[w]=sigplus*np.exp(-(devcos[w]/latspread)**2/.2)
        y/=(2.*np.pi)**1.5*latspread
    return y, devcos

def sdotsink(sigma, sigmax):
'''
sink term in surface density
sdotsink(sigma, sigmax)
'''
    w=np.where(sigma>(sigmax/100.))
    y=0.0*sigma
    if(np.size(w)>0):
        y[w]=1.0*sigma[w]*np.exp(-sigmax/sigma[w])
    return y

# vertically integrated enthalpy int dPi/Sigma (in local immediate re-radiation approximation)
def enthalpy(sigma, dissipation, geff):
    sigplus=(sigma+np.fabs(sigma))/2.+sigfloor
    if(ifiso):
        return np.log(sigplus)*csqmin
    else:
        posdiss=(dissipation+np.fabs(dissipation))/2.
        beta = 1.-kappa*posdiss/geff
        wlevitating=np.where(beta<=betamin) # levitating case, when beta\lesssim 0
        if(np.size(wlevitating)>0):
            print str(np.size(wlevitating))+" levitating points"
            beta[wlevitating]=betamin
        csq=cssqscale*np.sqrt(kappa*sigplus)*(posdiss)**0.25/beta
        wcold=np.where(csq<csqmin)
        if(np.size(wcold)>0):
            print str(np.size(wcold))+" cold points"
            csq[wcold]=csqmin
        return 2.*csq
    
def betasolve(y):
'''
    solves the equation x/(1-x)**0.25=y for x
'''
    x = Symbol('x')
    xs=solve(x/(1.-x)**0.25-y, x)
    return xs[0]
# main loop
time1 = time.clock() # time loop

nout=nrest

for ncycle in np.arange(itmax+1):
    #for ncycle in range(2): #debug option
    t = (ncycle+nrest*outskip)*dt

    # get vort,u,v,sigma on grid
    vortg = x.sph2grid(vortSpec)
    sig  = x.sph2grid(sigSpec)
    accflag = x.sph2grid(accflagSpec)
    divg  = x.sph2grid(divSpec)
    pressg  = x.sph2grid(pressSpec)

    ug,vg = x.getuv(vortSpec,divSpec)

#    print('t=%10.5f ms' % (t*1e3*tscale))

    # vorticity flux
    tmpg1 = ug*vortg
    tmpg2 = vg*vortg
    ddivdtSpec[:,nnew], dvortdtSpec[:,nnew] = x.getVortDivSpec(tmpg1,tmpg2)
    dvortdtSpec[:,nnew] *= -1
    tmpg = x.sph2grid(ddivdtSpec[:,nnew])
    tmpg1 = ug*sig; tmpg2 = vg*sig
    tmpSpec, dsigdtSpec[:,nnew] = x.getVortDivSpec(tmpg1,tmpg2)
    dsigdtSpec[:,nnew] *= -1
    # energy (pressure) flux:
    tmpg1 = ug*pressg; tmpg2 = vg*pressg
    tmpSpec, dpressdtSpec[:,nnew] = x.getVortDivSpec(tmpg1,tmpg2)
    geff=-grav+(ug**2+vg**2)/rsphere # effective gravity
    wunbound=np.where(geff>=0.)
    if(np.size(wunbound)>0):
        print str(np.size(wunbound))+" unbound points with geff>0"
        ii=raw_input('')
    beta = betasolve(2.e-6*sig/pressg*(-geff*sig)**0.25) # beta as a function of sigma, press, geff
    dpressdtSpec[:,nnew] *= -1
    dpressdtSpec[:,nnew] += divg * pressg / 3. /(1.-beta/2.)

    # dissipation estimates:
    dissvortSpec=vortSpec*hyperdiff_expanded #expanded exponential diffusion term
    dissdivSpec=divSpec*hyperdiff_expanded 
    wnan=np.where(np.isnan(dissvortSpec+dissdivSpec))
    if(np.size(wnan)>0):
        dissvortSpec[wnan]=0. ;  dissdivSpec[wnan]=0.
    dissug, dissvg = x.getuv(dissvortSpec, dissdivSpec)
    dissipation=(ug*dissug+vg*dissvg) # v . dv/dt_diss
    if(np.size(wunbound)>0):
        geff[wunbound]=0.
        print "ug from "+str(ug.min())+" to "+str(ug.max())
        print "vg from "+str(vg.min())+" to "+str(vg.max())
        rr=raw_input(".")
#    print "geff between "+str(geff.min())+" and "+str(geff.max())
#    rr=raw_input(".")
#    enth=enthalpy(sig, dissipation, geff)
#    print "enthalpy between "+str(enth.min())+" and "+str(enth.max())
    # csmin**2*np.log((sig+np.fabs(sig))/2.+sigfloor) # stabilizing equation of state (in fact, it is enthalpy)
    engy=0.5*(ug**2+vg**2) # kinetic energy per unit mass (merge this with baroclinic term?)
    tmpSpec = x.grid2sph(engy)
    ddivdtSpec[:,nnew] += -x.lap*tmpSpec

    # baroclinic terms in vorticity and divirgence:
    gradp1, gradp2 = x.getGrad(pressSpec)  # ; grads1, grads2 = x.getGrad(sigSpec)
    vortpressbarSpec, divpressbarSpec = x.getVortDivSpec(gradp1/sig,gradp2/sig)
    # x.grid2sph((gradp1 * grads2 - gradp2 * grads1)*7./8.)
    # x.getVortDivSpec(tmpg1,tmpg2)
    ddivdtSpec[:,nnew] += divpressbarSpec *7./8.
    dvortdtSpec[:,nnew] += vortpressbarSpec *7./8.

    # source terms in mass:
    sdotplus, sina=sdotsource(lats, lons, latspread)
    sdotminus=sdotsink(sig, sigmax)
    sdotSpec=x.grid2sph(sdotplus-sdotminus)
    dsigdtSpec[:,nnew] += sdotSpec

    # source term in vorticity
    vortdot=sdotplus/sig*(2.*overkepler/rsphere**1.5*sina-vortg)
    vortdotSpec=x.grid2sph(vortdot)
    dvortdtSpec[:,nnew] += vortdotSpec

    # energy sources and sinks:
    qplus = sig * dissipation
    qminus = 7./3./kappa * pressg/sig *(1.-beta)
    qns = (csqmin/csqscale)**4  # conversion of (minimal) speed of sound to flux
    dpressdtSpec[:,nnew] += (qplus - qminus + qns) / 3. /(1.-beta/2.)
    # passive scalar evolution:
    tmpg1 = ug*accflag; tmpg2 = vg*accflag
    tmpSpec, dacctmp = x.getVortDivSpec(tmpg1,tmpg2)
    daccflagdtSpec[:,nnew] = -dacctmp # a*div(v) - div(a*v)
    daccflagdt =  (1.-accflag) * (sdotplus/sig) + accflag * divg 
    daccflagdtSpec[:,nnew] += x.grid2sph(daccflagdt)
    
    # update vort,div,phiv with third-order adams-bashforth.
    # forward euler, then 2nd-order adams-bashforth time steps to start
    if ncycle == 0:
        dvortdtSpec[:,nnow] = dvortdtSpec[:,nnew]
        dvortdtSpec[:,nold] = dvortdtSpec[:,nnew]
        ddivdtSpec[:,nnow] = ddivdtSpec[:,nnew]
        ddivdtSpec[:,nold] = ddivdtSpec[:,nnew]
        dsigdtSpec[:,nnow] = dsigdtSpec[:,nnew]
        dsigdtSpec[:,nold] = dsigdtSpec[:,nnew]
        dpressdtSpec[:,nnow] = dpressdtSpec[:,nnew]
        dpressdtSpec[:,nold] = dpressdtSpec[:,nnew]
        daccflagdtSpec[:,nnow] = daccflagdtSpec[:,nnew]
        daccflagdtSpec[:,nold] = daccflagdtSpec[:,nnew]
    elif ncycle == 1:
        dvortdtSpec[:,nold] = dvortdtSpec[:,nnew]
        ddivdtSpec[:,nold] = ddivdtSpec[:,nnew]
        dsigdtSpec[:,nold] = dsigdtSpec[:,nnew]
        dpressdtSpec[:,nold] = dpressdtSpec[:,nnew]
        daccflagdtSpec[:,nold] = daccflagdtSpec[:,nnew]

    vortSpec += dt*( \
    (23./12.)*dvortdtSpec[:,nnew] - (16./12.)*dvortdtSpec[:,nnow]+ \
    (5./12.)*dvortdtSpec[:,nold] )

    divSpec += dt*( \
    (23./12.)*ddivdtSpec[:,nnew] - (16./12.)*ddivdtSpec[:,nnow]+ \
    (5./12.)*ddivdtSpec[:,nold] )

    sigSpec += dt*( \
    (23./12.)*dsigdtSpec[:,nnew] - (16./12.)*dsigdtSpec[:,nnow]+ \
    (5./12.)*dsigdtSpec[:,nold] )

    pressSpec += dt*( \
    (23./12.)*dpressdtSpec[:,nnew] - (16./12.)*dpressdtSpec[:,nnow]+ \
    (5./12.)*dpressdtSpec[:,nold] )

    accflagSpec += dt*( \
    (23./12.)*daccflagdtSpec[:,nnew] - (16./12.)*daccflagdtSpec[:,nnow]+ \
    (5./12.)*daccflagdtSpec[:,nold] )

    # total kinetic energy loss
    #    dissSpec=(vortSpec**2+divSpec**2)*(1.-hyperdiff_fact)/x.lap
    # implicit hyperdiffusion for vort and div
    vortSpec *= hyperdiff_fact
    divSpec *= hyperdiff_fact
    sigSpec *= sigma_diff 
    pressSpec *= sigma_diff 
#    accflagSpec *= sigma_diff 

    # switch indices, do next time step
    nsav1 = nnew; nsav2 = nnow
    nnew = nold; nnow = nsav1; nold = nsav2

    if(ncycle % floor(outskip/10) ==0):
        print('t=%10.5f ms' % (t*1e3*tscale))

    #plot & save
    if (ncycle % outskip == 0):

        mass=sig.sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
#        mass_acc=(sig*accflag).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
#        mass_native=(sig*(1.-accflag)).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
        energy=(sig*engy).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
        if(ifplot):
            visualize(t, nout,
                      lats, lons, 
                      vortg, divg, ug, vg, sig, press, accflag, dissipation, 
                      #                  mass, energy,
                      engy,
                      hbump,
                      rsphere,
                      conf)

        #file I/O
        f5io.saveSim(f5, nout, t,
                     energy, mass, 
                     vortg, divg, ug, vg, sig, press, accflag, dissipation
                     )
        nout += 1

        
#end of time cycle loop

time2 = time.clock()
print('CPU time = ',time2-time1)

# ffmpeg -f image2 -r 15 -pattern_type glob -i 'out/swater*.png' -pix_fmt yuv420p -b 4096k out/swater.mp4
