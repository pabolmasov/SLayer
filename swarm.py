"""
Main module of SLayer, supposed to be run under ipython as 
> %run swarm.py

All the parameters are set in conf.py
"""

import numpy as np
import scipy.ndimage as spin
import shtns
import matplotlib.pyplot as plt
import time
import os
import h5py
from sympy.solvers import solve
from sympy import Symbol
import scipy.interpolate as si

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
from conf import dt_cfl, omega, rsphere, sig0, sigfloor, overkepler, tscale, dtout
from conf import bump_amp, bump_phi0, bump_lon0, bump_alpha, bump_beta  #initial perturbation parameters
from conf import efold, ndiss, efold_diss
from conf import csqmin, csqinit, cssqscale, kappa, mu, betamin # EOS parameters
from conf import itmax, outskip
from conf import ifplot
from conf import sigplus, sigmax, latspread #source and sink terms
from conf import incle, slon0
from conf import ifrestart, nrest, restartfile
from conf import ewind
from conf import tfric
from conf import ifwindlosses

############################
# beta calibration
bmin=betamin ; bmax=1.-betamin ; nb=1000
# hard limits for stability; bmin\sim 1e-7 approximately corresponds to Coulomb coupling G\sim 1,
# hence there is even certain physical meaning in bmin
b=(bmax-bmin)*((np.arange(nb)+0.5)/np.double(nb))+bmin
bx=b/(1.-b)**0.25
b[0]=0. ; bx[0]=0.  # ; b[nb-1]=1e3 ; bx[nb-1]=1.
betasolve_p=si.interp1d(bx, b, kind='linear', bounds_error=False, fill_value=1.)
#(bmin,bmax)) # as a function of pressure
betasolve_e=si.interp1d(bx/(1.-b/2.)/3., b, kind='linear', bounds_error=False,fill_value=1.)
#fill_value=(bmin,bmax)) # as a function of energy
# for k in np.arange(nb):
#    print str(bx[k])+" -> "+str(b[k])+"\n"
# rr=raw_input("d")
######################################

##################################################
# setup up spherical harmonic instance, set lats/lons of grid
x = Spharmt(conf.nlons, conf.nlats, conf.ntrunc, conf.rsphere, gridtype='gaussian')
lons,lats = np.meshgrid(x.lons, x.lats)

# guide grids for plotting
lons1d = (180./np.pi)*x.lons-180.
lats1d = (180./np.pi)*x.lats

############
# time step
dt=dt_cfl # if we are in trouble, dt=1./(1./dtcfl + 1./dtthermal)

#######################################################
## initial conditions: ###
# initial velocity field 
ug = 2.*omega*np.cos(lats)*rsphere # without R, as we are using vorticity and we do not want to multiply and then to divide by the radius
vg = ug*0.

# density perturbation
hbump = bump_amp*np.cos(lats)*np.exp(-((lons-bump_lon0)/bump_alpha)**2)*np.exp(-(bump_phi0-lats)**2/bump_beta)

# initial vorticity, divergence in spectral space
vortSpec, divSpec =  x.getVortDivSpec(ug/rsphere,vg/rsphere) 
vortg = x.sph2grid(vortSpec)
vortgNS = x.sph2grid(vortSpec) # rotation of the neutron star 
divg  = x.sph2grid(divSpec)

# create (hyper)diffusion factor; normal diffusion corresponds to ndiss=4
hyperdiff_fact = np.exp((-dt/efold)*(x.lap/np.abs(x.lap).max())**(ndiss/2))
sigma_diff = hyperdiff_fact
hyperdiff_expanded = (x.lap/np.abs(x.lap).max())**(ndiss/2) / efold
diss_diff = np.exp((-dt/efold_diss)*(x.lap/np.abs(x.lap).max())**(ndiss/2)) # -- additional diffusion factor applied to energy dissipation (as there is high-frequency noise in dissipation function that we do not want to be introduced again through dissipation function)
# supposedly a stationary solution:
sig=sig0*np.exp(0.5*(omega*rsphere*np.cos(lats))**2/csqinit)
# *(np.cos(lats))**((omega*rsphere)**2/csqinit)+sigfloor
# print "initial sigma: "+str(sig.min())+" to "+str(sig.max())
# ii=raw_input("")
# in pressure, there should not be any strong bumps, as we do not want artificial shock waves
pressg = sig * csqinit / (1. + hbump)
geff=-grav+(ug**2+vg**2)/rsphere # effective gravity
sigpos=(sig+np.fabs(sig))/2. # we need to exclude negative sigma points from calculation
beta = betasolve_p(cssqscale*sig/pressg*np.sqrt(np.sqrt(-geff*sigpos))) # beta as a function of sigma, press, geff
energyg = pressg / 3. / (1.-beta/2.)
# vortg *= (1.-hbump*2.) # some initial condition for vorticity (cyclone?)
accflag=hbump*0.
#print accflag.max(axis=1)
#print accflag.max(axis=0)
#ii=raw_input("f")
# initialize spectral tendency arrays
ddivdtSpec  = np.zeros(vortSpec.shape, np.complex)
dvortdtSpec = np.zeros(vortSpec.shape, np.complex)
dsigdtSpec  = np.zeros(vortSpec.shape, np.complex)
# dpressdtSpec  = np.zeros(vortSpec.shape, np.complex)
denergydtSpec  = np.zeros(vortSpec.shape, np.complex)
daccflagdtSpec = np.zeros(vortSpec.shape, np.complex)

###########################################################
# restart module:

if(ifrestart):
    vortg, divg, sig, energyg, accflag, t0 = f5io.restart(restartfile, nrest, conf)
else:
    t0=0.
    nrest=0
        
sigSpec  = x.grid2sph(sig)
# pressSpec  = x.grid2sph(pressg)
energySpec  = x.grid2sph(energyg)
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
    
    #    w=np.where(np.fabs(devcos)<(latspread*5.))
    #    if(np.size(w)>0):
    y=sigplus*np.exp(-(devcos/latspread)**2/2.)
    #        y/=(2.*np.pi)**1.5*latspread
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

# main loop
time1 = time.clock() # time loop

nout=nrest ;  t=t0 ; tstore=t0 # starting counters

for ncycle in np.arange(itmax+1):
    # for ncycle in range(2): # debug option
    # get vort,u,v,sigma on grid
    vortg = x.sph2grid(vortSpec)
    sig  = x.sph2grid(sigSpec)
    accflag = x.sph2grid(accflagSpec)
    divg  = x.sph2grid(divSpec)
    energyg  = x.sph2grid(energySpec)
    ug,vg = x.getuv(vortSpec,divSpec) # velocity components
    geff=-grav+(ug**2+vg**2)/rsphere # effective gravity
    geff=(geff-np.fabs(geff))/2. # only negative geff
    sigpos=(sig+np.fabs(sig))/2. # we need to exclude negative sigma points from calculation
    # there could be bias if many sigma<0 points appear
    beta = betasolve_e(cssqscale*sig/energyg*np.sqrt(np.sqrt(-geff*sigpos))) # beta as a function of sigma, energy, and geff
    wbnan=np.where(np.isnan(beta))
    if(np.size(wbnan)>0):
        print "beta = "+str(beta[wbnan])
        print "geff = "+str(geff[wbnan])
        print "sig = "+str(sig[wbnan])
        print "energy = "+str(energyg[wbnan])
        ii=raw_input("betanan")
    pressg=energyg / 3. / (1.-beta/2.)
    # vorticity flux
    tmpg1 = ug*vortg
    tmpg2 = vg*vortg
    ddivdtSpec, dvortdtSpec = x.getVortDivSpec(tmpg1 /rsphere,tmpg2 /rsphere) # all the nablas should contain an additional 1/R multiplier
    dvortdtSpec *= -1
    tmpg = x.sph2grid(ddivdtSpec)
    tmpg1 = ug*sig; tmpg2 = vg*sig
    tmpSpec, dsigdtSpec = x.getVortDivSpec(tmpg1 /rsphere,tmpg2 /rsphere) # all the nablas should contain an additional 1/R multiplier
    dsigdtSpec *= -1
    # energy (pressure) flux:
    tmpg1 = ug*energyg; tmpg2 = vg*energyg
    tmpSpec, denergydtSpec = x.getVortDivSpec(tmpg1 /rsphere,tmpg2 /rsphere) # all the nablas should contain an additional 1/R multiplier
    wunbound=np.where(geff>=0.) # extreme case; we can be unbound due to pressure
    if(np.size(wunbound)>0):
        print str(np.size(wunbound))+" unbound points with geff>0"
        #        ii=raw_input('')
        # maybe not so bad if geff=0 exactly? sitting at the Eddington limit...
        geff[wunbound]=0.
    denergydtSpec *= -1
    denergydtSpec += x.grid2sph(divg * pressg)

    # dissipation estimates:
    dissvortSpec=vortSpec*hyperdiff_expanded #expanded exponential diffusion term
    dissdivSpec=divSpec*hyperdiff_expanded 
    wnan=np.where(np.isnan(dissvortSpec+dissdivSpec))
    if(np.size(wnan)>0):
        dissvortSpec[wnan]=0. ;  dissdivSpec[wnan]=0.
    dissug, dissvg = x.getuv(dissvortSpec*rsphere, dissdivSpec*rsphere)
    dissipation=(ug*dissug+vg*dissvg) # v . dv/dt_diss
    dissipation = x.sph2grid(x.grid2sph(dissipation)*diss_diff) # smoothing dissipation 
    if(np.size(wunbound)>0):
        geff[wunbound]=0.
        print "ug from "+str(ug.min())+" to "+str(ug.max())
        print "vg from "+str(vg.min())+" to "+str(vg.max())
        rr=raw_input(".")
    kenergy=0.5*(ug**2+vg**2) # kinetic energy per unit mass (merge this with baroclinic term?)
    tmpSpec = x.grid2sph(kenergy)
    ddivdtSpec += -x.lap*tmpSpec

    # baroclinic terms in vorticity and divirgence:
    gradp1, gradp2 = x.getGrad(x.grid2sph(pressg)/rsphere)  # ; grads1, grads2 = x.getGrad(sigSpec)
    vortpressbarSpec, divpressbarSpec = x.getVortDivSpec(gradp1/sig/rsphere,gradp2/sig/rsphere)
    # each nabla should have its 1/R multiplier
    ddivdtSpec += -divpressbarSpec 
    dvortdtSpec += -vortpressbarSpec

    # energy sources and sinks:   
    qplus = sigpos * dissipation 
    qminus = (-geff) * sigpos / 3. / (1.+kappa*sigpos) * (1.-beta) 
    qns = (csqmin/cssqscale)**4  # conversion of (minimal) speed of sound to flux

    # Bernoulli constant:
    B=(ug**2+vg**2)/2.+3.75*pressg/sig-1./rsphere # v^2/2+15/4 Pi/Sigma -1/R
        
    # source terms in mass:
    sdotplus, sina=sdotsource(lats, lons, latspread)
    sdotminus=sdotsink(sig, sigmax)
    sdotSpec=x.grid2sph(sdotplus-sdotminus)
    dsigdtSpec += sdotSpec

    # source term in vorticity
    vortdot=sdotplus/sig*(2.*overkepler/rsphere**1.5*sina-vortg)+(vortgNS-vortg)/tfric # +sdotminus/sig*vortg
    divdot=-divg/tfric # friction term for divergence
    vortdotSpec=x.grid2sph(vortdot)
    divdotSpec=x.grid2sph(divdot)
    dvortdtSpec += vortdotSpec
    ddivdtSpec += divdotSpec
    denergydtSpec += x.grid2sph((qplus - qminus + qns)+(sdotplus*csqmin-pressg/sig*sdotminus) * 3. * (1.-beta/2.))
    denergyg = x.sph2grid(denergydtSpec) # maybe we can optimize this?
    dt_thermal=1./(np.fabs(denergyg)/energyg).max()
    if( dt_thermal <= (5. * dt) ): # very rapid thermal evolution; we can artificially 
        dt=1./(1./dt_cfl+1./dt_thermal)
    else:
        dt=dt_cfl
    # passive scalar evolution:
    tmpg1 = ug*accflag; tmpg2 = vg*accflag
    tmpSpec, dacctmp = x.getVortDivSpec(tmpg1,tmpg2)
    daccflagdtSpec = -dacctmp # a*div(v) - div(a*v)
    daccflagdt =  (1.-accflag) * (sdotplus/sig) + accflag * divg 
    daccflagdtSpec += x.grid2sph(daccflagdt)
    
    # at last, the time step
    t += dt
    vortSpec += dvortdtSpec * dt

    divSpec += ddivdtSpec * dt

    sigSpec += dsigdtSpec * dt

    energySpec += denergydtSpec * dt

    accflagSpec += daccflagdtSpec * dt

    # implicit hyperdiffusion for vort and div
    vortSpec *= hyperdiff_fact
    divSpec *= hyperdiff_fact
    sigSpec *= sigma_diff 
    energySpec *= sigma_diff 
    #    accflagSpec *= sigma_diff 

    #    if(ncycle % np.floor(outskip/10) ==0):
    #        print('t=%10.5f ms' % (t*1e3*tscale))
    if(ncycle % 100 ==0 ): # make sure it's alive
        print('t=%10.5f ms' % (t*1e3*tscale))
        print " dt(CFL) = "+str(dt_cfl)
        print " dt(thermal) = "+str(dt_thermal)
        print "dt = "+str(dt)

    #plot & save
    if ( t >  (tstore+dtout)):
        tstore=t
        if(ifplot):
            visualize(t, nout,
                      lats, lons, 
                      vortg, divg, ug, vg, sig, pressg, beta, accflag, dissipation, 
#                      hbump,
                      rsphere,
                      conf)

        #file I/O
        f5io.saveSim(f5, nout, t,
                     vortg, divg, ug, vg, sig, energyg, beta,
                     accflag, dissipation,
                     conf)
        nout += 1

        
#end of time cycle loop

time2 = time.clock()
print('CPU time = ',time2-time1)

# ffmpeg -f image2 -r 15 -pattern_type glob -i 'out/swater*.png' -pix_fmt yuv420p -b 4096k out/swater.mp4
