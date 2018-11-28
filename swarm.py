"""
Main module of SLayer, supposed to be run under ipython as 
> %run swarm.py

All the parameters are set in conf.py by default; output directory is called 'out' and is created if needed. Main output is called out/run.hdf5 and is supplemented by png snapshots if ifplot=True

If you want to use several setup files and output directories, you can try to run swarm with additional parameters:
> %run swarm.py your_conf your_out
where your setup is written in your_conf.py and the output directory is your_out
"""
from __future__ import print_function
from __future__ import division

from builtins import input
from builtins import str
from past.utils import old_div
import numpy as np
import scipy.ndimage as spin
import shtns
# import matplotlib.pyplot as plt
import time
import os
import h5py
# from sympy.solvers import solve
# from sympy import Symbol
import scipy.interpolate as si
import imp
import sys

#Code modules:
from spharmt import Spharmt 
import f5io as f5io #module for file I/O
import conf as conf #importing simulation setup module

from timer import Timer

####################################
# extra arguments (allow to run several slayers in parallel and write results to arbitrary output directories)
f5io.outdir = 'out'
if(np.size(sys.argv)>1):
    print("launched with arguments "+str(', '.join(sys.argv)))
    # new conf file
    newconf=sys.argv[1]
    print("loaded "+newconf+".py instead of the standard setup file conf.py")
    if(np.size(sys.argv)>2):
        # alternative output directory
        f5io.outdir = sys.argv[2]
else:
    newconf='conf'
fp, pathname, description = imp.find_module(newconf)
imp.load_module('conf', fp, pathname, description)
fp.close()
if not os.path.exists(f5io.outdir):
    os.makedirs(f5io.outdir)

print('output directory '+str(f5io.outdir))
f5 = h5py.File(f5io.outdir+'/run.hdf5', "w")

#import simulation parameters to global scope
from conf import logSE # if we are working with logarithms of Sigma and Energy
from conf import nlons, nlats # dimensions of the simulation area
from conf import grav, rsphere, mass1 # gravity, radius of the star, mass of the star (solar)
from conf import omega, sig0, overkepler, tscale # star rotation frequency, initial density level, deviation from Kepler for the falling matter
from conf import ifscaledt, dt_cfl_factor, dt_out_factor # scaling for the time steps
from conf import ifscalediff
from conf import bump_amp, bump_lat0, bump_lon0, bump_dlon, bump_dlat  #initial perturbation parameters
from conf import ktrunc, ndiss, ktrunc_diss # e-folding time scale for the hyper-diffusion, order of hyper-diffusion, e-folding time for dissipation smoothing
from conf import ddivfac, jitterskip
from conf import csqmin, csqinit, cssqscale, kappa, mu, betamin, sigmafloor, energyfloor # physical parameters 
from conf import isothermal, gammainit, kinit # initial EOS
from conf import outskip, tmax # frequency of diagnostic outputs, maximal time
from conf import ifplot # if we make plots
from conf import sigplus, latspread, incle, slon0, tturnon # source term
from conf import ifrestart, nrest, restartfile # restart setup
from conf import tfric, tdepl # interaction with the NS surface: friction and depletion times
from conf import iftwist, twistscale # twist test parameters
from conf import eps_deformation
# from conf import sigmascale # not used directly

if(ifplot):
    from plots import visualize

from jitter import jitternod, jitterturn
    
############################
# beta calibration
bmin=betamin ; bmax=1.-betamin ; nb=10000
# hard limits for stability; bmin\sim 1e-7 approximately corresponds to Coulomb coupling G\sim 1,
# hence there is even certain physical meaning in bmin
# "beta" part of the main loop is little-time-consuming independently of nb
b = (bmax-bmin)*(old_div((np.arange(nb)+0.5),np.double(nb)))+bmin
bx = b/(1.-b)**0.25
b[0]=0. ; bx[0]=0.0  # ; b[nb-1]=1e3 ; bx[nb-1]=1.
betasolve_p=si.interp1d(bx, b, kind='linear', bounds_error=False, fill_value=1.)
# as a function of pressure
betasolve_e=si.interp1d(bx/(1.-b/2.)/3., b, kind='linear', bounds_error=False,fill_value=1.)
# as a function of energy
######################################

##################################################
# setup up spherical harmonic instance, set lats/lons of grid
x = Spharmt(conf.nlons, conf.nlats, conf.ntrunc, conf.rsphere, gridtype='gaussian')
x1 = Spharmt(3*conf.nlons, 3*conf.nlats, 3*conf.ntrunc, conf.rsphere, gridtype='gaussian')
lons,lats = np.meshgrid(x.lons, x.lats)
############
# time steps
dx=np.fabs((lons[1:-1,1:]-lons[1:-1,:-1])*np.cos(lats[1:-1,:-1])).min()/2. * rsphere
dy=np.fabs(x.lats[1:]-x.lats[:-1]).min()/2. * rsphere
dt_cfl = dt_cfl_factor / (1./dx + 1./dy) # basic CFL limit for light velocity
print("dt(CFL) = "+str(dt_cfl)+"GM/c**3 = "+str(dt_cfl*tscale)+"s")
dt=dt_cfl 
dt_out=dt_out_factor*rsphere**(1.5)/np.sqrt(mass1) # time step for output (we need to resolve the dynamic time scale)
print("dt_out = "+str(dt_out)+"GM/c**3 = "+str(dt_out*tscale)+"s")
#######################################################
## initial conditions: ###
# initial velocity field  (pure rotation)
ug = omega*np.cos(lats)*rsphere
vg = 0.00*omega*np.sin(lats)*np.cos(lons)
if(iftwist):
    ug *= (lats/twistscale) / np.sqrt(1.+(lats/twistscale)**2) # twisted sphere test
ug0 = omega*np.cos(lats)*rsphere

# initial vorticity, divergence in spectral space
vortSpec, divSpec = x.getVortDivSpec(ug,vg) 
vortg = x.sph2grid(vortSpec)
# vortg += 2.*omega*rsphere*np.sin(lats)*np.cos(lons)*0.01
vortgNS = vortg # rotation of the neutron star 
divg  = x.sph2grid(divSpec)

# create (hyper)diffusion factor; normal diffusion corresponds to ndiss=2 (x.lap is already nabla^2)
# print(x.lap)
lapmin=np.abs(x.lap[np.abs(x.lap.real)>0.]).min()
lapmax=np.abs(x.lap[np.abs(x.lap.real)>0.]).max()
hyperdiff_expanded = (-x.lap/(lapmax*ktrunc**2))**(ndiss/2) # positive! let us care somehow about the mean flow
# hyperdiff_expanded = hyperdiff_expanded - hyperdiff_expanded[hyperdiff_expanded>0.].min()
# hyperdiff_expanded[0] = 0. # care for the overall rotation trend 
hyperdiff_fact = np.exp(-hyperdiff_expanded*dt) # dt will change in the main loop
# print(hyperdiff_expanded)
# input('fdslkjsa')
div_diff =  np.exp(-ddivfac*hyperdiff_expanded*dt)# divergence factor enhanced. 
sigma_diff = hyperdiff_fact # sigma and energy are also artificially smoothed
if(ktrunc_diss>0.):
    diss_diff = np.exp(-hyperdiff_expanded * (ktrunc / ktrunc_diss)**(ndiss) * dt)

# initialize spectral tendency arrays
ddivdtSpec  = np.zeros(vortSpec.shape, np.complex)
dvortdtSpec = np.zeros(vortSpec.shape, np.complex)
dsigdtSpec  = np.zeros(vortSpec.shape, np.complex)
denergydtSpec  = np.zeros(vortSpec.shape, np.complex)
daccflagdtSpec = np.zeros(vortSpec.shape, np.complex)

###########################################################
# restart module:
if(ifrestart):
    vortg, divg, sig, energyg, accflag, t0 = f5io.restart(restartfile, nrest, conf)
else:
    t0=0.
    nrest=0
    # supposedly a stationary solution:
    if(isothermal): # works fine if rotation is very slow; otherwise, the contrast in density between the poles and the equator
        sig=sig0*np.exp(-0.5*(omega*rsphere*np.sin(lats))**2/csqinit)
        pressg = sig * csqinit
    else:
        if(gammainit == 0.): # constant-Sigma solution; poles are very cold in this solution, but the contrasts in density are minimal
            print("sigma=const")
            sig=np.cos(lats)*0.+sig0
            pressg=sig*csqinit # +0.5*(omega*rsphere*np.cos(lats))**2) # check this formula!
            #            ii=input("/")
        else: # free gammainit "polytropic" relation \Pi \propto \Sigma^\Gammainit
            sig=(sig0**(gammainit-1.)+0.5*(gammainit-1.)/gammainit*(omega*rsphere*np.cos(lats))**2/kinit)**(1./(gammainit-1.))
            pressg=kinit*sig**gammainit
    # density perturbation
    hbump = bump_amp*np.exp(-((lons-bump_lon0)/bump_dlon)**2/2.)*np.exp(-((lats-bump_lat0)/bump_dlat)**2/2.)
    sig*=hbump+1.
    print("sigma = "+str(sig.min())+" to "+str(sig.max()))
    print("press = "+str(pressg.min())+" to "+str(pressg.max()))
    #    sig=x.sph2grid(x.grid2sph(sig))
    #    print("sigma = "+str(sig.min())+" to "+str(sig.max()))
    #    ii=input('s')
    # in pressure, there should not be any strong bumps, as we do not want artificial shock waves
    geff=-grav+(ug**2+vg**2-ug0**2)/rsphere
    sigpos=(sig+np.fabs(sig))/2. # we need to exclude negative sigma points from calculation (should there be any?)
    beta = betasolve_p(cssqscale*sig/pressg*np.sqrt(np.sqrt(-geff*sigpos))) # beta as a function of sigma, press, geff
    energyg = pressg * 3. * (1.-beta/2.)
    accflag = hbump*0. # initially, the tracer is 0 everywhere

# spectral arrays... 
if(logSE):
    sigSpec  = x.grid2sph(np.log(sig)) # we use logarithms of Sigma and E as both quantities always have the same sign but change by several orders of magnitude
    energySpec  = x.grid2sph(np.log(energyg))
else:
    sigSpec  = x.grid2sph(sig)
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
    y=sigplus*np.exp(-0.5*(devcos/latspread)**2)
    return y, devcos

def sdotsink(sigma):
    '''
    NOT USED NOW!
    sink term in surface density
    sdotsink(sigma) 
    could be made more elaborate
    '''
    if(tdepl>0.):
        return sigma/tdepl
    else:
        return sigma*0.

# sources:
sdotmax, sina = sdotsource(lats, lons, latspread) # surface density source and sine of the distance towards the rotation axis of the falling matter (normally, slightly offset to the rotation of the star)
vort_source=2.*overkepler/rsphere**1.5*sina
# * np.exp(-(sina/latspread)**2)+vortgNS*(1.-np.exp(-(sina/latspread)**2))
# !!! let us try again a smooth version
# *np.exp(-(sina/latspread)**2)+vortgNS*(1.-np.exp(-(sina/latspread)**2)) # vorticity source ; divergence source is assumed zero
# if Omega_source = Omega * (1-0.75 sin^2(a)), vort \propto sina*(1.-0.75*(2.*sina**2-1.)/(2.*latspread))
ud,vd = x.getuv(x.grid2sph(vort_source),x.grid2sph(vort_source)*0.) # velocity components of the source
beta_acc = 1. # gas-dominated matter
# beta_acc = 0. # radiation-dominated matter
csqinit_acc = (overkepler*latspread)**2 / rsphere
energy_source_max = sdotmax*csqinit_acc* 3. * (1.-beta_acc/2.) *0. 

# main loop
time1 = time.clock() # time loop

nout=nrest ;  t=t0 ; tstore=t0 # starting counters
ncycle=0

# Timer for profiling
timer = Timer(["total", "step", "io"], 
            ["init-grid", "beta", "fluxes", 
             "diffusion","baroclinic", "source-terms",
             "passive-scalar", "time-step", "diffusion2"])
timer.start("total")


while(t<(tmax+t0)):
    ##################################################
    # initialize on grid
    timer.start_comp("init-grid")

    vortg = x.sph2grid(vortSpec)
    if(logSE):
        lsig  = x.sph2grid(sigSpec)
        sig = np.exp(lsig)
        sigpos = sig
        lenergyg  = x.sph2grid(energySpec)
        energyg = np.exp(lenergyg)
        energypos = energyg
    else:
        sig  = x.sph2grid(sigSpec)-sigmafloor
        sigpos = (sig + np.abs(sig))/2.+sigmafloor
        sig+=sigmafloor
        lsig = np.log(sigpos)
        energyg  = x.sph2grid(energySpec)-energyfloor
        energypos = (energyg + np.abs(energyg))/2.+energyfloor
        energyg+=energyfloor
        lenergyg = np.log(energypos)
    accflag = x.sph2grid(accflagSpec)
    divg  = x.sph2grid(divSpec)
    ug,vg = x.getuv(vortSpec,divSpec) # velocity components

    timer.stop_comp("init-grid")
    ##################################################

    geff=-grav+(ug**2+vg**2-ug0**2)/rsphere # effective gravity
    geff=(geff-np.fabs(geff))/2. # only negative geff

    ##################################################
    # pressure ratio 
    timer.start_comp("beta")

    beta = betasolve_e(cssqscale*sigpos/energypos*np.sqrt(np.sqrt(-geff*sigpos))) # beta as a function of sigma, energy, and geff
    wbnan=np.where(np.isnan(beta))

    if (np.size(wbnan)>0) | (geff.max()>0.): #  | (sig.min()<0.) | (energyg.min()<0.):
        print("sigmin = "+str(sig.min()))
        print("energymin = "+str(energyg.min()))
        wpressmin=energyg.argmin()
        print("beta[] = "+str(np.reshape(beta,np.size(beta))[wpressmin]))
        print("beta = "+str(beta[wbnan]))
        print("geff = "+str(geff[wbnan]))
        print("ug = "+str(ug[wbnan]))
        print("vg = "+str(vg[wbnan]))
        print("sig = "+str(sig[wbnan]))
        print("energy = "+str(energyg[wbnan]))
        print("lenergy = "+str(lenergyg[wbnan]))
        # ii=input("betanan")
        f5.close()
        if(ifplot):
            visualize(t, -1, lats, lons, 
                      vortg, divg, ug, vg, sig, pressg, beta, accflag, qminus, qplus+qns, 
                      conf, f5io.outdir) # crash visualization
        sys.exit()
    pressg = energyg / 3. / (1.-beta/2.) # beta is not the source of all evil
    cssqmax = (pressg/sig).max() # estimate for maximal speed of sound
    vsqmax = (ug**2+vg**2).max()

    # shock watch !!!
    divmachsq = divg**2 * (dx**2 + dy**2) / (pressg/sig) # Mach^2 for divergence ; divmachsq \gtrsim 1 means a shock wave
    divg *= divg / np.sqrt(divmachsq + 1.)
    
    timer.stop_comp("beta")
    ##################################################
    timer.start_comp("fluxes")

    # vorticity flux
    #    tmpg1 = ug*vortg ;    tmpg2 = vg*vortg
    ddivdtSpec, dvortdtSpec = x.getVortDivSpec(ug*vortg, vg*vortg ) # all the nablas already contain an additional 1/R multiplier
    dvortdtSpec *= -1.
    # divergence flux
    ddivdtSpec += - x.lap * x.grid2sph(ug**2+vg**2+ug0**2) / 2. 
    #    tmpg = x.sph2grid(ddivdtSpec)
    #    tmpg1 = ug*lsig;  tmpg2 = vg*lsig
    if(logSE):
        tmpSpec, dsigdtSpec = x.getVortDivSpec(ug*lsig, vg*lsig ) 
        tmpSpec, denergydtSpec = x.getVortDivSpec(ug*lenergyg, vg*lenergyg) 
    else:
        tmpSpec, dsigdtSpec = x.getVortDivSpec(ug*sig, vg*sig )
        tmpSpec, denergydtSpec = x.getVortDivSpec(ug*energyg, vg*energyg)
    denergydtSpec *= -1.
    dsigdtSpec *= -1.
    if(logSE):
        dsigdtSpec+=x.grid2sph((lsig-1.) * divg) # dlnSigma/dt = div(v) * (lnSigma -1) - div(v ln Sigma) + ...
        denergydtSpec += x.grid2sph((lenergyg-1.) * divg)

    timer.stop_comp("fluxes")
    ##################################################
    # diffusion 
    timer.start_comp("diffusion")
    
    #    hyperdiff_perdt=hyperdiff_expanded
    # dissipation estimates:
    if(ifscalediff):
        dtscale = dt_cfl/dt # fixed smoothing per single time step
    else:
        dtscale = 1. # fixed smoothing per unit time
    if(tfric>0.):
        dissvortSpec=vortSpec*(hyperdiff_expanded*dtscale+1./tfric) #expanded exponential diffusion term
        dissdivSpec=divSpec*(ddivfac*hyperdiff_expanded*dtscale+1./tfric) # need to incorporate for omegaNS in the friction term
    else:
        dissvortSpec=vortSpec*hyperdiff_expanded*dtscale #expanded exponential diffusion term
        dissdivSpec=divSpec*hyperdiff_expanded*dtscale # need to incorporate for omegaNS in the friction term
        
    dissug, dissvg = x.getuv(dissvortSpec, dissdivSpec)
    dissipation = (ug*dissug+vg*dissvg) # -v . dv/dt_diss  # positive if it is real dissipation, because hyperdiff_expanded is positive
    #    lost_angmoz = sig * dissug * np.sin(lats) * rsphere # angular momentum loss (z component)
    # energy sources and sinks:   
    qplus = sig * dissipation  
    # qminus = (-geff/kappa) * energyg / (energyg+energyfloor) * (1.-beta)  # unphysical, but small energies are frozen
    qminus = (-geff/kappa) * (1.-beta) # vertical integration excludes rho or sigma; no 3 here (see section "vertical structure" in the paper)
    qns = (csqmin/cssqscale)**4  # conversion of (minimal) speed of sound to flux
    timer.stop_comp("diffusion")
    ##################################################
    # baroclinic terms in vorticity and divirgence:
    timer.start_comp("baroclinic")

    gradp1, gradp2 = x.getGrad(x.grid2sph(pressg))  
    vortpressbarSpec, divpressbarSpec = x.getVortDivSpec(gradp1/sig,gradp2/sig) # each nabla already has its rsphere
    ddivdtSpec += -divpressbarSpec 
    dvortdtSpec += -vortpressbarSpec

    timer.stop_comp("baroclinic")
    ##################################################
    # source terms in mass:
    timer.start_comp("source-terms")

    #     sdotplus, sina=sdotsource(lats, lons, latspread) # sufficient to calculate once!
    #    sdotminus=sdotsink(sig)
    #    sdotplus, sina = sdotsource(lats, lons, latspread, t)
    if(tturnon>0.):
        sdotplus = sdotmax * (1.-np.exp(-t/tturnon))
        energy_source = energy_source_max * (1.-np.exp(-t/tturnon))
    else:
        sdotplus = sdotmax
        energy_source = energy_source_max 
    #    sdotSpec=x.grid2sph(sdotplus/sig-1./tdepl)
    if(tdepl>0.):
        lsdot = sdotplus/sig-1./tdepl
    else:
        lsdot = sdotplus/sig

    if(logSE):
        dsigdtSpec_srce = x.grid2sph(lsdot)
    else:
        dsigdtSpec_srce = x.grid2sph(lsdot*sig)
    # source term in vorticity
    #    domega=(vort_source-vortg) # difference in net vorticity
    
    vortdot =  sdotplus/sig * (vort_source-vortg)
    divdot  = -sdotplus/sig * divg
    if(tfric>0.):
        vortdot += (vortgNS-vortg)/tfric # +sdotminus/sig*vortg
        divdot  += -divg/tfric # friction term for divergence
       
    dvortdtSpec_srce = x.grid2sph(vortdot)
    ddivdtSpec_srce = x.grid2sph(divdot)
    if(logSE):
        thermalterm = ( qplus - qminus + qns ) / energyg
        denergydtaddterms = -divg / 3. /(1.-beta/2.) + \
                            (0.5*sdotplus*((vg-vd)**2+(ug-ud)**2)  +  energy_source) / energyg
    else:
        thermalterm = ( qplus - qminus + qns )
        denergydtaddterms = -divg * pressg + (0.5*sdotplus*((vg-vd)**2+(ug-ud)**2)  +  energy_source)
    if(tdepl>0.):
        if(logSE):
            denergydtaddterms -= 1./tdepl
        else:
            denergydtaddterms -= energyg/tdepl
    if(ktrunc_diss>0.):
        diss_diff = np.exp(-hyperdiff_expanded * (ktrunc / ktrunc_diss)**ndiss * dt * dtscale)
        denergydtSpec_srce = x.grid2sph( thermalterm ) *diss_diff  + x.grid2sph( denergydtaddterms)  
    else:
        denergydtSpec_srce = x.grid2sph( thermalterm+ denergydtaddterms )

    timer.stop_comp("source-terms")
    ##################################################
    # crash branch
    if(lenergyg.min()<-100.):
        wdtspecnan=np.where(lenergyg<-100.)
        print("l minimal energy "+str(lenergyg.min()))
        print("l maximal energy "+str(lenergyg.max()))
        print("l minimal sigma "+str(lsig.min()))
        print("l maximal sigma "+str(lsig.max()))
        print("beta = "+str(beta[wdtspecnan]))
        print("the dangerous terms (without energy):\n")
        print((-divg * pressg)[wdtspecnan])
        print(((qplus - qminus + qns))[wdtspecnan])
        print((sdotplus*((vg-vd)**2+(ug-ud)**2))[wdtspecnan])
#        print((sdotplus*csqinit_acc* 3. * (1.-beta_acc/2.)-1./tdepl * energyg)[wdtspecnan])
        print("time from last output "+str(t-tstore+dt_out))
        f5.close()
        sys.exit()

    ##################################################

    #    denergyg=x.sph2grid(denergydtSpec)
    if(logSE):
        dt_thermal=1./(np.abs(thermalterm)+np.abs(denergydtaddterms)).max()
        dt_accr=1./(np.abs(sdotplus)).max()
    else:
        dt_thermal=1./((np.abs(thermalterm)+np.abs(denergydtaddterms))/energypos).max()
        dt_accr=1./(np.abs(sdotplus/sig)).max()
    if(ifscaledt):
        dt=0.5/(np.sqrt(np.maximum(1.*cssqmax,3.*vsqmax))/dt_cfl+5./dt_thermal+5./dt_accr+1./dt_out) # dt_accr may safely equal to inf, checked
        # dt=1./(1./dt_cfl+1./dt_thermal+2./dt_accr+1./dt_out)
    else:
        dt=dt_cfl
    if(dt <= 1e-10):
        print(" dt(CFL, sound) = "+str(dt_cfl/np.sqrt(cssqmax)))
        print(" dt(CFL, adv) = "+str(dt_cfl/np.sqrt(vsqmax)))
        print(" dt(thermal) = "+str(dt_thermal))
        print(" dt(accr) = "+str(dt_accr))
        print("dt = "+str(dt))
        print("time from last output "+str(t-tstore+dt_out))
        f5.close()
        sys.exit()

    ##################################################
    # passive scalar evolution:
    timer.start_comp("passive-scalar")
    #    tmpg1 = ug*accflag; tmpg2 = vg*accflag
    tmpSpec, dacctmp = x.getVortDivSpec(ug*accflag,vg*accflag)
    daccflagdtSpec = -dacctmp
    #    daccflagdt =  (1.-accflag) * sdotplus/sig + accflag * divg 
    daccflagdtSpec += x.grid2sph(accflag * divg) # da/dt = - div(av) + a div(v)
    daccflagdtSpec_srce = x.grid2sph((1.-accflag) * sdotplus/sig)
    
    timer.stop_comp("passive-scalar")
    ##################################################
    # at last, the time step
    timer.start_comp("time-step")

    t += dt ; ncycle+=1
    vortSpec += (dvortdtSpec)  * dt
    divSpec += (ddivdtSpec)  * dt
    sigSpec += (dsigdtSpec)  * dt
    energySpec += (denergydtSpec) * dt
    accflagSpec += (daccflagdtSpec) * dt

    timer.stop_comp("time-step")
    ##################################################
    #diffusion
    timer.start_comp("diffusion2")

    if(ifscalediff):
        hyperdiff_fact = np.exp(-hyperdiff_expanded*dt)
        div_diff =  np.exp(-ddivfac*hyperdiff_expanded*dt)# divergence factor enhanced. 
        sigma_diff = hyperdiff_fact

    vortSpec *= hyperdiff_fact
    divSpec *= div_diff
    sigSpec *= sigma_diff
    energySpec *= sigma_diff
    accflagSpec *= sigma_diff # do we need to smooth it?

    # adding source terms: 
    vortSpec += dvortdtSpec_srce * dt
    divSpec += ddivdtSpec_srce * dt
    sigSpec += dsigdtSpec_srce * dt
    energySpec += denergydtSpec_srce * dt
    accflagSpec +=  daccflagdtSpec_srce * dt

    if(jitterskip>0):
        if(ncycle % jitterskip == 0):
            #        vortg0 = vortg ; divg0 = divg ; sig0 = sig ; pressg0=pressg
            dphi = np.random.rand()*np.pi-np.pi/2. # /np.double(x.nlons)
            dlon = np.random.rand()*2.*np.pi-np.pi
            #        dphi = np.pi / 4.
            #        print("jitter by "+str(dphi))
            # jitternod(vort, div, sig, energy, incl, grid, grid1)
            vortg1, divg1, sig1, energyg1, accflag1 = jitterturn(x.sph2grid(vortSpec), x.sph2grid(divSpec), x.sph2grid(sigSpec), x.sph2grid(energySpec), x.sph2grid(accflagSpec), dlon, x, x1)
            vortg2, divg2, sig2, energyg2, accflag2 = jitternod(vortg1, divg1, sig1, energyg1, accflag1, -dphi, x1, x1)
            vortg1, divg1, sig1, energyg1, accflag1 = jitternod(vortg2, divg2, sig2, energyg2, accflag2,  dphi, x1, x1)
            vortg2, divg2, sig2, energyg2, accflag2 = jitterturn(vortg1, divg1, sig1, energyg1, accflag1,  -dlon, x1, x)
            vortSpec = x.grid2sph(vortg2)  ; divSpec = x.grid2sph(divg2) ; sigSpec = x.grid2sph(sig2)  ; energySpec =  x.grid2sph(energyg2) ; accflagSpec = x.grid2sph(accflag2)
            #        print("jitter by "+str(dphi))
            #        r = input("j")

    timer.stop_comp("diffusion2")
    ##################################################
    timer.lap("step") 
    
    ##################################################

    if(ncycle % outskip ==0 ): # make sure it's alive
        print("lg(E) range "+str(lenergyg.min())+" to "+str(lenergyg.max()))
        print("lg(Sigma) range "+str(lsig.min())+" to "+str(lsig.max()))
        print('t=%10.5f ms' % (t*1e3*tscale))
        print("ncycle = "+str(ncycle))
        print("csqmax = "+str(cssqmax)+", vsqmax = "+str(vsqmax))
        print(" dt(CFL, light) = "+str(dt_cfl))
        print(" dt(CFL, sound) = "+str(dt_cfl/np.sqrt(cssqmax)))
        print(" dt(CFL, adv) = "+str(dt_cfl/np.sqrt(vsqmax)))
        print(" dt(thermal) = "+str(dt_thermal))
        print(" dt(accr) = "+str(dt_accr))
        print("dt = "+str(dt))
        time2 = time.clock()
        print('simulation time = '+str(time2-time1)+'s')
        print("about "+str(t/tmax*100.)+"% done")
        print("(delta * dx / cs**2)_max = "+str(divmachsq.max()))
        
    ##################################################
    # I/O

    #plot & save
    if ( t >=  tstore):

        timer.start("io")

        tstore=t+dt_out
        if(ifplot):
            visualize(t, nout,
                      lats, lons, 
                      vortg, divg, ug, vg, sig, pressg, beta, accflag, qminus, qplus+qns, 
                      conf, f5io.outdir)
        else:
            print(" \n  writing data point number "+str(nout)+"\n\n")
        #file I/O
        f5io.saveSim(f5, nout, t,
                     vortg, divg, ug, vg, sig, energyg, beta,
                     accflag, dissipation, qminus, qplus, lsdot*sig,
                     conf)
        nout += 1
        sys.stdout.flush()
        
        timer.stop("io")

        #print simulation run-time statistics
        timer.stats("step")
        timer.stats("io")
        timer.comp_stats()

        timer.start("step") #refresh lap counter (avoids IO profiling)
        timer.purge_comps()

#end of time cycle loop
f5.close()
time2 = time.clock()
print(('CPU time = ',time2-time1))

# ffmpeg -f image2 -r 35 -pattern_type glob -i 'out/swater*.png' -pix_fmt yuv420p -b 4096k out/swater.mp4

timer.stop("total")
timer.stats("total")
