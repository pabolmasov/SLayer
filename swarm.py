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
from conf import omega, sig0, overkepler, tscale, eqrot # star rotation frequency, initial density level, deviation from Kepler for the falling matter
from conf import ifscaledt, dt_cfl_factor, dt_out_factor # scaling for the time steps
from conf import ifscalediff
from conf import bump_amp, bump_lat0, bump_lon0, bump_dlon, bump_dlat  #initial perturbation parameters
from conf import ktrunc, ndiss, ktrunc_diss # e-folding time scale for the hyper-diffusion, order of hyper-diffusion, e-folding time for dissipation smoothing
from conf import ddivfac, jitterskip, decosine # numerical tricks to suppress certain numerical instabilities
from conf import csqmin, csqinit, cssqscale, kappa, mu, betamin, sigmafloor, energyfloor # physical parameters 
# from conf import isothermal, gammainit, kinit # initial EOS
from conf import outskip, tmax # frequency of diagnostic outputs, maximal time
from conf import ifplot # if we make plots
from conf import sigplus, latspread, incle, slon0, tturnon # source term
from conf import ifrestart, nrest, restartfile # restart setup
# interaction with the NS surface: 
from conf import tfric, tdepl, satsink # friction and depletion times, saturation sink flag
from conf import iftwist, twistscale # twist test parameters
from conf import eps_deformation # deformation of the NS surface 
# turning off certain physical effects:
from conf import nocool, noheat, fixedEOS, gammaEOS

if(ifplot):
    from plots import visualize

from jitter import jitternod, jitterturn
    
############################
# beta calibration
bmin=betamin ; bmax=1.-betamin ; nb=10000
# hard limits for stability; bmin\sim 1e-7 approximately corresponds to Coulomb coupling G\sim 1,
# hence there is even certain physical meaning in bmin
# "beta" part of the main loop is little-time-consuming independently of nb
b = (bmax-bmin)*((np.arange(nb)+0.5)/np.double(nb))+bmin
bx = b/(1.-b)**0.25
b[0]=0. ; bx[0]=0.0  # ; b[nb-1]=1e3 ; bx[nb-1]=1.
betasolve_p=si.interp1d(bx, b, kind='linear', bounds_error=False, fill_value=1.)
# as a function of pressure
betasolve_e=si.interp1d(bx/(1.-b/2.)/3., b, kind='linear', bounds_error=False,fill_value=1.)
# as a function of energy
######################################

#############################################################
# fixed EOS:
def pEOS(sig):
    return sig0 * csqinit * (sig/sig0)**gammaEOS

def eEOS(sig):
    return sig0 * csqinit * (sig/sig0)**gammaEOS / (gammaEOS-1.)

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

##################################################
# setup up spherical harmonic instance, set lats/lons of grid
x = Spharmt(conf.nlons, conf.nlats, conf.ntrunc, conf.rsphere, gridtype='gaussian')
x1 = Spharmt(4*conf.nlons, 4*conf.nlats, 4*conf.ntrunc, conf.rsphere, gridtype='gaussian') # needed for jitter
lons,lats = np.meshgrid(x.lons, x.lats)
############
# time step estimate
dx=np.fabs((lons[1:-1,1:]-lons[1:-1,:-1])*np.cos(lats[1:-1,:-1])).min()/2. * rsphere
dy=np.fabs(x.lats[1:]-x.lats[:-1]).min()/2. * rsphere
dt_cfl = dt_cfl_factor / (1./dx + 1./dy) # basic CFL limit for light velocity
print("dt(CFL) = "+str(dt_cfl)+"GM/c**3 = "+str(dt_cfl*tscale)+"s")
dt=dt_cfl 
dt_out=dt_out_factor*rsphere**(1.5)/np.sqrt(mass1) # time step for output (we need to resolve the dynamic time scale)
print("dt_out = "+str(dt_out)+"GM/c**3 = "+str(dt_out*tscale)+"s")
# sources:
sdotmax, sina = sdotsource(lats, lons, latspread) # surface density source and sine of the distance towards the rotation axis of the falling matter (normally, slightly offset to the rotation of the star)
vort_source = 2.*overkepler/rsphere**1.5 * sina
# * np.exp(-(sina/latspread)**2)+vortgNS*(1.-np.exp(-(sina/latspread)**2))
ud,vd = x.getuv(x.grid2sph(vort_source),x.grid2sph(vort_source)*0.) # velocity components of the source
# beta_acc = 1. # gas-dominated matter
beta_acc = 0. # radiation-dominated matter
csqinit_acc = csqinit # (overkepler*latspread)**2 / rsphere
energy_source_max = sdotmax*csqinit_acc* 3. * (1.-beta_acc/2.)

#######################################################
## initial conditions: ###
# initial velocity field  (pure rotation)
ug = omega*np.cos(lats)*rsphere
if(eqrot):
    ug = ( overkepler / rsphere**0.5 * np.exp(-(sina/latspread)**2)+omega*(1.-np.exp(-(sina/latspread)**2))) * np.cos(lats)
vg = 0.00*omega*np.sin(lats)*np.cos(lons)
if(iftwist):
    ug *= (lats/twistscale) / np.sqrt(1.+(lats/twistscale)**2) # twisted sphere test
ug0 = omega*np.cos(lats)*rsphere

# initial vorticity, divergence in spectral space
vortSpec, divSpec = x.getVortDivSpec(ug,vg) 
vortg = x.sph2grid(vortSpec)
# vortg += 2.*omega*rsphere*np.sin(lats)*np.cos(lons)*0.01
vortgNS = vortg # rotation of the neutron star 
vortSpecNS = vortSpec # rotation of the neutron star, spectral space
divg  = x.sph2grid(divSpec)

# create (hyper)diffusion factor; normal diffusion corresponds to ndiss=2 (x.lap is already nabla^2)
# print(x.lap)
lapmin=np.abs(x.lap[np.abs(x.lap.real)>0.]).min()
lapmax=np.abs(x.lap[np.abs(x.lap.real)>0.]).max()
# hyperdiff_expanded = (-x.lap/(lapmax*ktrunc**2))**(ndiss/2) # positive! let us care somehow about the mean flow
poslap = (abs(x.lap)>=0.)
hyperdiff_expanded = -np.minimum(((x.lap-x.lap[poslap].max())/(lapmax*ktrunc**2))**(ndiss/2), 0.)
if(decosine):
    hyperdiff_expanded = np.abs(x.lap * (x.lap + 1./rsphere**2) / (lapmax*ktrunc**2))**(2.)
# hyperdiff_expanded = hyperdiff_expanded - hyperdiff_expanded[hyperdiff_expanded>0.].min()
# hyperdiff_expanded[0] = 0. # care for the overall rotation trend 
hyperdiff_fact = np.exp(-hyperdiff_expanded*dt) # dt will change in the main loop
# print(x.lap)
# input('fdslkjsa')
div_diff =  np.exp(-ddivfac*hyperdiff_expanded*dt)# divergence factor enhanced
sigma_diff = hyperdiff_fact # sigma and energy are also artificially smoothed
if(ktrunc_diss>0.):
    diss_diff = np.exp(-hyperdiff_expanded * (ktrunc / ktrunc_diss)**(ndiss) * dt)

qns = (csqmin/cssqscale)**4  # conversion of (minimal) speed of sound to flux

# initialize spectral tendency arrays
ddivdtSpec  = np.zeros(vortSpec.shape, np.complex)
dvortdtSpec = np.zeros(vortSpec.shape, np.complex)
dsigdtSpec  = np.zeros(vortSpec.shape, np.complex)
denergydtSpec  = np.zeros(vortSpec.shape, np.complex)
daccflagdtSpec = np.zeros(vortSpec.shape, np.complex)

###########################################################
# restart and more IC:
if(ifrestart):
# restart module:
    vortg, divg, sig, energyg, accflag, t0 = f5io.restart(restartfile, nrest, conf)
else:
    t0=0.
    nrest=0
    # supposedly a stationary solution:
    sig=np.cos(lats)*0.+sig0
    pressg=sig*csqinit # +0.5*(omega*rsphere*np.cos(lats))**2) # check this formula!
    # density perturbation
    hbump = bump_amp*np.exp(-((lons-bump_lon0)/bump_dlon)**2/2.)*np.exp(-((lats-bump_lat0)/bump_dlat)**2/2.)
    sig*=hbump+1. # entropy disturbance 
    if(fixedEOS):
        pressg *= (hbump+1.)**gammaEOS
    print("sigma = "+str(sig.min())+" to "+str(sig.max()))
    print("press = "+str(pressg.min())+" to "+str(pressg.max()))
    #    sig=x.sph2grid(x.grid2sph(sig))
    #    print("sigma = "+str(sig.min())+" to "+str(sig.max()))
    #    ii=input('s')
    # in pressure, there should not be any strong bumps, as we do not want artificial shock waves
    geff=-grav+(ug**2+vg**2-ug0**2)/rsphere
    sigpos=(sig+np.fabs(sig))/2. # we need to exclude negative sigma points from calculation (should there be any? just in case!)
    beta = betasolve_p(cssqscale*sig/pressg*np.sqrt(np.sqrt(-geff*sigpos))) # beta as a function of sigma, press, geff
    energyg = pressg * 3. * (1.-beta/2.)
    energy_init = energyg
    sig_init = sig
    accflag = hbump*0. # initially, the tracer is 0 everywhere

# spectral arrays...
# logSE option does not respect conservation laws, I would not advise using it
if(logSE):
    sigSpec  = x.grid2sph(np.log(sig)) # we use logarithms of Sigma and E as both quantities always have the same sign but change by several orders of magnitude
    energySpec  = x.grid2sph(np.log(energyg))
else:
    sigSpec  = x.grid2sph(sig)
    energySpec  = x.grid2sph(energyg)
accflagSpec  = x.grid2sph(accflag)
divSpec  = x.grid2sph(divg)
vortSpec = x.grid2sph(vortg)
#############################################################################################

###################################################
# Save simulation setup to file
f5io.saveParams(f5, conf)

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
        sig  = x.sph2grid(sigSpec)
        sigpos = (sig+sigmafloor + np.abs(sig-sigmafloor))/2. # applying the effective floor to sig
        #        sig+=sigmafloor # making sure there are no negative points
        lsig = np.log(sigpos) # do we use lsig/lenergyg?
        if(fixedEOS):
            energyg = eEOS(sig) # we replace all the thermal transfer by a fixed EOS
            energypos = energyg
        else:
            energyg  = x.sph2grid(energySpec)
            energypos = (energyg+energyfloor + np.abs(energyg-energyfloor))/2.
        #        energyg+=energyfloor # making sure there are no negative points
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
    if(fixedEOS):
        pressg = pEOS(sig)
    else:
        pressg = energyg / 3. / (1.-beta/2.) # beta is not the source of all evil
    cssqmax = (pressg/sig).max() # estimate for maximal speed of sound
    vsqmax = (ug**2+vg**2).max()

    # shock watch !!!
    divmachsq = divg**2 * (dx**2 + dy**2) / (pressg/sig) # Mach^2 for divergence ; divmachsq \gtrsim 1 means a shock wave
#    if(divmachsq.max()>100.):
#        print("divmachsqmax = "+str(divmachsq.max()))
#        print("estimated machsq = "+str((omega*rsphere)**2/csqinit)+" for the whole star,\n")
#        print("... and "+str((omega*rsphere/np.double(x.nlats))**2/csqinit))
#        input("mach")
    #        divg *= divg / np.sqrt(divmachsq + 1.)
    #    divSpec = x.grid2sph(divg/ np.sqrt(divmachsq + 1.)) # could be optimized
    
    timer.stop_comp("beta")
    ##################################################
    timer.start_comp("fluxes")

    # vorticity flux
    #    tmpg1 = ug*vortg ;    tmpg2 = vg*vortg
    ddivdtSpec, dvortdtSpec = x.getVortDivSpec( ug*vortg, vg*vortg ) # all the nablas already contain an additional 1/R multiplier
    dvortdtSpec *= -1.
    # divergence flux
    ddivdtSpec += - x.lap * x.grid2sph(ug**2+vg**2+ug0**2) / 2. 
    #    tmpg = x.sph2grid(ddivdtSpec)
    #    tmpg1 = ug*lsig;  tmpg2 = vg*lsig
    if(logSE):
        tmpSpec, dsigdtSpec = x.getVortDivSpec(ug*lsig, vg*lsig ) 
        tmpSpec, denergydtSpec = x.getVortDivSpec(ug*lenergyg, vg*lenergyg) 
    else:
        tmpSpec, dsigdtSpec = x.getVortDivSpec(ug*sig, vg*sig)
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
    if(noheat):
        dissvortSpec=vortSpec*0. ; dissdivSpec=divSpec*0.
    else:
        if(tfric>0.):
            dissvortSpec=(vortSpec-vortSpecNS)*(hyperdiff_expanded*dtscale+1./tfric) #expanded exponential diffusion term; NS rotation is subtracted to exclude negative dissipation
            dissdivSpec=divSpec*(ddivfac*hyperdiff_expanded*dtscale+1./tfric) # need to incorporate for omegaNS in the friction term
        else:
            dissvortSpec=(vortSpec-vortSpecNS)*hyperdiff_expanded*dtscale #expanded exponential diffusion term; NS rotation is subtracted to exclude negative dissipation
            dissdivSpec=divSpec*hyperdiff_expanded*dtscale # need to incorporate for omegaNS in the friction term
    dissug, dissvg = x.getuv(dissvortSpec, dissdivSpec)
    dissipation = ((ug-ug0)*dissug+vg*dissvg) # -v . dv/dt_diss  # positive if it is real dissipation, because hyperdiff_expanded is positive
    # note that the motion of the NS is subtracted to avoid negative dissipation of the basic flow

    # energy sources and sinks:   
    qplus = sig * dissipation
    if(nocool):
        qminus = geff*0.
    else:
        qminus = (-geff/kappa) * (1.-beta) # vertical integration excludes rho or sigma; no 3 here (see section "vertical structure" in the paper)
    timer.stop_comp("diffusion")
    ##################################################
    # baroclinic terms in vorticity and divirgence:
    timer.start_comp("baroclinic")

    gradp1, gradp2 = x.getGrad(x.grid2sph(pressg))  
    vortpressbarSpec, divpressbarSpec = x.getVortDivSpec(gradp1/sig, gradp2/sig) # each nabla already has its rsphere
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
        sdotminus = sig/tdepl
    else:
        sdotminus = 0.
    
    if(logSE):
        lsdot = (sdotplus-sdotminus)/sig
        dsigdtSpec_srce = x.grid2sph(lsdot)
        sdot = sig * lsdot
    else:
        sdot = sdotplus-sdotminus
        dsigdtSpec_srce = x.grid2sph(sdot)
    # source term in vorticity
    #    domega=(vort_source-vortg) # difference in net vorticity

    gradsdot1, gradsdot2 = x.getGrad(x.grid2sph(sdotplus/sig))

    vortdot =  sdotplus/sig * (vort_source -vortg) \
               + (vd-vg) * gradsdot1 - (ud-ug) * gradsdot2
    divdot  =  -sdotplus/sig * divg \
               + (ud-ug) * gradsdot1 + (vd-vg) * gradsdot2
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
        dt_accr=1./(np.abs(sdotplus)+np.abs(sdotminus)).max()
    else:
        dt_thermal=1./((np.abs(thermalterm)+np.abs(denergydtaddterms))/energypos).max()
        dt_accr=1./((np.abs(sdotplus)+np.abs(sdotminus))/sig).max()
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
        print(" min(E)="+str(energypos.min()))
        print(" min(Sigma)="+str(sigpos.min()))
        #        print("tdepl = "+str(tdepl))
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
        print("(delta * dx / cs)_max = "+str(np.sqrt(divmachsq.max())))
        
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
                     accflag, dissipation, qminus, qplus, sdot,
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
