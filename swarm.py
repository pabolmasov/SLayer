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
from sympy.solvers import solve
# from sympy import Symbol
import scipy.interpolate as si
import imp
import sys

#Code modules:
from spharmt import Spharmt 
import f5io as f5io #module for file I/O
import conf as conf #importing simulation setup module

####################################
# extra arguments (allow to run several slayers in parallel and write results to arbitrary outputs)
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
##################################################
# setup code environment
# f5io.outdir = 'out'
if not os.path.exists(f5io.outdir):
    os.makedirs(f5io.outdir)

f5 = h5py.File(f5io.outdir+'/run.hdf5', "w")

#import simulation parameters to global scope
from conf import nlons, nlats
from conf import grav, rsphere
from conf import dt_cfl, omega, rsphere, sig0, sigfloor, overkepler, tscale, dtout
from conf import bump_amp, bump_lat0, bump_lon0, bump_dlon, bump_dlat  #initial perturbation parameters
from conf import efold, ndiss, efold_diss
from conf import csqmin, csqinit, cssqscale, kappa, mu, betamin # EOS parameters
from conf import itmax, outskip, tmax
from conf import ifplot
from conf import sigplus, sigmax, latspread #source and sink terms
from conf import incle, slon0
from conf import ifrestart, nrest, restartfile
from conf import tfric
from conf import ifscalediffusion
from conf import iftwist, twistscale

if(ifplot):
    from plots import visualize

############################
# beta calibration
bmin=betamin ; bmax=1.-betamin ; nb=1000
# hard limits for stability; bmin\sim 1e-7 approximately corresponds to Coulomb coupling G\sim 1,
# hence there is even certain physical meaning in bmin
b=(bmax-bmin)*(old_div((np.arange(nb)+0.5),np.double(nb)))+bmin
bx=old_div(b,(1.-b)**0.25)
b[0]=0. ; bx[0]=0.  # ; b[nb-1]=1e3 ; bx[nb-1]=1.
betasolve_p=si.interp1d(bx, b, kind='linear', bounds_error=False, fill_value=1.)
#(bmin,bmax)) # as a function of pressure
betasolve_e=si.interp1d(bx/(1.-old_div(b,2.))/3., b, kind='linear', bounds_error=False,fill_value=1.)
#fill_value=(bmin,bmax)) # as a function of energy
# for k in np.arange(nb):
#    print str(bx[k])+" -> "+str(b[k])+"\n"
# rr=raw_input("d")
######################################

##################################################
# setup up spherical harmonic instance, set lats/lons of grid
x = Spharmt(conf.nlons, conf.nlats, conf.ntrunc, conf.rsphere, gridtype='gaussian') # rsphere is known by x!! Is Gaussian grid what we need?
lons,lats = np.meshgrid(x.lons, x.lats)

# guide grids for plotting
lons1d = (old_div(180.,np.pi))*x.lons-180.
lats1d = (old_div(180.,np.pi))*x.lats

############
# time step
dt=dt_cfl # if we are in trouble, dt=1./(1./dtcfl + 1./dtthermal)

#######################################################
## initial conditions: ###
# initial velocity field 
ug = 2.*omega*np.cos(lats)*rsphere
vg = ug*0.
if(iftwist):
    ug *= (lats/twistscale) / np.sqrt(1.+(lats/twistscale)**2)

# density perturbation
# hbump = bump_amp*np.cos(lats)*np.exp(-((lons-bump_lon0)/bump_alpha)**2)*np.exp(-((bump_phi0-lats)/bump_beta)**2)
hbump = bump_amp*np.exp(-(old_div((lons-bump_lon0),bump_dlon))**2)*np.exp(-(old_div((lats-bump_lat0),bump_dlat))**2)

# initial vorticity, divergence in spectral space
vortSpec, divSpec =  x.getVortDivSpec(ug,vg) 
vortg = x.sph2grid(vortSpec)
vortgNS = x.sph2grid(vortSpec) # rotation of the neutron star 
divg  = x.sph2grid(divSpec)

# create (hyper)diffusion factor; normal diffusion corresponds to ndiss=4
hyperdiff_fact = np.exp((old_div(-dt,efold))*(old_div(x.lap,np.abs(x.lap).max()))**(old_div(ndiss,2)))
sigma_diff = hyperdiff_fact
hyperdiff_expanded = old_div((old_div(x.lap,np.abs(x.lap).max()))**(old_div(ndiss,2)), efold)
diss_diff = np.exp((old_div(-dt,efold_diss))*(old_div(x.lap,np.abs(x.lap).max()))**(old_div(ndiss,2))) # -- additional diffusion factor applied to energy dissipation (as there is high-frequency noise in dissipation function that we do not want to be introduced again through dissipation function)
# supposedly a stationary solution:
sig=sig0*np.exp(0.5*(omega*rsphere*np.cos(lats))**2/csqinit)
# *(np.cos(lats))**((omega*rsphere)**2/csqinit)+sigfloor
# print "initial sigma: "+str(sig.min())+" to "+str(sig.max())
# ii=raw_input("")
# in pressure, there should not be any strong bumps, as we do not want artificial shock waves
pressg = sig * csqinit / (1. + hbump)
geff=-grav+old_div((ug**2+vg**2),rsphere) # effective gravity
sigpos=old_div((sig+np.fabs(sig)),2.) # we need to exclude negative sigma points from calculation
beta = betasolve_p(cssqscale*sig/pressg*np.sqrt(np.sqrt(-geff*sigpos))) # beta as a function of sigma, press, geff
energyg = pressg / 3. / (1.-old_div(beta,2.))
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
    y=sigplus*np.exp(old_div(-(old_div(devcos,latspread))**2,2.))
    #        y/=(2.*np.pi)**1.5*latspread
    return y, devcos

def sdotsink(sigma, sigmax):
    '''
    sink term in surface density
    sdotsink(sigma, sigmax)
    '''
    w=np.where(sigma>(old_div(sigmax,100.)))
    y=0.0*sigma
    if(np.size(w)>0):
        y[w]=1.0*sigma[w]*np.exp(old_div(-sigmax,sigma[w]))
    return y

# main loop
time1 = time.clock() # time loop

nout=nrest ;  t=t0 ; tstore=t0 # starting counters
ncycle=0

while(t<(tmax+t0)):
# ncycle in np.arange(itmax+1):
    # get vort,u,v,sigma on grid
    vortg = x.sph2grid(vortSpec)
    sig  = x.sph2grid(sigSpec)
    accflag = x.sph2grid(accflagSpec)
    divg  = x.sph2grid(divSpec)
    energyg  = x.sph2grid(energySpec)
    ug,vg = x.getuv(vortSpec,divSpec) # velocity components
    geff=-grav+old_div((ug**2+vg**2),rsphere) # effective gravity
    geff=old_div((geff-np.fabs(geff)),2.) # only negative geff
    sigpos=old_div((sig+np.fabs(sig)),2.) # we need to exclude negative sigma points from calculation
    # there could be bias if many sigma<0 points appear
    beta = betasolve_e(cssqscale*sig/energyg*np.sqrt(np.sqrt(-geff*sigpos))) # beta as a function of sigma, energy, and geff
    wbnan=np.where(np.isnan(beta))
    if(np.size(wbnan)>0):
        print("beta = "+str(beta[wbnan]))
        print("geff = "+str(geff[wbnan]))
        print("sig = "+str(sig[wbnan]))
        print("energy = "+str(energyg[wbnan]))
        ii=input("betanan")
    pressg=energyg / 3. / (1.-old_div(beta,2.))
    cssqmax = (pressg/sig).max() # estimate for maximal speed of sound
    vsqmax = (ug**2+vg**2).max()
    # vorticity flux
    tmpg1 = ug*vortg ;    tmpg2 = vg*vortg
    ddivdtSpec, dvortdtSpec = x.getVortDivSpec(tmpg1 ,tmpg2 ) # all the nablas already contain an additional 1/R multiplier
    dvortdtSpec *= -1
    tmpg = x.sph2grid(ddivdtSpec)
    tmpg1 = ug*sig; tmpg2 = vg*sig
    tmpSpec, dsigdtSpec = x.getVortDivSpec(tmpg1 ,tmpg2 ) # all the nablas should contain an additional 1/R multiplier
    dsigdtSpec *= -1
    # energy (pressure) flux:
    tmpg1 = ug*energyg; tmpg2 = vg*energyg
    tmpSpec, denergydtSpec = x.getVortDivSpec(tmpg1, tmpg2) # all the nablas should contain an additional 1/R multiplier
    #    wunbound=np.where(geff>=0.) # extreme case; we can be unbound due to pressure
    denergydtSpec *= -1
    denergyg_adv = x.sph2grid(denergydtSpec) # for debugging
    # dissipation estimates:
    dissvortSpec=vortSpec*(hyperdiff_expanded+old_div(1.,tfric)) #expanded exponential diffusion term
    dissdivSpec=divSpec*(hyperdiff_expanded+old_div(1.,tfric)) # need to incorporate for omegaNS in the friction term
    wnan=np.where(np.isnan(dissvortSpec+dissdivSpec))
    if(np.size(wnan)>0):
        dissvortSpec[wnan]=0. ;  dissdivSpec[wnan]=0.
    dissug, dissvg = x.getuv(dissvortSpec, dissdivSpec)
    dissipation=(ug*dissug+vg*dissvg) # -v . dv/dt_diss
    #    dissipation = (dissipation + np.fabs(dissipation))/2. # only positive!
    dissipation = x.sph2grid(x.grid2sph(dissipation)*diss_diff) # smoothing dissipation 
    #    if(np.size(wunbound)>0):
#        geff[wunbound]=0.
#        print "ug from "+str(ug.min())+" to "+str(ug.max())
#        print "vg from "+str(vg.min())+" to "+str(vg.max())
#        rr=raw_input(".")
    kenergy=0.5*(ug**2+vg**2) # kinetic energy per unit mass (merge this with baroclinic term?)
    tmpSpec = x.grid2sph(kenergy)
    ddivdtSpec += -x.lap*tmpSpec

    # baroclinic terms in vorticity and divirgence:
    gradp1, gradp2 = x.getGrad(x.grid2sph(pressg))  # ; grads1, grads2 = x.getGrad(sigSpec)
    vortpressbarSpec, divpressbarSpec = x.getVortDivSpec(old_div(gradp1,sig),old_div(gradp2,sig)) # each nabla already has its rsphere
    ddivdtSpec += -divpressbarSpec 
    dvortdtSpec += -vortpressbarSpec

    # energy sources and sinks:   
    qplus = sigpos * dissipation 
    qminus = (-geff) * sigpos / 3. / (1.+kappa*sigpos) * (1.-beta) 
    qns = (old_div(csqmin,cssqscale))**4  # conversion of (minimal) speed of sound to flux
        
    # source terms in mass:
    sdotplus, sina=sdotsource(lats, lons, latspread)
    sdotminus=sdotsink(sig, sigmax)
    sdotSpec=x.grid2sph(sdotplus-sdotminus)
    dsigdtSpec += sdotSpec

    # source term in vorticity
    vortdot=sdotplus/sig*(2.*overkepler/rsphere**1.5*sina-vortg)+old_div((vortgNS-vortg),tfric) # +sdotminus/sig*vortg
    divdot=old_div(-divg,tfric) # friction term for divergence
    vortdotSpec=x.grid2sph(vortdot)
    divdotSpec=x.grid2sph(divdot)
    dvortdtSpec += vortdotSpec
    ddivdtSpec += divdotSpec
    denergydtSpec += x.grid2sph(-divg * pressg+(qplus - qminus + qns)+(sdotplus*csqmin-pressg/sig*sdotminus) * 3. * (1.-old_div(beta,2.)))
    denergyg = x.sph2grid(denergydtSpec) # maybe we can optimize this?
    denergyg1= denergyg_adv-divg*pressg + (qplus - qminus + qns)+(sdotplus*csqmin-pressg/sig*sdotminus) * 3. * (1.-old_div(beta,2.))
    energyslip=np.abs(denergyg-denergyg1).max()
    if(energyslip>1.):
        dq= qplus - qminus + qns
        dsrc=(sdotplus*csqmin-pressg/sig*sdotminus) * 3. * (1.-old_div(beta,2.))
        print("dE = "+str(energyslip))
        print("correctness of inverse transform (with hyperdiff)"+str(np.abs(denergydtSpec*hyperdiff_fact-x.grid2sph(x.sph2grid(denergydtSpec))).max()))
        print("correctness of inverse transform "+str(np.abs(denergydtSpec-x.grid2sph(x.sph2grid(denergydtSpec))).max()))
        print("correctness of inverse transform "+str(np.abs(denergydtSpec-x.grid2sph(x.sph2grid(denergydtSpec))).max()))
        print("correctness of inverse transform "+str(np.abs(denergyg_adv-x.sph2grid(x.grid2sph(denergyg_adv))).max()))
        print("accuracy (grid) "+str(np.abs(x.sph2grid(x.grid2sph(denergyg_adv-divg * pressg+dq+dsrc))-(denergyg_adv-divg * pressg+dq+dsrc)).max()))
        print("mean "+str((x.sph2grid(denergydtSpec)-(denergyg_adv-divg * pressg+dq+dsrc)).mean()))
        print("std of nabla(vE) "+str(denergyg_adv.std()))
        print("std of delta * Pi "+str((divg * pressg).std()))
        print("accuracy (sph) "+str(np.abs(denergydtSpec-x.grid2sph(denergyg_adv-divg * pressg+dq+dsrc)).max()))
        print("accuracy (sph->grid) "+str(np.abs(x.sph2grid(denergydtSpec-x.grid2sph(denergyg_adv-divg * pressg+dq+dsrc))).max()))
        print("accuracy (grid->sph) "+str(np.abs(x.grid2sph(x.sph2grid(denergydtSpec)-(denergyg_adv-divg * pressg+dq+dsrc))).max()))
        print("accuracy (grid, rms) "+str((x.sph2grid(denergydtSpec)-(denergyg_adv-divg * pressg+dq+dsrc)).std()))
        rr=input('//')
    #    dt_thermal=1./(np.fabs(denergyg)/(energyg+dt_cfl*np.fabs(denergyg))).max()
    dt_thermal=old_div(0.5,(old_div(np.fabs(denergyg),energyg)).max())
    wtrouble=(old_div(np.fabs(denergyg),energyg)).argmax()
    if(dt_thermal <= 1e-10):
        dsrc=(sdotplus*csqmin-pressg/sig*sdotminus) * 3. * (1.-old_div(beta,2.))
        print("E = "+str(energyg.flatten()[wtrouble]))
        print("sig = "+str(sig.flatten()[wtrouble]))
        print("beta = "+str(beta.flatten()[wtrouble]))
        print("divg = "+str(divg.flatten()[wtrouble]))
        print("dE = "+str(denergyg.flatten()[wtrouble]))
        print("Q+ +QNS= "+str((qplus+qns).flatten()[wtrouble]))
        print("Q- = "+str(qminus.flatten()[wtrouble]))
        print("nabla(vE) = "+str((denergyg_adv).flatten()[wtrouble]))
        print("delta * P = "+str((divg * pressg).flatten()[wtrouble]))
        print("delta(dE) = "+str(np.fabs(denergyg-denergyg_adv-qplus-qns+qminus+divg * pressg-dsrc).flatten()[wtrouble]))
        print("dissipation = "+str((dissipation).flatten()[wtrouble]))
        print("source term = "+str(dsrc.flatten()[wtrouble]))
        print("nearby points: ")
        print("E = "+str(energyg.flatten()[wtrouble-3:wtrouble+3]))
        print("Q+ = "+str(qplus.flatten()[wtrouble-3:wtrouble+3]))
        rr=input()
    #    if( dt_thermal <= (10. * dt) ): # very rapid thermal evolution; we can artificially decrease the time step
    dt=old_div(0.5,2.*np.sqrt(np.minimum(cssqmax,vsqmax))/dt_cfl+1./dt_thermal+1./dtout)
#    else:
#        dt=dt_cfl # maybe we can try increasing dt_cfl = dt0 /sqrt(csqmax)?
    # passive scalar evolution:
    tmpg1 = ug*accflag; tmpg2 = vg*accflag
    tmpSpec, dacctmp = x.getVortDivSpec(tmpg1,tmpg2)
    daccflagdtSpec = -dacctmp # a*div(v) - div(a*v)
    daccflagdt =  (1.-accflag) * (old_div(sdotplus,sig)) + accflag * divg 
    daccflagdtSpec += x.grid2sph(daccflagdt)
    
    # at last, the time step
    t += dt ; ncycle+=1
    vortSpec += dvortdtSpec * dt

    divSpec += ddivdtSpec * dt

    sigSpec += dsigdtSpec * dt

    energySpec += denergydtSpec * dt

    accflagSpec += daccflagdtSpec * dt

    # implicit hyperdiffusion for vort and div
    if(ifscalediffusion):
        vortSpec *= hyperdiff_fact**(old_div(dt,dt_cfl))
        divSpec *= hyperdiff_fact**(old_div(dt,dt_cfl))
        sigSpec *= sigma_diff**(old_div(dt,dt_cfl))
        energySpec *= sigma_diff**(old_div(dt,dt_cfl))
    else:
        vortSpec *= hyperdiff_fact
        divSpec *= hyperdiff_fact
        sigSpec *= sigma_diff
        energySpec *= sigma_diff
      
    #    accflagSpec *= sigma_diff 

    #    if(ncycle % np.floor(outskip/10) ==0):
    #        print('t=%10.5f ms' % (t*1e3*tscale))
    if(ncycle % (old_div(outskip,10)) ==0 ): # make sure it's alive
        print('t=%10.5f ms' % (t*1e3*tscale))
        print(" dt(CFL, sound) = "+str(dt_cfl*np.sqrt(cssqmax)))
        print(" dt(CFL, adv) = "+str(dt_cfl*np.sqrt(vsqmax)))
        print(" dt(thermal) = "+str(dt_thermal))
        print("dt = "+str(dt))
        time2 = time.clock()
        print('simulation time = '+str(time2-time1)+'s')
        print("about "+str(t/tmax*100.)+"% done") 

    #plot & save
    if ( t >  (tstore+dtout)):
        tstore=t
        if(ifplot):
            visualize(t, nout,
                      lats, lons, 
                      vortg, divg, ug, vg, sig, pressg, beta, accflag, dissipation, 
#                      hbump,
#                      rsphere,
                      conf, f5io.outdir)

        #file I/O
        f5io.saveSim(f5, nout, t,
                     vortg, divg, ug, vg, sig, energyg, beta,
                     accflag, dissipation,
                     conf)
        nout += 1

        
#end of time cycle loop

time2 = time.clock()
print(('CPU time = ',time2-time1))

# ffmpeg -f image2 -r 15 -pattern_type glob -i 'out/swater*.png' -pix_fmt yuv420p -b 4096k out/swater.mp4
