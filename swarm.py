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

f5 = h5py.File(f5io.outdir+'/run.hdf5', "w")

#import simulation parameters to global scope
from conf import nlons, nlats
from conf import grav, rsphere, mass1
from conf import omega, rsphere, sig0, sigfloor, overkepler, tscale
from conf import bump_amp, bump_lat0, bump_lon0, bump_dlon, bump_dlat  #initial perturbation parameters
from conf import efold, ndiss, efold_diss
from conf import csqmin, csqinit, cssqscale, kappa, mu, betamin # EOS parameters
from conf import isothermal, gammainit, kinit
from conf import outskip, tmax
from conf import ifplot
from conf import sigplus, sigmax, latspread #source and sink terms
from conf import incle, slon0
from conf import ifrestart, nrest, restartfile
from conf import tfric
from conf import iftwist, twistscale
# from conf import sigmascale

if(ifplot):
    from plots import visualize

############################
# beta calibration
bmin=betamin ; bmax=1.-betamin ; nb=1000
# hard limits for stability; bmin\sim 1e-7 approximately corresponds to Coulomb coupling G\sim 1,
# hence there is even certain physical meaning in bmin
b = (bmax-bmin)*(old_div((np.arange(nb)+0.5),np.double(nb)))+bmin
bx = b/(1.-b)**0.25
b[0]=0. ; bx[0]=0.  # ; b[nb-1]=1e3 ; bx[nb-1]=1.
betasolve_p=si.interp1d(bx, b, kind='linear', bounds_error=False, fill_value=1.)
# as a function of pressure
betasolve_e=si.interp1d(bx/(1.-b/2.)/3., b, kind='linear', bounds_error=False,fill_value=1.)
# as a function of energy
######################################

##################################################
# setup up spherical harmonic instance, set lats/lons of grid
x = Spharmt(conf.nlons, conf.nlats, conf.ntrunc, conf.rsphere, gridtype='gaussian')
# rsphere is known by x!! Is Gaussian grid what we need?
lons,lats = np.meshgrid(x.lons, x.lats)
############
# time steps
dx=np.fabs((lons[1:-1,1:]-lons[1:-1,:-1])*np.cos(lats[1:-1,:-1])).min()/2. * rsphere
dy=np.fabs(x.lats[1:]-x.lats[:-1]).min()/2. * rsphere
dt_cfl = 0.2 / (1./dx + 1./dy) # basic CFL limit for light velocity
print("dt(CFL) = "+str(dt_cfl)+"GM/c**3 = "+str(dt_cfl*tscale)+"s")
dt=dt_cfl 
dtout=0.25*rsphere**(1.5)/np.sqrt(mass1) # time step for output (we need to resolve the dynamic time scale)
print("dtout = "+str(dtout)+"GM/c**3 = "+str(dtout*tscale)+"s")
#######################################################
## initial conditions: ###
# initial velocity field  (pure rotation)
ug = omega*np.cos(lats)*rsphere
vg = ug*0.
if(iftwist):
    ug *= (lats/twistscale) / np.sqrt(1.+(lats/twistscale)**2)
# density perturbation
hbump = bump_amp*np.exp(-((lons-bump_lon0)/bump_dlon)**2/2.)*np.exp(-((lats-bump_lat0)/bump_dlat)**2/2.)

# initial vorticity, divergence in spectral space
vortSpec, divSpec = x.getVortDivSpec(ug,vg) 
vortg = x.sph2grid(vortSpec)
vortgNS = x.sph2grid(vortSpec) # rotation of the neutron star 
divg  = x.sph2grid(divSpec)

# create (hyper)diffusion factor; normal diffusion corresponds to ndiss=2 (x.lap is already nabla^2)
hyperdiff_expanded = ((x.lap/np.abs(x.lap).max()))**(ndiss/2)/efold
hyperdiff_fact = np.exp(-hyperdiff_expanded*dt) # if efold scales with dt, this should work
sigma_diff = hyperdiff_fact # sigma and energy are also artificially smoothed
diss_diff = np.exp(-(x.lap/np.abs(x.lap).max())**(ndiss/2)/efold_diss) # dissipation is artifitially smoother by a larger amount

# initialize spectral tendency arrays
ddivdtSpec  = np.zeros(vortSpec.shape+(3,), np.complex)
dvortdtSpec = np.zeros(vortSpec.shape+(3,), np.complex)
dsigdtSpec  = np.zeros(vortSpec.shape+(3,), np.complex)
# dpressdtSpec  = np.zeros(vortSpec.shape+(3,), np.complex)
denergydtSpec  = np.zeros(vortSpec.shape+(3,), np.complex)
daccflagdtSpec = np.zeros(vortSpec.shape+(3,), np.complex)

# Cycling integers for integrator
nnew = 0
nnow = 1
nold = 2

###########################################################
# restart module:
if(ifrestart):
    vortg, divg, sig, energyg, accflag, t0 = f5io.restart(restartfile, nrest, conf)
else:
    t0=0.
    nrest=0
    # supposedly a stationary solution:
    if(isothermal):
        sig=sig0*np.exp(0.5*(omega*rsphere*np.cos(lats))**2/csqinit)
        pressg = sig * csqinit / (1. + hbump)
    else:
        sig=(sig0**(gammainit-1.)+0.5*(gammainit-1.)/gammainit*(omega*rsphere*np.cos(lats))**2/kinit)**(1./(gammainit-1.))
        pressg=kinit*sig**gammainit
    sig*=hbump+1.
    # in pressure, there should not be any strong bumps, as we do not want artificial shock waves
    geff=-grav+old_div((ug**2+vg**2),rsphere)
    # sigpos=old_div((sig+np.fabs(sig)),2.) # we need to exclude negative sigma points from calculation
    beta = betasolve_p(cssqscale*sig/pressg*np.sqrt(np.sqrt(-geff*sig))) # beta as a function of sigma, press, geff
    energyg = pressg * 3. * (1.-beta/2.)
    accflag=hbump*0.
    
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
    y=sigplus*np.exp(-0.5*(devcos/latspread)**2)
    return y, devcos

def sdotsink(sigma, sigmax):
    '''
    sink term in surface density
    sdotsink(sigma, sigmax)
    '''
    w=np.where(sigma>(sigmax/100.))
    y=0.0*sigma
    tff=1.
    if((sigmax>0.)&(np.size(w)>0)):
        y[w]=sigma[w]/tff*np.exp(-sigmax/sigma[w])
    return y

# source velocities:
sdotplus, sina = sdotsource(lats, lons, latspread)
omega_source=2.*overkepler/rsphere**1.5*sina
ud,vd = x.getuv(x.grid2sph(omega_source),x.grid2sph(omega_source)*0.) # velocity components of the source

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
    geff=-grav+(ug**2+vg**2)/rsphere # effective gravity
    geff=(geff-np.fabs(geff))/2. # only negative geff
    #    sigpos=(sig+np.fabs(sig))/2. # we need to exclude negative sigma points from calculation
    # there could be bias if many sigma<0 points appear
    if((sig.min()<0.)|(energyg.min()<0.)):
        print("sigmin = "+str(sig.min()))
        print("energymin = "+str(energyg.min()))
        wpressmin=energyg.argmin()
        print("beta[] = "+str(np.reshape(beta,np.size(beta))[wpressmin]))
        wneg=np.where((sig<0.) | (energyg<0.))
        nneg=np.size(wneg)
        print(str(nneg)+" negative points")
        f5.close()
        sys.exit()
        # sig=(sig+np.fabs(sig))/2.+sigfloor ; energyg=(energyg+np.fabs(energyg))/2.+sigfloor*csqmin
    beta = betasolve_e(cssqscale*sig/energyg*np.sqrt(np.sqrt(-geff*sig))) # beta as a function of sigma, energy, and geff
    wbnan=np.where(np.isnan(beta))
    if(np.size(wbnan)>0):
        print("beta = "+str(beta[wbnan]))
        print("geff = "+str(geff[wbnan]))
        print("sig = "+str(sig[wbnan]))
        print("energy = "+str(energyg[wbnan]))
        # ii=input("betanan")
        print(str(np.size(wbnan))+" beta=NaN points")
        f5.close()
        sys.exit()
    pressg=energyg / 3. / (1.-beta/2.)
    cssqmax = (pressg/sig).max() # estimate for maximal speed of sound
    vsqmax = (ug**2+vg**2).max()
    # vorticity flux
    tmpg1 = ug*vortg ;    tmpg2 = vg*vortg
    ddivdtSpec[:,nnew], dvortdtSpec[:,nnew] = x.getVortDivSpec(tmpg1, tmpg2 ) # all the nablas already contain an additional 1/R multiplier
    dvortdtSpec[:,nnew] *= -1
    #    tmpg = x.sph2grid(ddivdtSpec)
    tmpg1 = ug*sig; tmpg2 = vg*sig
    tmpSpec, dsigdtSpec[:,nnew] = x.getVortDivSpec(tmpg1, tmpg2 ) # all the nablas should contain an additional 1/R multiplier
    dsigdtSpec *= -1
    # energy (pressure) flux:
    tmpg1 = ug*energyg; tmpg2 = vg*energyg
    tmpSpec, denergydtSpec[:,nnew] = x.getVortDivSpec(tmpg1, tmpg2) # all the nablas should contain an additional 1/R multiplier
    #    wunbound=np.where(geff>=0.) # extreme case; we can be unbound due to pressure
    denergydtSpec[:,nnew] *= -1
    denergydtSpec0 = denergydtSpec[:,nnew]
    denergyg_adv = x.sph2grid(denergydtSpec[:,nnew]) # for debugging
    # dissipation estimates:
    if(tfric>0.):
        dissvortSpec=vortSpec*(hyperdiff_expanded+old_div(1.,tfric)) #expanded exponential diffusion term
        dissdivSpec=divSpec*(hyperdiff_expanded+old_div(1.,tfric)) # need to incorporate for omegaNS in the friction term
    else:
        dissvortSpec=vortSpec*hyperdiff_expanded #expanded exponential diffusion term
        dissdivSpec=divSpec*hyperdiff_expanded # need to incorporate for omegaNS in the friction term
        
    wnan=np.where(np.isnan(dissvortSpec+dissdivSpec))
    if(np.size(wnan)>0):
        dissvortSpec[wnan]=0. ;  dissdivSpec[wnan]=0.
    dissug, dissvg = x.getuv(dissvortSpec, dissdivSpec)
    dissipation=(ug*dissug+vg*dissvg)  # -v . dv/dt_diss 
    # dissipation = (dissipation + np.fabs(dissipation))/2. # only positive!
    if(efold_diss>0.):
        dissipation = x.sph2grid(x.grid2sph(dissipation)*diss_diff) # smoothing dissipation 
    kenergy=0.5*(ug**2+vg**2) # kinetic energy per unit mass (merge this with baroclinic term?)
    tmpSpec = x.grid2sph(kenergy) # * hyperdiff_fact
    ddivdtSpec[:,nnew] += -tmpSpec * x.lap

    # baroclinic terms in vorticity and divirgence:
    gradp1, gradp2 = x.getGrad(x.grid2sph(pressg))  # ; grads1, grads2 = x.getGrad(sigSpec)
    #  * hyperdiff_fact )
    vortpressbarSpec, divpressbarSpec = x.getVortDivSpec(gradp1/sig,gradp2/sig) # each nabla already has its rsphere
    ddivdtSpec[:,nnew] += -divpressbarSpec 
    dvortdtSpec[:,nnew] += -vortpressbarSpec

    # energy sources and sinks:   
    qplus = sig * dissipation 
    qminus = (-geff) * sig / 3. / (1.+kappa*sig) * (1.-beta) 
    qns = (csqmin/cssqscale)**4  # conversion of (minimal) speed of sound to flux
        
    # source terms in mass:
    #     sdotplus, sina=sdotsource(lats, lons, latspread) # sufficient to calculate once!
    sdotminus=sdotsink(sig, sigmax)
    sdotSpec=x.grid2sph(sdotplus-sdotminus)
    dsigdtSpec[:,nnew] += sdotSpec

    # source term in vorticity
    domega=(omega_source-vortg) # difference in net vorticity
    #    print("sina from "+str(sina.min())+" to "+str(sina.max()))
    #    print("Dvort = "+str((sdotplus*domega).min())+".."+str((sdotplus*domega).max()))
    #    print("vort = "+str((sdotplus*domega).min())+".."+str((sdotplus*domega).max()))
    vortdot=sdotplus/sig*domega
    divdot=-sdotplus/sig*divg
    if(tfric>0.):
        vortdot+=(vortgNS-vortg)/tfric # +sdotminus/sig*vortg
        divdot+=-divg/tfric # friction term for divergence
       
    vortdotSpec=x.grid2sph(vortdot)
    divdotSpec=x.grid2sph(divdot)
    dvortdtSpec[:,nnew] += vortdotSpec
    ddivdtSpec[:,nnew] += divdotSpec
    csqinit_acc = (overkepler*latspread)**2/rsphere # initial distribution of accreted matter suggests certain temperature
    beta_acc=1.0
    denergydtSpec[:,nnew] += x.grid2sph(-divg * pressg + (qplus - qminus + qns)
                                +0.5*sdotplus*((vg-vd)**2+(ug-ud)**2) # initial dissipation
                                #                                +(sdotplus-sdotminus)/sig*energyg)
                                +(sdotplus*csqinit_acc* 3. * (1.-beta_acc/2.)-energyg*sdotminus/sig) )
    # there are two additional source terms for E:
    # 1) accreting matter is hot
    # 2) half of the energy goes to heat when accretion spins up the material of the SL

    # passive scalar evolution:
    tmpg1 = ug*accflag; tmpg2 = vg*accflag
    tmpSpec, dacctmp = x.getVortDivSpec(tmpg1,tmpg2)
    daccflagdtSpec[:,nnew] = -dacctmp # a*div(v) - div(a*v)
    daccflagdt =  (1.-accflag) * sdotplus/sig + accflag * divg 
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
        denergydtSpec[:,nnow] = denergydtSpec[:,nnew]
        denergydtSpec[:,nold] = denergydtSpec[:,nnew]
        daccflagdtSpec[:,nnow] = daccflagdtSpec[:,nnew]
        daccflagdtSpec[:,nold] = daccflagdtSpec[:,nnew]
    elif ncycle == 1:
        dvortdtSpec[:,nold] = dvortdtSpec[:,nnew]
        ddivdtSpec[:,nold] = ddivdtSpec[:,nnew]
        dsigdtSpec[:,nold] = dsigdtSpec[:,nnew]
        denergydtSpec[:,nold] = denergydtSpec[:,nnew]
        daccflagdtSpec[:,nold] = daccflagdtSpec[:,nnew]
        # regular step with a 3rd order Adams-Bashforth
    vortSpec += dt*( \
    (23./12.)*dvortdtSpec[:,nnew] - (16./12.)*dvortdtSpec[:,nnow]+ \
    (5./12.)*dvortdtSpec[:,nold] )

    divSpec += dt*( \
    (23./12.)*ddivdtSpec[:,nnew] - (16./12.)*ddivdtSpec[:,nnow]+ \
    (5./12.)*ddivdtSpec[:,nold] )

    sigSpec += dt*( \
    (23./12.)*dsigdtSpec[:,nnew] - (16./12.)*dsigdtSpec[:,nnow]+ \
    (5./12.)*dsigdtSpec[:,nold] )

    energySpec += dt*( \
                       (23./12.)*denergydtSpec[:,nnew] - (16./12.)*denergydtSpec[:,nnow]+ \
                       (5./12.)*denergydtSpec[:,nold] )

    accflagSpec += dt*( \
    (23./12.)*daccflagdtSpec[:,nnew] - (16./12.)*daccflagdtSpec[:,nnow]+ \
    (5./12.)*daccflagdtSpec[:,nold] )

    # switch indices, do next time step
    nsav1 = nnew; nsav2 = nnow
    nnew = nold; nnow = nsav1; nold = nsav2

    # time step advancement
    t += dt ; ncycle+=1

    # at last, the time step
    '''    
    t += dt ; ncycle+=1
    vortSpec += dvortdtSpec * dt
    divSpec += ddivdtSpec * dt
    sigSpec += dsigdtSpec * dt
    energySpec += denergydtSpec * dt
    accflagSpec += daccflagdtSpec * dt
    '''
    #    hyperdiff_fact = np.exp(-hyperdiff_expanded*dt)
    #    sigma_diff = hyperdiff_fact
    #    diss_diff = np.exp(-hyperdiff_expanded * efold / efold_diss * dt)

    vortSpec *= hyperdiff_fact
    divSpec *= hyperdiff_fact
    sigSpec *= sigma_diff # test: what about not smoothing sigma?
    energySpec *= sigma_diff
    
    if(ncycle % (old_div(outskip,10)) ==0 ): # make sure it's alive
        print('t=%10.5f ms' % (t*1e3*tscale))
#        print(" dt(CFL, sound) = "+str(dt_cfl/np.sqrt(cssqmax)))
#        print(" dt(CFL, adv) = "+str(dt_cfl/np.sqrt(vsqmax)))
#        print(" dt(thermal) = "+str(dt_thermal))
#        print(" dt(accr) = "+str(dt_accr))
#        print("dt = "+str(dt))
        time2 = time.clock()
        print('simulation time = '+str(time2-time1)+'s')
        print("about "+str(t/tmax*100.)+"% done") 

    #plot & save
    if ( t >=  tstore):
        tstore=t+dtout
        if(ifplot):
            visualize(t, nout,
                      lats, lons, 
                      vortg, divg, ug, vg, sig, pressg, beta, accflag, dissipation, 
                      conf, f5io.outdir)
        else:
            print(" \n  writing data point number "+str(nout)+"\n\n")
        #file I/O
        f5io.saveSim(f5, nout, t,
                     vortg, divg, ug, vg, sig, energyg, beta,
                     accflag, dissipation,
                     conf)
        nout += 1
        
#end of time cycle loop
f5.close()
time2 = time.clock()
print(('CPU time = ',time2-time1))

# ffmpeg -f image2 -r 15 -pattern_type glob -i 'out/swater*.png' -pix_fmt yuv420p -b 4096k out/swater.mp4
# ls -ntr out/*png 
