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
from conf import incle

##################################################
# setup up spherical harmonic instance, set lats/lons of grid
x = Spharmt(conf.nlons, conf.nlats, conf.ntrunc, conf.rsphere, gridtype='gaussian')
lons,lats = np.meshgrid(x.lons, x.lats)


# guide grids for plotting
lons1d = (180./np.pi)*x.lons-180.
lats1d = (180./np.pi)*x.lats


# initial velocity field 
ug = omega*rsphere*(np.cos(lats)*np.cos(incle)-np.sin(incle)*np.sin(lats)*np.sin(lons))
vg = omega*rsphere*np.sin(incle)*np.cos(lons)

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
vortg *= (1.-hbump*2.) # some initial condition for vorticity (cyclone?)
accflag=hbump*0.
#print accflag.max(axis=1)
#print accflag.max(axis=0)
#ii=raw_input("f")
# initialize spectral tendency arrays
ddivdtSpec  = np.zeros(vortSpec.shape+(3,), np.complex)
dvortdtSpec = np.zeros(vortSpec.shape+(3,), np.complex)
dsigdtSpec  = np.zeros(vortSpec.shape+(3,), np.complex)
daccflagdtSpec = np.zeros(vortSpec.shape+(3,), np.complex)

# Cycling integers for integrator
nnew = 0
nnow = 1
nold = 2

###########################################################
# restart module:
ifrestart=True

if(ifrestart):
    restartfile='out/runOLD.hdf5'
    nrest=13296 # No of the restart output
    #    nrest=5300 # No of the restart output
    vortg, digg, sig, accflag = f5io.restart(restartfile, nrest, conf)

else:
    nrest=0
        
sigSpec  = x.grid2sph(sig)
accflagSpec  = x.grid2sph(accflag)
divSpec  = x.grid2sph(divg)
vortSpec = x.grid2sph(vortg)

###################################################
# Save simulation setup to file
f5io.saveParams(f5, conf)


    
##################################################
# source/sink term
def sdotsource(lats, lons, latspread):
    y=np.zeros((nlats,nlons), np.float)
    w=np.where(np.fabs(np.sin(lats))<(latspread*5.))
    if(np.size(w)>0):
        y[w]=sigplus*np.exp(-(np.sin(lats[w])/latspread)**2/.2)
    return y

def sdotsink(sigma, sigmax):
    w=np.where(sigma>(sigmax/100.))
    y=0.0*sigma
    if(np.size(w)>0):
        y[w]=1.0*sigma[w]*np.exp(-sigmax/sigma[w])
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
    accflag = x.sph2grid(accflagSpec)
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

    # source terms:
    sdotplus=sdotsource(lats, lons, latspread)
    sdotminus=sdotsink(sig, sigmax)
    sdotSpec=x.grid2sph(sdotplus-sdotminus)
    dsigdtSpec[:,nnew] += sdotSpec

    vortdot=sdotplus/sig*(2.*overkepler/rsphere**1.5*np.sin(lats)-vortg)
    vortdotSpec=x.grid2sph(vortdot)
    dvortdtSpec[:,nnew] += vortdotSpec

    # passive scalar evolution:
    '''
    wacc0=np.where(accflag<0.)
    if(np.size(wacc0)>0):
        accflag[wacc0]=0.
    wacc1=np.where(accflag>1.)
    if(np.size(wacc1)>0):
        accflag[wacc1]=1.
    '''
    #    agradu, agradv = x.getGrad(accflagSpec) # unstable?
    #    daccflagdt = - ug * agradu - vg * agradv
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
        daccflagdtSpec[:,nnow] = daccflagdtSpec[:,nnew]
        daccflagdtSpec[:,nold] = daccflagdtSpec[:,nnew]
    elif ncycle == 1:
        dvortdtSpec[:,nold] = dvortdtSpec[:,nnew]
        ddivdtSpec[:,nold] = ddivdtSpec[:,nnew]
        dsigdtSpec[:,nold] = dsigdtSpec[:,nnew]
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

    accflagSpec += dt*( \
    (23./12.)*daccflagdtSpec[:,nnew] - (16./12.)*daccflagdtSpec[:,nnow]+ \
    (5./12.)*daccflagdtSpec[:,nold] )

    # total kinetic energy loss
    #    dissSpec=(vortSpec**2+divSpec**2)*(1.-hyperdiff_fact)/x.lap
    dissvortSpec=vortSpec*(1.-hyperdiff_fact)
    dissdivSpec=divSpec*(1.-hyperdiff_fact)
    wnan=np.where(np.isnan(dissvortSpec+dissdivSpec))
    if(np.size(wnan)>0):
        dissvortSpec[wnan]=0. ;  dissdivSpec[wnan]=0.
    #    dissipation=(x.sph2grid(dissvortSpec)*vortg+x.sph2grid(dissdivSpec)*divg)/dt
    dissug, dissvg = x.getuv(dissvortSpec, dissdivSpec)
    dissipation=(ug*dissug+vg*dissvg)/dt
    # implicit hyperdiffusion for vort and div
    vortSpec *= hyperdiff_fact
    divSpec *= hyperdiff_fact
    sigSpec *= sigma_diff 
#    accflagSpec *= sigma_diff 

    # switch indices, do next time step
    nsav1 = nnew; nsav2 = nnow
    nnew = nold; nnow = nsav1; nold = nsav2

    if(ncycle % 10 ==0):
        print('t=%10.5f ms' % (t*1e3*tscale))

    #plot & save
    if (ncycle % outskip == 0):

        mass=sig.sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
#        mass_acc=(sig*accflag).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
#        mass_native=(sig*(1.-accflag)).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
        energy=(sig*engy).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
        
        visualize(t, nout,
                  lats, lons, 
                  vortg, divg, ug, vg, sig, accflag, dissipation,
#                  mass, energy,
                  engy,
                  hbump,
                  rsphere,
                  conf)

        #file I/O
        f5io.saveSim(f5, nout, t,
                     energy, mass, 
                     vortg, divg, ug, vg, sig, accflag, dissipation
                     )
        nout += 1

        
#end of time cycle loop

time2 = time.clock()
print('CPU time = ',time2-time1)


