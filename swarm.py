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


##################################################
# setup code environment
f5io.outdir = 'out'
if not os.path.exists(f5io.outdir):
    os.makedirs(f5io.outdir)

f5 = h5py.File(f5io.outdir+'/run.hdf5', "w")

from conf import nlons, nlats
from conf import dt, omega, rsphere, sig0, overkepler, tscale
from conf import hamp, phi0, lon0, alpha, beta  #initial height perturbation parameters
from conf import efold, ndiss
from conf import cs
from conf import itmax
from conf import sigfloor
from conf import sigmax, latspread #source term



##################################################
#prepare figure etc
fig = plt.figure(figsize=(10,10))
gs = plt.GridSpec(5, 10)
gs.update(hspace = 0.2)
gs.update(wspace = 0.6)

axs = []
for row in [0,1,2,3,4]:
    axs.append( plt.subplot(gs[row, 0:5]) )
    axs.append( plt.subplot(gs[row, 6:10]) )


##################################################
# setup up spherical harmonic instance, set lats/lons of grid
x = Spharmt(conf.nlons, conf.nlats, conf.ntrunc, conf.rsphere, gridtype='gaussian')
lons,lats = np.meshgrid(x.lons, x.lats)


# guide grids for plotting
lons1d = (180./np.pi)*x.lons-180.
lats1d = (180./np.pi)*x.lats

lonsDeg = (180./np.pi)*lons-180.
latsDeg = (180./np.pi)*lats


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



# solve nonlinear balance eqn to get initial zonal geopotential,
# add localized bump (not balanced).
vortg = x.sph2grid(vortSpec)
tmpg1 = ug*vortg; tmpg2 = vg*vortg
tmpSpec1, tmpSpec2 = x.getVortDivSpec(tmpg1,tmpg2)
tmpSpec2 = x.grid2sph(0.5*(ug**2+vg**2))
sigSpec = x.invlap*tmpSpec1 - tmpSpec2
sig = sig0*(np.exp(-(omega*rsphere/cs)**2/2.*(1.-np.cos(lats))) + hbump) # exact solution + perturbation
sig_init_base = sig0*np.exp(-(omega*rsphere/cs)**2/2.*(1.-np.cos(lats)))
sig_init = sig0*(np.exp(-(omega*rsphere/cs)**2/2.*(1.-np.cos(lats))) + hbump) # exact solution + perturbation
sigSpec = x.grid2sph(sig)
# sdotSpec = x.grid2sph(sdot)

# initialize spectral tendency arrays
ddivdtSpec  = np.zeros(vortSpec.shape+(3,), np.complex)
dvortdtSpec = np.zeros(vortSpec.shape+(3,), np.complex)
dsigdtSpec  = np.zeros(vortSpec.shape+(3,), np.complex)

# Cycling integers for integrator
nnew = 0
nnow = 1
nold = 2



###########################################################
# restart module:
ifrestart=True

if(ifrestart):
    restartfile='out/runOLD.hdf5'

    #nrest=1400 # No of the restart output
    nrest=1 # No of the restart output
    vortg, digg, sig = f5io.restart(restartfile, nrest, conf)

    sigSpec  = x.grid2sph(sig)
    divSpec  = x.grid2sph(divg)
    vortSpec = x.grid2sph(vortg)

else:
    nrest=0
        

###################################################
# Save simulation setup to file
f5io.saveParams(f5, conf)



def visualizeSprofile(ax, data, title="", log=False):
    # latitudal profile
    ax.cla()
    ax.plot(latsDeg, data, ',k')
    ax.set_xlabel('latitude, deg')
    ax.set_ylabel(title)
    if(log):
        ax.set_yscale('log')

def visualizeTwoprofiles(ax, data1, data2, title1="", title2="",ome=False, log=False):
    # latitudal profile
    ax.cla()
    ax.plot(latsDeg, data1, ',k', label=title1)
    ax.plot(latsDeg, data2, ',r', label=title2)
    if(ome):
        ax.plot(latsDeg, omega*rsphere*np.cos(lats), color='b', linewidth=1)
#    ax.legend()
    ax.set_xlabel('latitude, deg')
    ax.set_ylabel(title1+', '+title2)
    if(log):
        ax.set_yscale('log')

def visualizeMap(ax, data, vmin=0.0, vmax=1.0, title=""):

    """ 
    make a contour map plot of the incoming data array (in grid)
    """
    ax.cla()

    print title, " min/max:", data.min(), data.max()

    #make fancy 
    ax.minorticks_on()
    ax.set_ylabel(title)

    #ax.set_xlabel('longitude')
    #ax.set_ylabel('latitude')

    ax.set_xticks(np.arange(-180,181,60))
    ax.set_yticks(np.linspace(-90,90,10))

    ax.pcolormesh(
            lonsDeg,
            latsDeg,
            data,
            vmin=vmin,
            vmax=vmax,
            cmap='plasma',
            )

    #ax.axis('equal')


def visualizeMapVecs(ax, xx, yy, title=""):

    """ 
    make a quiver map plot of the incoming vector field (in grid)
    """
    ax.cla()
    ax.minorticks_on()
    ax.set_ylabel(title)
    ax.set_xticks(np.arange(-180,181,60))
    ax.set_yticks(np.linspace(-90,90,10))

    M = np.hypot(xx, yy)

    print title, " min/max vec len: ", M.min(), M.max()

    vv=np.sqrt(xx**2+yy**2)
    vvmax=vv.max() # normalization

    sk = 10
    sigma = [sk/2., sk/2.]
    xx = spin.filters.gaussian_filter(xx, sigma, mode='constant')*100./vvmax
    yy = spin.filters.gaussian_filter(yy, sigma, mode='constant')*100./vvmax
    
    ax.quiver(
        lonsDeg[::sk, ::sk],
        latsDeg[::sk, ::sk],
        xx[::sk, ::sk], yy[::sk, ::sk],
#        M[::sk, ::sk],
        pivot='mid',
        units='x',
        linewidth=1.0,
        color='k',
        scale=8.0,
    )

    #ax.scatter(
    #        lonsDeg[::sk, ::sk],
    #        latsDeg[::sk, ::sk],
    #        color='k',
    #        s=5,
    #          )
    
##################################################
# source/sink term
def sdotsource(lats, lons, latspread):
    return 0.1*np.ones((nlats,nlons), np.float)*np.exp(-(np.sin(lats)/latspread)**2/.2)

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

    sdotplus=sdotsource(lats, lons, latspread)
    sdotminus=sdotsink(sig, sigmax)
    sdotSpec=x.grid2sph(sdotplus-sdotminus)

    vortdot=sdotplus/sig*(2.*overkepler/rsphere**1.5*np.sin(lats)-vortg)
    vortdotSpec=x.grid2sph(vortdot)
    
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
        vorm=np.fabs(vortg-2.*omega*np.sin(lats)).max()
        divm=np.fabs(divg).max()
        print "vorticity: "+str(vortg.min())+" to "+str(vortg.max())
        print "divergence: "+str(divg.min())+" to "+str(divg.max())
        print "azimuthal U: "+str(ug.min())+" to "+str(ug.max())
        print "polar V: "+str(vg.min())+" to "+str(vg.max())
        print "Sigma: "+str(sig.min())+" to "+str(sig.max())
        print "maximal dissipation "+str(dissipation.max())
        print "minimal dissipation "+str(dissipation.min())
        mass=sig.sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
        energy=(sig*engy).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
        print "total mass = "+str(mass)
        print "total energy = "+str(energy)
        print "net energy = "+str(energy/mass)
        dismax=(dissipation*sig).max()
        visualizeMap(axs[0], vortg-2.*omega*np.sin(lats), -vorm*1.1, vorm*1.1, title="Vorticity")
        visualizeTwoprofiles(axs[1], vortg, 2.*omega*np.sin(lats), title1=r"$v_\varphi$", title2=r"$R\Omega$")
        visualizeMap(axs[2], divg,  -1.1*divm, 1.1*divm, title="Divergence")
        visualizeSprofile(axs[3], divg, title=r"$(\nabla \cdot v)$")
        visualizeMap(axs[4], np.log(sig/sig_init_base),  np.log((sig/sig_init_base).min()*0.9),  np.log((sig/sig_init_base).max()*1.1),  title=r'$\Sigma$')
        visualizeTwoprofiles(axs[5], sig/sig_init_base, sig_init/sig_init_base, title1="$\Sigma$", title2="$\Sigma_0$",log=True)
        visualizeMap(axs[6], np.log(np.fabs(dissipation*sig)), np.log(dismax*1.e-5).max(), np.log(dismax*1.5).max(),  title=r'Dissipation')
        visualizeSprofile(axs[7], dissipation*sig,  title=r'Dissipation', log=True)
#        du=ug-omega*rsphere*np.cos(lats) ; dv=vg
#        vabs=du**2+dv**2+cs**2 
#        dunorm=du/vabs  ; dvnorm=dv/vabs ; 
        visualizeMapVecs(axs[8], ug-omega*rsphere*np.cos(lats), vg, title="Velocities")
        visualizeTwoprofiles(axs[9], ug, vg, title1=r"$v_\varphi$", title2=r"$v_\theta$", ome=True)
        axs[0].set_title('{:6.2f} ms'.format( t*tscale*1e3) )
        scycle = str(nout).rjust(6, '0')
        plt.savefig(f5io.outdir+'/swater'+scycle+'.png' ) #, bbox_inches='tight') 
        nout+=1

        #file I/O
        #     if (ncycle % outskip == 0):
        scycle = str(nout).rjust(6, '0')
        grp = f5.create_group("cycle_"+scycle)
        grp.attrs['t']         = t # time
        grp.attrs['mass']         = mass # total mass
        grp.attrs['energy']         = energy # total mechanical energy

        grp.create_dataset("vortg", data=vortg)
        grp.create_dataset("divg",  data=divg)
        grp.create_dataset("ug",    data=ug)
        grp.create_dataset("vg",    data=vg)
        grp.create_dataset("sig",   data=sig)
        grp.create_dataset("diss",  data=dissipation)
        
#end of time cycle loop

time2 = time.clock()
print('CPU time = ',time2-time1)


