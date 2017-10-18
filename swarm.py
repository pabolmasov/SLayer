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
import shtns
import matplotlib.pyplot as plt
import time
from spharmt import Spharmt 
import os
import h5py


##################################################
#prepare figure etc
fig = plt.figure(figsize=(10,10))
gs = plt.GridSpec(5, 10)
gs.update(hspace = 0.2)
gs.update(wspace = 0.6)

axs = []
axs.append( plt.subplot(gs[0, 0:5]) )
axs.append( plt.subplot(gs[0, 6:10]) )
axs.append( plt.subplot(gs[1, 0:5]) )
axs.append( plt.subplot(gs[1, 6:10]) )
axs.append( plt.subplot(gs[2, 0:5]) )
axs.append( plt.subplot(gs[2, 6:10]) )
axs.append( plt.subplot(gs[3, 0:5]) )
axs.append( plt.subplot(gs[3, 6:10]) )
axs.append( plt.subplot(gs[4, 0:5]) )
axs.append( plt.subplot(gs[4, 6:10]) )

directory = 'out/'
if not os.path.exists(directory):
    os.makedirs(directory)


##################################################
# grid, time step info
nlons = 256              # number of longitudes
ntrunc = int(nlons/3)    # spectral truncation (to make it alias-free)
nlats = int(nlons/2)     # for gaussian grid
tscale=6.89631e-06 # time units are GM/c**3 \simeq 
dt = 1.e-8                 # time step in seconds
dt/=tscale
print "dt = "+str(dt)+"GM/c**3 = "+str(dt*tscale)+"s"
# rr=raw_input("?")
itmax = 10000000 # number of iterations

# parameters for test
rsphere = 6.04606 # earth radius
pspin = 1e-2 # spin period, in seconds
omega = 2.*np.pi/pspin/1.45e5    # rotation rate
overkepler=0.9
grav = 1./rsphere**2     # gravity

phi0 = np.pi/3.
lon0=np.pi/3.
# phi1 = 0.5*np.pi - phi0
# phi2 = 0.05*np.pi
# en = np.exp(-4.0/(phi1-phi0)**2)
alpha = 1./3.
beta = 1./15.
hamp=0.5
sigfloor = 0.1
sig0 = 10.       # own neutron star atmosphere

efold = 10000.*dt    # efolding timescale at ntrunc for hyperdiffusion
ndiss = 8           # order for hyperdiffusion

# setup up spherical harmonic instance, set lats/lons of grid
x = Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
lons,lats = np.meshgrid(x.lons, x.lats)
# f = 2.*omega*np.sin(lats) # Coriolis # no additional Coriolis term needed

# guide grids for plotting
lons1d = (180./np.pi)*x.lons-180.
lats1d = (180./np.pi)*x.lats

#lonsDeg = (180./np.pi)*x.lons-180.
#latsDeg = (180./np.pi)*x.lats
lonsDeg = (180./np.pi)*lons-180.
latsDeg = (180./np.pi)*lats

# zonal jet
vg = np.zeros((nlats,nlons), np.float)
ug = np.ones((nlats,nlons), np.float)*np.cos(lats)*omega*rsphere

# height perturbation.
hbump = hamp*np.cos(lats)*np.exp(-((lons-lon0)/alpha)**2)*np.exp(-(phi0-lats)**2/beta)

# initial vorticity, divergence in spectral space
vortSpec, divSpec =  x.getVortDivSpec(ug,vg)
vortg = x.sph2grid(vortSpec)
divg  = x.sph2grid(divSpec)

# source term:
sigmax=1.e8
latspread=0.2 # spread in radians
# 1e3*np.ones((nlats,nlons), np.float)*np.exp(-(np.sin(lats)/latspread)**2/.2)-exp(sig/sigmax)
# 
cs=0.01 # speed of sound
csq=cs**2

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
sig_init = sig0*(np.exp(-(omega*rsphere/cs)**2/2.*(1.-np.cos(lats))) + hbump) # exact solution + perturbation
sigSpec = x.grid2sph(sig)
# sdotSpec = x.grid2sph(sdot)

# initialize spectral tendency arrays
ddivdtSpec  = np.zeros(vortSpec.shape+(3,), np.complex)
dvortdtSpec = np.zeros(vortSpec.shape+(3,), np.complex)
dsigdtSpec  = np.zeros(vortSpec.shape+(3,), np.complex)
nnew = 0
nnow = 1
nold = 2


###################################################
# Save simulation setup to file
f5 = h5py.File("out/run.hdf5", "w")
grp0 = f5.create_group("params")

grp0.attrs['nlons']      = nlons
grp0.attrs['ntrunc']     = ntrunc
grp0.attrs['nlats']      = nlats
grp0.attrs['tscale']     = tscale
grp0.attrs['dt']         = dt
grp0.attrs['itmax']      = itmax
grp0.attrs['rsphere']    = rsphere
grp0.attrs['pspin']      = pspin
grp0.attrs['omega']      = omega
grp0.attrs['overkepler'] = overkepler
grp0.attrs['grav']       = grav
grp0.attrs['sig0']       = sig0
grp0.attrs['cs']         = cs



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
    ax.quiver(
        lonsDeg[::sk, ::sk],
        latsDeg[::sk, ::sk],
        xx[::sk, ::sk]*100./vvmax, yy[::sk, ::sk]*100./vvmax,
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
    return 0.*np.ones((nlats,nlons), np.float)*np.exp(-(np.sin(lats)/latspread)**2/.2)

def sdotsink(sigma, sigmax):
    return 0.1*sigma*np.exp(-sigmax/sigma)



# main loop
time1 = time.clock() # time loop

nout=0

for ncycle in range(itmax+1):
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
    press=cs**2*np.log((sig+np.fabs(sig))/2.+sigfloor) # stabilizing equation of state
    tmpSpec = x.grid2sph(press+0.5*(ug**2+vg**2))
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

    vortdot=sdotplus/sig*(2.*overkepler*np.sin(lats)-vortg)
    vortdotSpec=x.grid2sph(vortdot)
    
    sigSpec += dt*sdotSpec # source term for density
    vortSpec += dt*vortdotSpec # source term for entropy

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
    if (ncycle % 10000 == 0):
        vorm=np.fabs(vortg-2.*omega*np.sin(lats)).max()
        divm=np.fabs(divg).max()
        print "vorticity: "+str(vortg.min())+" to "+str(vortg.max())
        print "divergence: "+str(divg.min())+" to "+str(divg.max())
        print "azimuthal U: "+str(ug.min())+" to "+str(ug.max())
        print "polar V: "+str(vg.min())+" to "+str(vg.max())
        print "Sigma: "+str(sig.min())+" to "+str(sig.max())
        print "maximal dissipation "+str(dissipation.max())
        dismax=(dissipation*sig).max()
        visualizeMap(axs[0], vortg-2.*omega*np.sin(lats), -vorm*1.1, vorm*1.1, title="Vorticity")
        visualizeTwoprofiles(axs[1], vortg, 2.*omega*np.sin(lats), title1=r"$v_\varphi$", title2=r"$R\Omega$")
        visualizeMap(axs[2], divg,  -1.1*divm, 1.1*divm, title="Divergence")
        visualizeSprofile(axs[3], divg, title=r"$(\nabla \cdot v)$")
        visualizeMap(axs[4], np.log(sig),  np.log(sig0*0.9),  np.log(sig.max()*1.1),  title=r'$\Sigma$')
        visualizeTwoprofiles(axs[5], sig, sig_init, title1="$\Sigma$", title2="$\Sigma_0$",log=True)
        visualizeMap(axs[6], np.log(dissipation*sig), np.log(dismax*1.e-5).max(), np.log(dismax*1.5).max(),  title=r'Dissipation')
        visualizeSprofile(axs[7], dissipation*sig,  title=r'Dissipation', log=True)
#        du=ug-omega*rsphere*np.cos(lats) ; dv=vg
#        vabs=du**2+dv**2+cs**2 
#        dunorm=du/vabs  ; dvnorm=dv/vabs ; 
        visualizeMapVecs(axs[8], ug*np.sqrt(rsphere), vg*np.sqrt(rsphere), title="Velocities")
        visualizeTwoprofiles(axs[9], ug, vg, title1=r"$v_\varphi$", title2=r"$v_\theta$", ome=True)
        axs[0].set_title('{:6.2f} ms'.format( t*tscale*1e3) )
        scycle = str(nout).rjust(6, '0')
        plt.savefig(directory+'swater'+scycle+'.png' ) #, bbox_inches='tight') 
        nout+=1


    #file I/O
    if (ncycle % 1000 == 0):
        scycle = str(ncycle).rjust(6, '0')
        grp = f5.create_group("cycle_"+scycle)

        grp.create_dataset("vortg", data=vortg)
        grp.create_dataset("divg",  data=divg)
        grp.create_dataset("ug",    data=ug)
        grp.create_dataset("vg",    data=vg)
        grp.create_dataset("sig",   data=sig)
        grp.create_dataset("diss",  data=dissipation)


        
#end of time cycle loop

time2 = time.clock()
print('CPU time = ',time2-time1)


