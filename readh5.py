import numpy as np
import shtns
import matplotlib.pyplot as plt
import time
from spharmt import Spharmt 
import os
import h5py


fname = 'out/run.hdf5'
f5 = h5py.File(fname,'r')

for name in f5:
    print name


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
# Read simulation parameters
params = f5['params/']
nlons      = params.attrs['nlons']
ntrunc     = params.attrs['ntrunc']
nlats      = params.attrs['nlats']
tscale     = params.attrs['tscale']
dt_cfl         = params.attrs['dt_cfl']
itmax      = params.attrs['itmax']
rsphere    = params.attrs['rsphere']
pspin      = params.attrs['pspin']
omega      = params.attrs['omega']
overkepler = params.attrs['overkepler']
grav       = params.attrs['grav']
cs         = params.attrs['cs']
sig0       = params.attrs['sig0']


# setup up spherical harmonic instance, set lats/lons of grid
x = Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
lons,lats = np.meshgrid(x.lons, x.lats)
lons1d = (180./np.pi)*x.lons-180.
lats1d = (180./np.pi)*x.lats
lonsDeg = (180./np.pi)*lons-180.
latsDeg = (180./np.pi)*lats



# TODO: do we need to save these to0?
hamp=0.5
phi0 = np.pi/3.
lon0=np.pi/3.
alpha = 1./3.
beta = 1./15.
hbump = hamp*np.cos(lats)*np.exp(-((lons-lon0)/alpha)**2)*np.exp(-(phi0-lats)**2/beta)
sig_init = sig0*(np.exp(-(omega*rsphere/cs)**2/2.*(1.-np.cos(lats))) + hbump) # exact solution + perturbation




##################################################
# visualize all cycles
def print_minmax(arr):
    return str(arr.min())+" to "+str(arr.max())


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


def visualizeCycle(dset, ncycle):
    vortg        = np.array( dset['vortg'] )
    divg         = np.array( dset['divg']  ) 
    ug           = np.array( dset['ug']    ) 
    vg           = np.array( dset['vg']    ) 
    sig          = np.array( dset['sig']   ) 
    dissipation  = np.array( dset['diss']  )
    t=dset.attrs['t']
    vorm=np.fabs(vortg-2.*omega*np.sin(lats)).max()
    divm=np.fabs(divg).max()

    #print "vorticity: "          + print_minmax(vortg)
    #print "divergence: "         + print_minmax(div)
    #print "azimuthal U: "        + print_minmax(ug)
    #print "polar V: "            + print_minmax(vg)
    #print "Sigma: "              + print_minmax(sig)
    #print "maximal dissipation " + print_minmax(diss)


    dismax=(dissipation*sig).max()
    visualizeMap(axs[0], vortg-2.*omega*np.sin(lats), -vorm*1.1, vorm*1.1, title="Vorticity")

    visualizeTwoprofiles(axs[1], vortg, 2.*omega*np.sin(lats), title1=r"$v_\varphi$", title2=r"$R\Omega$")

    visualizeMap(axs[2], divg,  -1.1*divm, 1.1*divm, title="Divergence")

    visualizeSprofile(axs[3], divg, title=r"$(\nabla \cdot v)$")

    visualizeMap(axs[4], np.log(sig),  np.log(sig0*0.9),  np.log(sig.max()*1.1),  title=r'$\Sigma$')

    visualizeTwoprofiles(axs[5], sig, sig_init, title1="$\Sigma$", title2="$\Sigma_0$",log=True)

    visualizeMap(axs[6], np.log(dissipation*sig), np.log(dismax*1.e-5).max(), np.log(dismax*1.5).max(),  title=r'Dissipation')

    visualizeSprofile(axs[7], dissipation*sig,  title=r'Dissipation', log=True)

    visualizeMapVecs(axs[8], ug*np.sqrt(rsphere), vg*np.sqrt(rsphere), title="Velocities")

    visualizeTwoprofiles(axs[9], ug, vg, title1=r"$v_\varphi$", title2=r"$v_\theta$", ome=True)

#    t = ncycle*dt
    axs[0].set_title('{:6.2f} ms'.format( t*tscale*1e3) )

    scycle = str(ncycle).rjust(6, '0')
    plt.savefig(directory+'swater'+scycle+'.png' ) #, bbox_inches='tight') 


dset = f5['cycle_001000']
visualizeCycle(dset, 1000)





