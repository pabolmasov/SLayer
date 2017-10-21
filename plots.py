# module for all the visualization tools & functions

import numpy as np
import scipy.ndimage as spin
import matplotlib.pyplot as plt


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



def visualizeSprofile(ax, latsDeg, data, title="", log=False):
    # latitudal profile
    ax.cla()
    ax.plot(latsDeg, data, ',k')
    ax.set_xlabel('latitude, deg')
    ax.set_ylabel(title)
    if(log):
        ax.set_yscale('log')

def visualizeTwoprofiles(ax, lonsDeg, latsDeg, data1, data2, title1="", title2="", log=False):
    # latitudal profile
    ax.cla()
    ax.plot(latsDeg, data1, ',k', label=title1)
    ax.plot(latsDeg, data2, ',r', label=title2)

    ax.set_xlabel('latitude, deg')
    ax.set_ylabel(title1+', '+title2)
    if(log):
        ax.set_yscale('log')

def visualizeMap(ax, lonsDeg, latsDeg, data, vmin=0.0, vmax=1.0, title=""):

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


def visualizeMapVecs(ax, lonsDeg, latsDeg, xx, yy, title=""):

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


def visualize(t, nout,
              lats, lons, 
              vortg, divg, ug, vg, sig, dissipation,
              mass, energy,
              engy,
              hbump,
              cf):

    lonsDeg = (180./np.pi)*lons-180.
    latsDeg = (180./np.pi)*lats


    vorm=np.fabs(vortg-2.*cf.omega*np.sin(lats)).max()

    print "vorticity: "+str(vortg.min())+" to "+str(vortg.max())
    print "divergence: "+str(divg.min())+" to "+str(divg.max())
    print "azimuthal U: "+str(ug.min())+" to "+str(ug.max())
    print "polar V: "+str(vg.min())+" to "+str(vg.max())
    print "Sigma: "+str(sig.min())+" to "+str(sig.max())
    print "maximal dissipation "+str(dissipation.max())
    print "minimal dissipation "+str(dissipation.min())
    print "total mass = "+str(mass)
    print "total energy = "+str(energy)
    print "net energy = "+str(energy/mass)

    dismax=(dissipation*sig).max()

    #vorticity
    visualizeMap(axs[0], 
                 lonsDeg, latsDeg, 
                 vortg-2.*cf.omega*np.sin(lats), 
                 -vorm*1.1, vorm*1.1, 
                 title="Vorticity")


    #
    visualizeTwoprofiles(axs[1], 
                        lonsDeg, latsDeg, 
                        vortg, 
                        2.*cf.omega*np.sin(lats), 
                        title1=r"$v_\varphi$", 
                        title2=r"$R\Omega$")


    #divergence
    divm=np.fabs(divg).max()
    visualizeMap(axs[2], 
                 lonsDeg, latsDeg, 
                 divg,  
                 -1.1*divm, 1.1*divm, 
                 title="Divergence")


    visualizeSprofile(axs[3], 
                      latsDeg, 
                      divg, 
                      title=r"$(\nabla \cdot v)$")


    # sigma
    sig_init_base = cf.sig0*np.exp(-(cf.omega*cf.rsphere/cf.cs)**2/2.*(1.-np.cos(lats)))
    sig_init = cf.sig0*(np.exp(-(cf.omega*cf.rsphere/cf.cs)**2/2.*(1.-np.cos(lats))) + hbump) # exact solution + perturbation

    visualizeMap(axs[4], 
                 lonsDeg, latsDeg, 
                 np.log(sig/sig_init_base),  
                 np.log((sig/sig_init_base).min()*0.9),  np.log((sig/sig_init_base).max()*1.1),  
                 title=r'$\Sigma$')

    visualizeTwoprofiles(axs[5], 
                         lonsDeg, latsDeg, 
                         sig/sig_init_base, 
                         sig_init/sig_init_base, 
                         title1="$\Sigma$", 
                         title2="$\Sigma_0$",
                         log=True)

    #dissipation
    visualizeMap(axs[6], 
                 lonsDeg, latsDeg, 
                 np.log(np.fabs(dissipation*sig)), 
                 np.log(dismax*1.e-5).max(), np.log(dismax*1.5).max(),  
                 title=r'Dissipation')


    visualizeSprofile(axs[7], 
                      latsDeg,
                      dissipation*sig,  
                      title=r'Dissipation', 
                      log=True)


    #velocities
    du=ug-cf.omega*cf.rsphere*np.cos(lats)
    dv=vg
    vabs=du**2+dv**2+cf.cs**2 
    dunorm=du/vabs
    dvnorm=dv/vabs

    visualizeMapVecs(axs[8], 
                     lonsDeg, latsDeg, 
                     ug-cf.omega*cf.rsphere*np.cos(lats), 
                     vg, 
                     title="Velocities")


    #velocity distributions
    visualizeTwoprofiles(axs[9], 
                         lonsDeg, latsDeg, 
                         ug, vg, 
                         title1=r"$v_\varphi$", 
                         title2=r"$v_\theta$" )
    axs[9].plot(latsDeg, cf.omega*cf.rsphere*np.cos(lats), color='b', linewidth=1)



    axs[0].set_title('{:6.2f} ms'.format( t*cf.tscale*1e3) )
    scycle = str(nout).rjust(6, '0')
    plt.savefig('out/swater'+scycle+'.png' ) #, bbox_inches='tight') 


