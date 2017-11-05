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

# converts two components to an angle
def tanrat(x,y):
    z=0.
    if((x*y)>0.):
        z=np.arctan(y/x)
        if(x<0.):
            z+=np.pi
    elif((x*y)<0.):
        z=np.arctan(y/x)+np.pi
        if(y<0.):
            z+=np.pi
    else:
        if(x==0.):
            if(y<0.):
                return 1.5*np.pi
        else:
            if(x<0.):
                return np.pi
    return z
    
def visualizeSprofile(ax, latsDeg, data, title="", log=False):
    # latitudal profile
    ax.cla()
    ax.plot(latsDeg, data, '.k',markersize=2)
    ax.set_xlabel('latitude, deg')
    ax.set_ylabel(title)
    if(log):
        ax.set_yscale('log')

def visualizeTwoprofiles(ax, lonsDeg, latsDeg, data1, data2, title1="", title2="", log=False):
    # latitudal profile
    ax.cla()
    ax.plot(latsDeg, data2, '.r',markersize=2)
    ax.plot(latsDeg, data1, '.k',markersize=2)
    if(title1 == "$\Sigma$"):
        ax.plot(latsDeg, data1-data2, '.b',markersize=2)

    ax.set_ylim(data2.min(), data1.max())
    ax.set_xlabel('latitude, deg')
    ax.set_ylabel(title1+', '+title2)
    if(log):
        ax.set_ylim(data1.min(), data1.max())
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
    vvmax=vv.std() # normalization

    sk = 10
    sigma = [sk/2., sk/2.]
    xx = spin.filters.gaussian_filter(xx, sigma, mode='constant')*40./vvmax
    yy = spin.filters.gaussian_filter(yy, sigma, mode='constant')*40./vvmax
    
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
              vortg, divg, ug, vg, sig, accflag, dissipation,
#              mass, energy,
              engy,
              hbump,
              rsphere,
              cf):

    lonsDeg = (180./np.pi)*lons-180.
    latsDeg = (180./np.pi)*lats

    nlons=np.size(lons) ; nlats=np.size(lats)
    
    print "visualize: accreted fraction from "+str(accflag.min())+" to "+str(accflag.max())
    
    vorm=np.fabs(vortg-2.*cf.omega*np.sin(lats)).max()

    mass=sig.sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
    mass_acc=(sig*accflag).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
    mass_native=(sig*(1.-accflag)).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
    energy=(sig*engy).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
    angmoz=(sig*ug*np.cos(lats)).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**3
    angmox=(sig*ug*np.sin(lats)*np.cos(lons)).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**3
    angmoy=(sig*ug*np.sin(lats)*np.sin(lons)).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**3
    vangmo=np.sqrt(angmox**2+angmoy**2+angmoz**2) # total angular momentum 

    print "angular momentum "+str(vangmo)+", inclined wrt z by "+str(np.arccos(angmoz/vangmo)*180./np.pi)+"deg"
    print "vorticity: "+str(vortg.min())+" to "+str(vortg.max())
    print "divergence: "+str(divg.min())+" to "+str(divg.max())
    print "azimuthal U: "+str(ug.min())+" to "+str(ug.max())
    print "polar V: "+str(vg.min())+" to "+str(vg.max())
    print "Sigma: "+str(sig.min())+" to "+str(sig.max())
    print "accretion flag: "+str(accflag.min())+" to "+str(accflag.max())
    print "maximal dissipation "+str(dissipation.max())
    print "minimal dissipation "+str(dissipation.min())
    print "total mass = "+str(mass)
    print "accreted mass = "+str(mass_acc)
    print "native mass = "+str(mass_native)
    print "total energy = "+str(energy)
    print "net energy = "+str(energy/mass)

    dismax=(dissipation*sig).max()

    #vorticity
    visualizeMap(axs[0], 
                 lonsDeg, latsDeg, 
                 vortg-2.*cf.omega*np.sin(lats), 
                 -vorm*1.1, vorm*1.1, 
                 title="Vorticity")
    axs[0].plot([tanrat(angmox, angmoy)*180./np.pi], [np.arcsin(angmoz/vangmo)*180./np.pi], 'or')

    #
    visualizeSprofile(axs[1], 
                      latsDeg, 
                      vortg,
                      title=r"$v_\varphi$")
    axs[1].plot(latsDeg, 2.*cf.omega*np.sin(lats), color='r', linewidth=1)
    axs[1].plot(latsDeg, 2.*cf.overkepler*cf.rsphere**(-1.5)*np.sin(lats), color='g', linewidth=1)

    #divergence
    divm=np.fabs(divg).max()
    visualizeMap(axs[2], 
                 lonsDeg, latsDeg, 
                 divg,  
                 -1.1*divm, 1.1*divm, 
                 title="Divergence")
    axs[2].plot([tanrat(angmox, angmoy)*180./np.pi], [np.arcsin(angmoz/vangmo)*180./np.pi], 'or')


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
    axs[4].plot([tanrat(angmox, angmoy)*180./np.pi], [np.arcsin(angmoz/vangmo)*180./np.pi], 'or')
    axs[4].contour(
        lonsDeg,
        latsDeg,
        accflag,
        levels=[0.5],
        color='k',
        linewidths=1,
    )

    visualizeTwoprofiles(axs[5], 
                         lonsDeg, latsDeg, 
                         sig/sig_init_base, 
                         sig/sig_init_base*accflag, 
                         title1="$\Sigma$", 
                         title2="$\Sigma_0$",
                         log=False)

    #passive scalar
    visualizeMap(axs[6], 
                 lonsDeg, latsDeg, 
                 accflag, 
                 -0.1, 1.1,  
                 title=r'Passive scalar')
    axs[6].plot([(np.pi/2.-np.arctan(angmoy/vangmo))*180./np.pi], [np.arcsin(angmoz/angmox)*180./np.pi], 'or')
    #dissipation
    visualizeMap(axs[7], 
                 lonsDeg, latsDeg, 
                 dissipation*sig, 
                 (dissipation*sig).min(), (dissipation*sig).max(),  
                 title=r'Dissipation')
    axs[7].plot([tanrat(angmox, angmoy)*180./np.pi], [np.arcsin(angmoz/vangmo)*180./np.pi], 'or')
    '''
    # passive scalar
    visualizeSprofile(axs[7], 
                      latsDeg,
                      accflag,  
                      title=r'Passive scalar', 
                      log=False)
    '''

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
    axs[7].plot([tanrat(angmox, angmoy)*180./np.pi], [np.arcsin(angmoz/vangmo)*180./np.pi], 'or')


    #velocity distributions
    visualizeTwoprofiles(axs[9], 
                         lonsDeg, latsDeg, 
                         ug, vg, 
                         title1=r"$v_\varphi$", 
                         title2=r"$v_\theta$" )
    axs[9].plot(latsDeg, cf.omega*cf.rsphere*np.cos(lats), color='b', linewidth=1)
    axs[9].plot(latsDeg, cf.overkepler*cf.rsphere**(-0.5)*np.cos(lats), color='g', linewidth=1)



    axs[0].set_title('{:6.2f} ms'.format( t*cf.tscale*1e3) )
    scycle = str(nout).rjust(6, '0')
    plt.savefig('out/swater'+scycle+'.png' ) #, bbox_inches='tight') 


