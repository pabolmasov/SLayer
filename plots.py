# module for all the visualization tools & functions

import numpy as np
import scipy.ndimage as spin
import matplotlib.pyplot as plt
import numpy.ma as ma

# font adjustment:
import matplotlib
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

##################################################

# converts two components to an angle (change to some implemented function!)
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

def visualizePoles(ax, angmo):
    # axes and angular momentum components (3-tuple)
    x,y,z=angmo
    polelon = tanrat(x, y)
    polelat = np.arcsin(z/np.sqrt(x**2+y**2+z**2))
    polelonDeg=((polelon/np.pi+1.)%2.)*180.-180. ;polelatDeg=(polelat/np.pi)*180.
    ax.plot([polelonDeg], [polelatDeg], '.r')
    ax.plot([(polelonDeg+360.) % 360.-180.], [-polelatDeg], '.r')
    
def visualizeSprofile(ax, latsDeg, data, title="", log=False):
    # latitudal profile
    ax.cla()
    ax.plot(latsDeg, data, ',k',markersize=2)
    ax.set_xlabel('latitude, deg')
    ax.set_ylabel(title)
    if(log):
        ax.set_yscale('log')

def visualizeTwoprofiles(ax, lonsDeg, latsDeg, data1, data2, title1="", title2="", log=False):
    # latitudal profile
    ax.cla()
    ax.plot(latsDeg, data2, ',r',markersize=2)
    ax.plot(latsDeg, data1, ',k',markersize=2)
    if(title1 == "$\Sigma$"):
        ax.plot(latsDeg, data1-data2, '.b',markersize=2)

    datamin=data1.min() ;    datamax=data1.max()
    if(data2.min()<datamin):
        datamin=data2.min()
    if(data2.max()>datamax):
        datamax=data2.max()
        
    ax.set_ylim(datamin, datamax)
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
    wnan=np.where(np.isnan(data))
    data_masked=ma.masked_where(np.isnan(data),data)
    #make fancy 
    ax.minorticks_on()
    ax.set_ylabel(title)

    #ax.set_xlabel('longitude')
    #ax.set_ylabel('latitude')

    ax.set_xticks(np.arange(-180,181,60))
    ax.set_yticks(np.linspace(-90,90,10))

    pc=ax.pcolormesh(
            lonsDeg,
            latsDeg,
            data_masked,
            vmin=vmin,
            vmax=vmax,
            cmap='plasma',
            )
    plt.colorbar(pc, ax=ax)
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
              vortg, divg, ug, vg, sig, press, beta, accflag, dissipation,
#              mass, energy,
#              engy,
#              hbump,
              rsphere,
              cf):
    engy=(ug**2+vg**2)/2.
    #prepare figure etc
    fig = plt.figure(figsize=(10,10))
    gs = plt.GridSpec(5, 10)
    gs.update(hspace = 0.2)
    gs.update(wspace = 0.6)

    axs = []
    for row in [0,1,2,3,4]:
        axs.append( plt.subplot(gs[row, 0:5]) )
        axs.append( plt.subplot(gs[row, 6:10]) )
        lonsDeg = (180./np.pi)*lons-180.
        latsDeg = (180./np.pi)*lats

    nlons=np.size(lons) ; nlats=np.size(lats)
    
    print "visualize: accreted fraction from "+str(accflag.min())+" to "+str(accflag.max())
    
    vorm=np.fabs(vortg-2.*cf.omega*np.sin(lats)).max()

    mdot=cf.sigplus * 4. * np.pi * cf.latspread * cf.rsphere**2 *np.sqrt(4.*np.pi)
    mdot_msunyr = mdot * 1.58649e-26 / cf.tscale
    mass=sig.sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
    mass_acc=(sig*accflag).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
    mass_native=(sig*(1.-accflag)).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
    energy=(sig*engy).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
    angmoz=(sig*ug*np.cos(lats)).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**3
    angmox=(sig*ug*np.sin(lats)*np.cos(lons)).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**3
    angmoy=(sig*ug*np.sin(lats)*np.sin(lons)).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**3
    vangmo=np.sqrt(angmox**2+angmoy**2+angmoz**2) # total angular momentum 

    print "t = "+str(t)
    print "angular momentum "+str(vangmo)+", inclined wrt z by "+str(np.arccos(angmoz/vangmo)*180./np.pi)+"deg"
    print "net angular momentum "+str(vangmo/mass)
    print "vorticity: "+str(vortg.min())+" to "+str(vortg.max())
    print "divergence: "+str(divg.min())+" to "+str(divg.max())
    print "azimuthal U: "+str(ug.min())+" to "+str(ug.max())
    print "polar V: "+str(vg.min())+" to "+str(vg.max())
    print "Sigma: "+str(sig.min())+" to "+str(sig.max())
    print "Pi: "+str(press.min())+" to "+str(press.max())
    print "accretion flag: "+str(accflag.min())+" to "+str(accflag.max())
    print "maximal dissipation "+str(dissipation.max())
    print "minimal dissipation "+str(dissipation.min())
    print "total mass = "+str(mass)
    print "accreted mass = "+str(mass_acc)
    print "native mass = "+str(mass_native)
    print "mdot = "+str(mdot_msunyr)
    print "estimated accreted mass = "+str(mdot*t*cf.tscale)
    print "total energy = "+str(energy)
    print "net energy = "+str(energy/mass)

    dismax=(dissipation*sig).max()

    cspos=(press/sig+np.fabs(press/sig))/2.
    
    #vorticity
    visualizeMap(axs[0], 
                 lonsDeg, latsDeg, 
                 vortg-2.*cf.omega*np.sin(lats), 
                 -vorm*1.1, vorm*1.1, 
                 title="$\Delta \omega$")
    # pressure
    visualizeMap(axs[1], 
                 lonsDeg, latsDeg, 
                 cspos, 
                 (cspos).min(), (cspos).max(), 
                 title="$\Pi/\Sigma c^2$")
    
    #    axs[0].plot([tanrat(angmox, angmoy)*180./np.pi], [np.arcsin(angmoz/vangmo)*180./np.pi], 'or')

    #
#    visualizeSprofile(axs[1], 
#                      latsDeg, 
#                      vortg,
#                      title=r"$v_\varphi$")
#    axs[1].plot(latsDeg, 2.*cf.omega*np.sin(lats), color='r', linewidth=1)
#    axs[1].plot(latsDeg, 2.*cf.overkepler*cf.rsphere**(-1.5)*np.sin(lats), color='g', linewidth=1)

    #divergence
    divm=np.fabs(divg).max()
    visualizeMap(axs[2], 
                 lonsDeg, latsDeg, 
                 divg,  
                 -1.1*divm, 1.1*divm, 
                 title="Divergence")
#    axs[2].plot([tanrat(angmox, angmoy)*180./np.pi], [np.arcsin(angmoz/vangmo)*180./np.pi], 'or')

    
    # gas-to-total pressure ratio
    visualizeMap(axs[3], 
                 lonsDeg, latsDeg, 
                 beta, 
                 beta.min(), beta.max(), 
                 title=r'$\beta$')

#    visualizeSprofile(axs[3], 
#                      latsDeg, 
#                      divg, 
#                      title=r"$(\nabla \cdot v)$")
    # sigma
    sigpos=(sig+np.fabs(sig))/2.+cf.sigfloor
    sig_init_base = cf.sig0*np.exp(0.5*(cf.omega*cf.rsphere*np.cos(lats))**2/cf.csqinit)
    # cf.sig0*np.exp(-(cf.omega*cf.rsphere)**2/cf.csqmin/2.*(1.-np.cos(lats)))

    visualizeMap(axs[4], 
                 lonsDeg, latsDeg, 
                 np.log(sigpos/sig_init_base),  
                 np.log((sigpos/sig_init_base).min()*0.9),  np.log((sigpos/sig_init_base).max()*1.1),  
                 title=r'$\Sigma$')
#    axs[4].plot([tanrat(angmox, angmoy)*180./np.pi], [np.arcsin(angmoz/vangmo)*180./np.pi], 'or')
    axs[4].contour(
        lonsDeg,
        latsDeg,
        accflag,
        levels=[0.5],
        colors='w',
        linewidths=1,
    )

    visualizeTwoprofiles(axs[5], 
                         lonsDeg, latsDeg, 
                         sigpos, 
                         sigpos*accflag, 
                         title1="$\Sigma$", 
                         title2="$\Sigma_0$",
                         log=True)
    axs[5].plot(cf.sigmax, color='g', linestyle='dotted')
    #passive scalar
    visualizeMap(axs[6], 
                 lonsDeg, latsDeg, 
                 accflag, 
                 -0.1, 1.1,  
                 title=r'Passive scalar')
#    axs[6].plot([(np.pi/2.-np.arctan(angmoy/vangmo))*180./np.pi], [np.arcsin(angmoz/angmox)*180./np.pi], 'or')
    #dissipation
    visualizeMap(axs[7], 
                 lonsDeg, latsDeg, 
                 dissipation*sig, 
                 (dissipation*sig).min(), (dissipation*sig).max(),  
                 title=r'Dissipation')
    visualizePoles(axs[7], (angmox, angmoy, angmoz))
#    axs[7].plot([tanrat(angmox, angmoy)%np.pi*180./np.pi], [np.arcsin(angmoz/vangmo)*180./np.pi], '.r')
#    axs[7].plot([-(tanrat(angmox, angmoy)%np.pi)*180./np.pi], [-np.arcsin(angmoz/vangmo)*180./np.pi], '.r')
    '''
    # passive scalar
    visualizeSprofile(axs[7], 
                      latsDeg,
                      accflag,  
                      title=r'Passive scalar', 
                      log=False)
    '''

    #velocities
    du=ug # -cf.omega*cf.rsphere*np.cos(lats)
    dv=vg
    vabs=du**2+dv**2
    dunorm=du/vabs
    dvnorm=dv/vabs

    visualizeMapVecs(axs[8], 
                     lonsDeg, latsDeg, 
                     ug, 
                     vg, 
                     title="Velocities")
    visualizePoles(axs[8], (angmox, angmoy, angmoz))

    #velocity distributions
    visualizeTwoprofiles(axs[9], 
                         lonsDeg, latsDeg, 
                         ug, vg, 
                         title1=r"$v_\varphi$", 
                         title2=r"$v_\theta$" )
    axs[9].plot(latsDeg, cf.omega*cf.rsphere*np.cos(lats), color='b', linewidth=1)
    axs[9].plot(latsDeg, cf.overkepler*cf.rsphere**(-0.5)*np.cos(lats), color='g', linewidth=1)

    axs[0].set_title('{:7.3f} ms'.format( t*cf.tscale*1e3) )
    scycle = str(nout).rjust(6, '0')
    plt.savefig('out/swater'+scycle+'.png' ) #, bbox_inches='tight') 
    plt.close()
    plt.clf()
    plt.plot(cspos, beta, '.k')
    plt.xscale('log')
    plt.xlabel(r'$c_{\rm s}$')
    plt.ylabel(r'$\beta$')
    plt.savefig('csbeta.eps')
    plt.close()
##########################################################################
#    
##########################################################################    
# post-factum visualizations form snapshooter:
def snapplot(lons, lats, sig, accflag, vx, vy, sks):
    # longitudes, latitudes, density field, accretion flag, velocity fields, alias for velocity output
    skx=sks[0] ; sky=sks[1]

    wpoles=np.where(np.fabs(lats)<90.)
    s0=sig[wpoles].min() ; s1=sig[wpoles].max()
    #    s0=0.1 ; s1=10. # how to make a smooth estimate?
    nlev=30
    levs=(s1/s0)**(np.arange(nlev)/np.double(nlev-1))*s0

    plt.clf()
    fig=plt.figure()
    plt.contourf(lons, lats, np.log10(sig),cmap='jet') #,levels=levs)
    plt.colorbar()
    plt.contour(lons, lats, accflag, levels=[0.5], colors='w') #,levels=levs)
    plt.quiver(lons[::skx, ::sky],
        lats[::skx, ::sky],
        vx[::skx, ::sky], vy[::skx, ::sky],
        pivot='mid',
        units='x',
        linewidth=1.0,
        color='k',
        scale=20.0,
    )
#    plt.ylim(-85.,85.)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    fig.set_size_inches(8, 5)
    plt.savefig('out/snapshot.png')
    plt.savefig('out/snapshot.eps')
    plt.close()
    # drawing poles:
    nlons=np.size(lons)
    tinyover=1./np.double(nlons)
    theta=lats
    plt.clf()
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    #    wnorth=np.where(lats>0.)
    tinyover=1./np.double(nlons)
    ax.contourf(lons*np.pi/180.*(tinyover+1.), theta, sig,cmap='jet',levels=levs)
    ax.contour(lons*np.pi/180.*(tinyover+1.), theta, accflag,colors='w',levels=[0.5])
    ax.set_rticks([30., 60.])
    ax.set_rmax(90.)
    plt.title('N') #, t='+str(nstep))
    plt.tight_layout()
    fig.set_size_inches(4, 4)
    plt.savefig('out/northpole.eps')
    plt.savefig('out/northpole.png')
    plt.close()
    plt.clf()
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    #    wnorth=np.where(lats>0.)
#    tinyover=0./np.double(nlons)
    ax.contourf(lons*np.pi/180.*(tinyover+1.), 180.*(1.+tinyover)-theta, sig,cmap='jet',levels=levs)
    ax.contour(lons*np.pi/180.*(tinyover+1.), 180.*(1.+tinyover)-theta, accflag,colors='w',levels=[0.5])
    ax.set_rticks([30., 60.])
    ax.set_rmax(90.)
    plt.tight_layout(pad=2)
    fig.set_size_inches(4, 4)
    plt.title('S') #, t='+str(nstep))
    plt.savefig('out/southpole.eps')
    plt.savefig('out/southpole.png')
    plt.close()

# general framework for a post-processed map of some quantity q
def somemap(lons, lats, q, outname):
    wnan=np.where(np.isnan(q))
    nnan=np.size(wnan)
    print outname+" somemap: "+str(nnan)+"NaN points out of "+str(np.size(q))
    plt.clf()
    fig=plt.figure()
    plt.contourf(lons, lats, q,cmap='jet') #,levels=levs)
    plt.colorbar()
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    fig.set_size_inches(8, 5)
    plt.savefig(outname)
    plt.close()
    
# vorticity correlated with other quantities
def vortgraph(lats, lons, vort, sig, energy, omegaNS, lonrange=[0.,360.]):
    lon1=lonrange[0] ; lon2=lonrange[1]
    w=np.where((lons>lon1)&(lons<lon2))
    do=vort+2.*omegaNS*np.cos(lats*np.pi/180.)
    plt.clf()
    plt.plot(lats, vort, ',k')
    plt.plot(lats, -2.*omegaNS*np.cos(lats*np.pi/180.),',r')
    plt.ylabel(r'$\omega$')
    plt.xlabel(r'$\theta$, deg')
    plt.savefig('out/vortgraph.eps')
    plt.close()
    domedian=np.median(do,axis=1) ; csmedian=np.median((energy/sig),axis=1)
    plt.clf()
    plt.plot(domedian, csmedian, color='k')
    plt.scatter(do[w], (energy/sig)[w], c=lats[w]*np.pi/180., cmap='jet', s=((lons[w]-(lon1+lon2)/2.)/(lon2-lon1))**2*100., marker='d', facecolors='none')
#    plt.plot((vort+2.*omegaNS*np.cos(lats*np.pi/180.))[w], (energy/sig)[w], markerfacecolors='none', markeredgecolors=lats[w]*np.pi/180., cmap='jet', markersize=((lons[w]-(lon1+lon2)/2.)/(lon2-lon1))**2*50.)
    plt.xlabel(r'$\Delta\omega$')
    plt.ylabel(r'$E/\Sigma$')
    #    plt.xscale('log')
    # plt.yscale('log')
    plt.ylim(np.percentile((energy/sig)[w], 1.), np.percentile((energy/sig)[w], 99.9)*1.2)
    plt.xlim(np.percentile(do[w], 1.)*1.1, np.percentile(do[w], 99.9)*1.1)
    plt.savefig('out/vortcs.eps')
    plt.close()

def dissgraph(sig, energy, diss, vsq, accflag):
    w=np.where(accflag>0.75)
    w0=np.where(accflag<0.25)
    plt.clf()
    plt.plot(sig, vsq/2., '.g')
    plt.plot(sig, energy/sig, '.k')
    plt.plot(sig[w], energy[w]/sig[w], '.b')
    plt.plot(sig[w0], energy[w0]/sig[w0], '.r')
    plt.xlabel(r'$\Sigma$')
    plt.ylabel(r'$E/\Sigma$')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim((energy/sig).min(),(energy/sig).max())
    plt.savefig('out/effeos.eps')
    plt.close()
    plt.clf()
    plt.plot(sig, diss, ',k')
    plt.plot(sig[w], diss[w], ',b')
    plt.plot(sig[w0], diss[w0], ',r')
    plt.xlabel(r'$\Sigma$')
    plt.ylabel(r'dissipation')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('out/dissigma.eps')
    plt.close()

# Effective gravity and Eddington violation diagnostic plot
def sgeffplot(sig, grav, geff, radgeff):
    plt.clf()
    plt.plot(sig, radgeff+geff, ',k')
    plt.plot(sig, geff, ',b')
    plt.plot(sig, geff*0.-grav, color='r')
    plt.xscale('log')
    plt.xlabel(r'$\Sigma$, g\,cm$^{-2}$')
    plt.ylabel(r'$g_{\rm eff}$, $GM/c$ units')
    plt.savefig('out/sgeff.eps')
    plt.close()

# Reynolds stress plot (maybe we should make a time-averaged plot; think of it!)
def reys(lons, lats, sig, ug,vg, energy, rsphere):

    omega=ug/np.cos(lats)/rsphere
    # shear domega/dtheta:
    nx, ny=np.shape(omega)
    omegamean=omega.mean(axis=1)
    dodthmean = (omegamean[1:]-omegamean[:-1])/(lats[1:,0]-lats[:-1,0])
#    dodthmean = dodth.mean(axis=1) / (omega[1:,:]+omega[:-1,:]).mean(axis=1)
    omegamean=(omegamean[1:]+omegamean[:-1])/2.
    energymean = (energy[1:,:]+energy[:-1,:]).mean(axis=1)/2.
    dug=ug ; dvg=vg
    for kx in np.arange(nx):
        dug[kx,:]=ug[kx,:]-ug[kx,:].mean()
        dvg[kx,:]=vg[kx,:]-vg[kx,:].mean()
    rxy = sig * dug * dvg
    rxy0 = sig * ug * vg
#    rxy0mean = (rxy0[1:,:]+rxy0[:-1,:]).mean(axis=1)/2.
    rxymean = (rxy[1:,:]+rxy[:-1,:]).mean(axis=1)/2.
    latsmidpoint=(lats[1:,0]+lats[:-1,0])/2.
    plt.clf()
    plt.plot(latsmidpoint, rxymean, color='k')
#    plt.plot(latsmidpoint, rxy0mean, color='r')
    plt.plot(latsmidpoint, energymean*dodthmean/omegamean, '.b')
    plt.xlabel(r'latitude, degrees')
    plt.ylabel(r'stress')
#    plt.ylim([rxymean.min(), rxymean.max()])
    plt.savefig('out/rxy.eps')
    plt.close()

    
