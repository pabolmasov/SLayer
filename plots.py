from __future__ import print_function
from __future__ import division
# module for all the visualization tools & functions

from builtins import str
from past.utils import old_div
import numpy as np
import scipy.ndimage as spin
import matplotlib.pyplot as plt
import numpy.ma as ma
from scipy.integrate import trapz

# font adjustment:
import matplotlib
from matplotlib import font_manager
from matplotlib import rc
rc('font',**{'family':'serif'})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 
from matplotlib import interactive

interactive(False)

##################################################

# converts two components to an angle (change to some implemented function!)
def tanrat(x,y):
    sx=np.size(x) ; sy=np.size(y)
    if(sx>1):
        if(sx!=sy):
            print("tanrat: array sizes do not match, "+str(sx)+" !="+str(sy))
            sx=minimum(sx,sy)
        z=np.zeros(sx, dtype=np.double)
        for k in np.arange(sx):
            z[k]=tanrat(x[k],y[k])
        return z
    else:
        z=0.
        if((x*y)>0.):
            z=np.arctan(old_div(y,x))
            if(x<0.):
                z+=np.pi
        elif((x*y)<0.):
            z=np.arctan(old_div(y,x))+np.pi
            if(y<0.):
                z+=np.pi
        else:
            if(x==0.):
                if(y<0.):
                    return 1.5*np.pi
                else:
                    return np.pi*0.5
            else:
                if(x<0.):
                    return np.pi
        return z

def visualizePoles(ax, angmo):
    # axes and angular momentum components (3-tuple)
    x,y,z=angmo
    polelon = tanrat(x, y)
    polelat = np.arcsin(z/np.sqrt(x**2+y**2+z**2))
    polelonDeg=180.*(polelon/np.pi-1.) ; polelatDeg=polelat/np.pi*180.
    ax.plot([polelonDeg], [polelatDeg], '.r')
    ax.plot([(polelonDeg+360.)%360.-180.], [-polelatDeg], '.r')
    
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

    print(title, " min/max:", data.min(), data.max())
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

    print(title, " min/max vec len: ", M.min(), M.max())

    vv=np.sqrt(xx**2+yy**2)
    vvmax=vv.std() # normalization
    nlons=np.size(np.unique(lonsDeg))
    sk = int(np.floor(5.*np.double(nlons)/128.))
    #    print "nlons="+str(nlons)
    sigma = [old_div(sk,2.), old_div(sk,2.)]
    xx = spin.filters.gaussian_filter(xx, sigma, mode='constant')*30./vvmax
    yy = spin.filters.gaussian_filter(yy, sigma, mode='constant')*30./vvmax
    
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
              vortg, divg, ug, vg, sig, press, beta, accflag, qminus, qplus,
              cf, outdir):
    energy= old_div(press, (3. * (1.-old_div(beta,2.))))# (ug**2+vg**2)/2.
    #prepare figure etc
    fig = plt.figure(figsize=(10,10))
    gs = plt.GridSpec(5, 10)
    gs.update(hspace = 0.2)
    gs.update(wspace = 0.6)

    axs = []
    for row in [0,1,2,3,4]:
        axs.append( plt.subplot(gs[row, 0:5]) )
        axs.append( plt.subplot(gs[row, 6:10]) )
        lonsDeg = (old_div(180.,np.pi))*lons-180.
        latsDeg = (old_div(180.,np.pi))*lats

    nlons=np.size(lons) ; nlats=np.size(lats)
    lats1d=np.unique(lats)
    clats=np.sin(lats1d)
    print("visualize: accreted fraction from "+str(accflag.min())+" to "+str(accflag.max()))
    
    vorm=np.fabs(vortg-2.*cf.omega*np.sin(lats)).max()

    mdot=cf.sigplus * 4. * np.pi * np.sin(cf.latspread) * cf.rsphere**2 *np.sqrt(4.*np.pi)
    mdot_msunyr = mdot * 1.58649e-18 / cf.tscale
    mass=trapz(sig.mean(axis=1), x=clats)
    mass_acc=trapz((sig*accflag).mean(axis=1), x=clats)
    mass_native=trapz((sig*(1.-accflag)).mean(axis=1), x=clats)

    #    mass=sig.sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
    #    mass_acc=(sig*accflag).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
    #    mass_native=(sig*(1.-accflag)).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
    thenergy=trapz(energy.mean(axis=1), x=clats)
    kenergy=old_div(trapz((sig*(ug**2+vg**2)).mean(axis=1), x=clats),2.)
#    (sig*engy).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
    angmoz=trapz((sig*ug*np.cos(lats)).mean(axis=1), x=clats)*cf.rsphere
    angmox=trapz((sig*(vg*np.sin(lons)-ug*np.sin(lats)*np.cos(lons))).mean(axis=1), x=clats)*cf.rsphere
    angmoy=trapz((sig*(vg*np.cos(lons)-ug*np.sin(lats)*np.sin(lons))).mean(axis=1), x=clats)*cf.rsphere
    vangmo=np.sqrt(angmox**2+angmoy**2+angmoz**2) # total angular momentum 

    print("t = "+str(t))
    print("angular momentum "+str(vangmo)+", inclined wrt z by "+str(np.arccos(old_div(angmoz,vangmo))*180./np.pi)+"deg")
    print("net angular momentum "+str(old_div(vangmo,mass)))
    print("vorticity: "+str(vortg.min())+" to "+str(vortg.max()))
    print("divergence: "+str(divg.min())+" to "+str(divg.max()))
    print("azimuthal U: "+str(ug.min())+" to "+str(ug.max()))
    print("polar V: "+str(vg.min())+" to "+str(vg.max()))
    print("Sigma: "+str(sig.min())+" to "+str(sig.max()))
    print("Pi: "+str(press.min())+" to "+str(press.max()))
    print("accretion flag: "+str(accflag.min())+" to "+str(accflag.max()))
    print("maximal Q^- "+str(qminus.max()))
    print("minimal Q^- "+str(qminus.min()))
    print("maximal Q^+ "+str(qplus.max()))
    print("minimal Q^+ "+str(qplus.min()))
    print("total mass = "+str(mass))
    print("accreted mass = "+str(mass_acc))
    print("native mass = "+str(mass_native))
    print("mdot = "+str(mdot_msunyr))
    print("estimated accreted mass = "+str(mdot*t*cf.tscale))
    print("total energy = "+str(thenergy)+"(thermal) + "+str(kenergy)+"(kinetic)")
    print("net energy = "+str(old_div((thenergy+kenergy),mass)))

    cspos=old_div((old_div(press,sig)+np.fabs(old_div(press,sig))),2.)
    
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
                 0., 1., 
                 title=r'$\beta$')

#    visualizeSprofile(axs[3], 
#                      latsDeg, 
#                      divg, 
#                      title=r"$(\nabla \cdot v)$")
    # sigma
    sigpos=old_div((sig+np.fabs(sig)),2.)+cf.sigfloor
#    sig_init_base = cf.sig0*np.exp(0.5*(cf.omega*cf.rsphere*np.cos(lats))**2/cf.csqinit)
    # cf.sig0*np.exp(-(cf.omega*cf.rsphere)**2/cf.csqmin/2.*(1.-np.cos(lats)))

    visualizeMap(axs[4], 
                 lonsDeg, latsDeg, 
                 np.log(sigpos),  
                 np.log(sigpos.min()*0.9),  np.log(sigpos.max()*1.1),  
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
    #passive scalar
    visualizeMap(axs[6], 
                 lonsDeg, latsDeg, 
                 accflag, 
                 -0.1, 1.1,  
                 title=r'Passive scalar')
#    axs[6].plot([(np.pi/2.-np.arctan(angmoy/vangmo))*180./np.pi], [np.arcsin(angmoz/angmox)*180./np.pi], 'or')
    #Q^-
    visualizeMap(axs[7], 
                 lonsDeg, latsDeg, 
                 np.log(qminus), 
                 np.log(qminus.min()), np.log(qminus.max()),  
                 title=r'$\ln Q^\pm$')
    visualizePoles(axs[7], (angmox, angmoy, angmoz))
    axs[7].contour(
        lonsDeg,
        latsDeg,
        np.log(qplus),
        colors='w',
        linewidths=1,
        levels=[np.log(qminus.min()), np.log(qminus.mean()), np.log(qminus.max())]
    )
    #velocities
    du=ug # -cf.omega*cf.rsphere*np.cos(lats)
    dv=vg
    vabs=du**2+dv**2
    dunorm=old_div(du,vabs)
    dvnorm=old_div(dv,vabs)

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
    axs[9].set_ylim(ug.min()+vg.min(), ug.max()+vg.max())
    axs[0].set_title('{:7.3f} ms'.format( t*cf.tscale*1e3) )
    scycle = str(nout).rjust(6, '0')
    plt.savefig(outdir+'/swater'+scycle+'.png' ) #, bbox_inches='tight') 
    plt.close()
##########################################################################
#    
##########################################################################    
# post-factum visualizations from snapshooter:
def snapplot(lons, lats, sig, accflag, vx, vy, sks, outdir='out'):
    # longitudes, latitudes, density field, accretion flag, velocity fields, alias for velocity output
    skx=sks[0] ; sky=sks[1]

    wpoles=np.where(np.fabs(lats)<90.)
    s0=sig[wpoles].min() ; s1=sig[wpoles].max()
    #    s0=0.1 ; s1=10. # how to make a smooth estimate?
    nlev=30
    levs=(old_div(s1,s0))**(old_div(np.arange(nlev),np.double(nlev-1)))*s0
    interactive(False)

    plt.clf()
    fig=plt.figure()
    plt.contourf(lons, lats, sig, cmap='jet',levels=levs)
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
    plt.ylim(-85.,85.)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    fig.set_size_inches(8, 5)
    plt.savefig(outdir+'/snapshot.png')
    plt.savefig(outdir+'/snapshot.eps')
    plt.close()
    # drawing poles:
    nlons=np.size(lons)
    tinyover=old_div(1.,np.double(nlons))
    theta=90.-lats
    plt.clf()
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    #    wnorth=np.where(lats>0.)
    tinyover=old_div(1.,np.double(nlons))
    ax.contourf(lons*np.pi/180.*(tinyover+1.), theta, sig,cmap='jet',levels=levs)
    ax.contour(lons*np.pi/180.*(tinyover+1.), theta, accflag,colors='w',levels=[0.5])
    ax.set_rticks([30., 60.])
    ax.set_rmax(90.)
    plt.title('N') #, t='+str(nstep))
    plt.tight_layout()
    fig.set_size_inches(4, 4)
    plt.savefig(outdir+'/northpole.eps')
    plt.savefig(outdir+'/northpole.png')
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
    plt.savefig(outdir+'/southpole.eps')
    plt.savefig(outdir+'/southpole.png')
    plt.close()

# general framework for a post-processed map of some quantity q
def somemap(lons, lats, q, outname):
    wnan=np.where(np.isnan(q))
    nnan=np.size(wnan)
    print(outname+" somemap: "+str(nnan)+"NaN points out of "+str(np.size(q)))
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
def vortgraph(lats, lons, vort, div, sig, energy, omegaNS, lonrange=[0.,360.], outdir='out'):
    lon1=lonrange[0] ; lon2=lonrange[1]
    w=np.where((lons>lon1)&(lons<lon2))
    do=vort+2.*omegaNS*np.cos(lats*np.pi/180.)
    plt.clf()
    plt.scatter(vort, div, c=sig)
    plt.colorbar()
    plt.plot(2.*omegaNS*np.sin(lats*np.pi/180.), div, ',k')
    plt.ylabel(r'$\delta$')
    plt.xlabel(r'$\omega$')
    plt.xlim(vort.min(), vort.max())
    plt.ylim(div.min(), div.max())
    plt.savefig(outdir+'/vortdiv.eps')
    plt.savefig(outdir+'/vortdiv.png')
    plt.close()
    
    plt.clf()
    plt.plot(lats, vort, ',k')
    plt.plot(lats, -2.*omegaNS*np.cos(lats*np.pi/180.),',r')
    plt.ylabel(r'$\omega$')
    plt.xlabel(r'$\theta$, deg')
    plt.savefig(outdir+'/vortgraph.eps')
    plt.close()
    domedian=np.median(do,axis=1) ; csmedian=np.median((old_div(energy,sig)),axis=1)
    plt.clf()
    plt.plot(domedian, csmedian, color='k')
    plt.scatter(do[w], (old_div(energy,sig))[w], c=lats[w]*np.pi/180., cmap='jet', s=(old_div((lons[w]-old_div((lon1+lon2),2.)),(lon2-lon1)))**2*100., marker='d', facecolors='none')
#    plt.plot((vort+2.*omegaNS*np.cos(lats*np.pi/180.))[w], (energy/sig)[w], markerfacecolors='none', markeredgecolors=lats[w]*np.pi/180., cmap='jet', markersize=((lons[w]-(lon1+lon2)/2.)/(lon2-lon1))**2*50.)
    plt.xlabel(r'$\Delta\omega$')
    plt.ylabel(r'$E/\Sigma$')
    #    plt.xscale('log')
    # plt.yscale('log')
    plt.ylim(np.percentile((old_div(energy,sig))[w], 1.), np.percentile((old_div(energy,sig))[w], 99.9)*1.2)
    plt.xlim(np.percentile(do[w], 1.)*1.1, np.percentile(do[w], 99.9)*1.1)
    plt.savefig(outdir+'/vortcs.eps')
    plt.close()

def dissgraph(sig, energy, diss, vsq, accflag, outdir='out'):
    w=np.where(accflag>0.75)
    w0=np.where(accflag<0.25)
    plt.clf()
    plt.plot(sig, old_div(vsq,2.), '.g')
    plt.plot(sig, old_div(energy,sig), '.k')
    plt.plot(sig[w], old_div(energy[w],sig[w]), '.b')
    plt.plot(sig[w0], old_div(energy[w0],sig[w0]), '.r')
    plt.xlabel(r'$\Sigma$')
    plt.ylabel(r'$E/\Sigma$')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim((old_div(energy,sig)).min(),(old_div(energy,sig)).max())
    plt.savefig(outdir+'/effeos.eps')
    plt.close()
    plt.clf()
    plt.plot(sig, diss, ',k')
    plt.plot(sig[w], diss[w], ',b')
    plt.plot(sig[w0], diss[w0], ',r')
    plt.xlabel(r'$\Sigma$')
    plt.ylabel(r'dissipation')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(outdir+'/dissigma.eps')
    plt.close()

# Effective gravity and Eddington violation diagnostic plot
def sgeffplot(sig, grav, geff, radgeff, outdir='out'):
    plt.clf()
    plt.plot(sig, radgeff+geff, ',k')
    plt.plot(sig, geff, ',b')
    plt.plot(sig, geff*0.-grav, color='r')
    plt.xscale('log')
    plt.xlabel(r'$\Sigma$, g\,cm$^{-2}$')
    plt.ylabel(r'$g_{\rm eff}$, $c^4/GM$ units')
    plt.savefig(outdir+'/sgeff.eps')
    plt.close()

# Reynolds stress plot (maybe we should make a time-averaged plot; think of it!)
def reys(lons, lats, sig, ug,vg, energy, rsphere, outdir='out'):

    omega=ug/np.cos(lats)/rsphere
    # shear domega/dtheta:
    nx, ny=np.shape(omega)
    omegamean=omega.mean(axis=1)
    dodthmean = old_div((omegamean[1:]-omegamean[:-1]),(lats[1:,0]-lats[:-1,0]))
#    dodthmean = dodth.mean(axis=1) / (omega[1:,:]+omega[:-1,:]).mean(axis=1)
    omegamean=old_div((omegamean[1:]+omegamean[:-1]),2.)
    energymean = old_div((energy[1:,:]+energy[:-1,:]).mean(axis=1),2.)
    dug=ug ; dvg=vg
    for kx in np.arange(nx):
        dug[kx,:]=ug[kx,:]-ug[kx,:].mean()
        dvg[kx,:]=vg[kx,:]-vg[kx,:].mean()
    rxy = sig * dug * dvg
    rxy0 = sig * ug * vg
#    rxy0mean = (rxy0[1:,:]+rxy0[:-1,:]).mean(axis=1)/2.
    rxymean = old_div((rxy[1:,:]+rxy[:-1,:]).mean(axis=1),2.)
    latsmidpoint=old_div((lats[1:,0]+lats[:-1,0]),2.)
    plt.clf()
    plt.plot(latsmidpoint, rxymean, color='k')
#    plt.plot(latsmidpoint, rxy0mean, color='r')
    plt.plot(latsmidpoint, energymean*dodthmean/omegamean, '.b')
    plt.xlabel(r'latitude, degrees')
    plt.ylabel(r'stress')
#    plt.ylim([rxymean.min(), rxymean.max()])
    plt.savefig(outdir+'/rxy.eps')
    plt.close()

########################################################################
# post-processing of remotely produced light curves and spectra
def pdsplot(infile="out/pdstots_diss"):
    lines = np.loadtxt(infile+".dat", comments="#", delimiter=" ", unpack=False)
    freq1=lines[:,0] ; freq2=lines[:,1]
    fc=(freq1+freq2)/2. # center of frequency interval
    f=lines[:,2] ; df=lines[:,3] # replace df with quantiles!
    nf=np.size(f)
    plt.clf()
    for kf in np.arange(nf):
        plt.plot([freq1[kf], freq2[kf]], [f[kf], f[kf]], color='k')
        plt.plot([fc[kf], fc[kf]], [f[kf]-df[kf], f[kf]+df[kf]], color='k')
    plt.xlabel(r'$f$, Hz')
    plt.ylabel(r'PDS, relative units')
    plt.xscale('log') ;    plt.yscale('log')
    plt.savefig(infile+'.png')
    plt.savefig(infile+'.eps')
    plt.close()

def dynsplot(infile="out/pds_diss"):
    lines = np.loadtxt(infile+".dat", comments="#", delimiter=" ", unpack=False)
    freq1=lines[:,1] ; freq2=lines[:,2] ;  t=lines[:,0] 
    fc=(freq1+freq2)/2. # center of frequency interval
    f=lines[:,3] ; df=lines[:,4] # replace df with quantiles!
    tun=np.unique(t) ; fun=np.unique(freq1)
    ntimes=np.size(tun) ;    nbins=np.size(fun)
    print("ntimes = "+str(ntimes))
    print("nbins = "+str(nbins))
    t2=np.zeros([ntimes+1, nbins+1], dtype=np.double)
    binfreq2=np.zeros([ntimes+1, nbins+1], dtype=np.double)
    f2=np.transpose(np.reshape(f, [nbins, ntimes]))
    df2=np.transpose(np.reshape(df, [nbins, ntimes]))
    tc=np.transpose(np.reshape(t, [nbins, ntimes]))
    fc=np.transpose(np.reshape(fc, [nbins, ntimes]))
    for kt in np.arange(ntimes-1):
        t2[kt,:]=tun[kt]-(tun[kt+1]-tun[kt])/2.
        binfreq2[kt,:-1]=fun[:]
    f2ma=ma.masked_invalid(f2)
    f2tot=f2ma.sum(axis=1)
#    for kt in np.arange(ntimes):
#        f2ma[kt,:]/=f2tot[kt]
    t2[ntimes-1,:]=tun[ntimes-1]+(tun[ntimes-1]-tun[ntimes-2])/2.
    t2[ntimes,:]=tun[ntimes-1]+(tun[ntimes-1]-tun[ntimes-2])*3./2.
    binfreq2[ntimes-1,:-1]=fun[:] ;   binfreq2[ntimes,:-1]=fun[:]
    binfreq2[:,-1]=freq2.max()
    w=np.isfinite(df2)&(df2>0.)
    pmin=f2ma.min() ; pmax=f2ma.max()
    print(binfreq2.min(),binfreq2.max())
    plt.clf()
    #  plt.contourf(t, fc, f2, cmap='jet')
    plt.pcolor(t2, binfreq2, np.log(f2ma), cmap='jet', vmin=np.log(pmin), vmax=np.log(pmax)) # tcenter2, binfreq2 should be corners
    # plt.contourf(tc, fc, np.log(f2), cmap='jet')
    #    plt.colorbar()
    #    plt.plot([t.min(), t.min()],[omega/2./np.pi,omega/2./np.pi], 'r')
    #    plt.plot([t.min(), t.max()],[2.*omega/2./np.pi,2.*omega/2./np.pi], 'r')
    plt.ylim(freq2.min(), freq2.max())
    plt.yscale('log')
    plt.ylabel('$f$, Hz')
    plt.xlabel('$t$, s')
    plt.savefig(infile+'.png')
    plt.close()
