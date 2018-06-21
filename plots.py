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
import os.path

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

import glob

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
            cmap='hot',
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

# main real-time visualization routine:
def visualize(t, nout,
              lats, lons, 
              vortg, divg, ug, vg, sig, press, beta, accflag, qminus, qplus,
              cf, outdir):
    energy= press/(3. * (1.-beta/2.))
    #prepare figure etc
    fig = plt.figure(figsize=(10,10))
    gs = plt.GridSpec(5, 10)
    gs.update(hspace = 0.2)
    gs.update(wspace = 0.6)

    axs = []
    for row in [0,1,2,3,4]:
        axs.append( plt.subplot(gs[row, 0:5]) )
        axs.append( plt.subplot(gs[row, 6:10]) )
        lonsDeg = 180./np.pi*lons-180.
        latsDeg = 180./np.pi*lats

    nlons=np.size(lons) ; nlats=np.size(lats)
    lats1d=np.unique(lats)
    clats=np.sin(lats1d)
    print("visualize: accreted fraction from "+str(accflag.min())+" to "+str(accflag.max()))
    
    vorm=np.fabs(vortg-2.*cf.omega*np.sin(lats)).max()

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
    print("total energy = "+str(thenergy)+"(thermal) + "+str(kenergy)+"(kinetic)")
    print("net energy = "+str(old_div((thenergy+kenergy),mass)))

    cspos=old_div((old_div(press,sig)+np.fabs(old_div(press,sig))),2.)
    
    #vorticity
    visualizeMap(axs[0], 
                 lonsDeg, latsDeg, 
                 vortg-2.*cf.omega*np.sin(lats), 
                 -vorm*1.1, vorm*1.1, 
                 title="$\Delta \omega$")
    # internal temperature
    tbottom=50.59*((1.-beta)*energy*cf.sigmascale/cf.mass1)**0.25
    visualizeMap(axs[1], 
                 lonsDeg, latsDeg, 
                 tbottom, 
                 tbottom.min(), tbottom.max(), 
                 title=r'$T_{\rm bottom}$, keV')

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

    visualizeMap(axs[4], 
                 lonsDeg, latsDeg, 
                 sig*cf.sigmascale,  
                 sig.min()*0.9*cf.sigmascale,  sig.max()*1.1*cf.sigmascale,  
                 title=r'$\Sigma$')

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
                         sig*cf.sigmascale, 
                         sig*accflag*cf.sigmascale, 
                         title1=r"$\Sigma$", 
                         title2=r"${\rm g \,cm^{-2}}$",
                         log=True)
    #passive scalar
    visualizeMap(axs[6], lonsDeg, latsDeg, 
                 accflag, -0.1, 1.1, title=r'tracer')
#    axs[6].plot([(np.pi/2.-np.arctan(angmoy/vangmo))*180./np.pi], [np.arcsin(angmoz/angmox)*180./np.pi], 'or')
    #Q^-
    teff=(qminus*cf.sigmascale/cf.mass1)**0.25*3.64 # effective temperature in keV
    visualizeMap(axs[7], lonsDeg, latsDeg, 
                 teff, teff.min(), teff.max(),  
                 title=r'$T_{\rm eff}$, keV')
    visualizePoles(axs[7], (angmox, angmoy, angmoz))
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
#
##########################################################################    
# post-factum visualizations for snapshooter:
def snapplot(lons, lats, sig, accflag, tb, vx, vy, sks, outdir='out'
             ,latrange=None, lonrange=None, t=None):
    # longitudes, latitudes, density field, accretion flag, some additional quantity (vorticity difference), velocity fields, alias for velocity output
    if((latrange == None) | (lonrange == None)):
        skx=sks[0] ; sky=sks[1]
    else:
        skx=2 ; sky=2

    wpoles=np.where(np.fabs(lats)<90.)
    s0=tb[wpoles].min() ; s1=tb[wpoles].max()
    #    s0=0.1 ; s1=10. # how to make a smooth estimate?
    nlev=30
    levs=(s1-s0)*((np.arange(nlev)-0.5)/np.double(nlev-1))+s0
    levs=np.unique(np.round(levs, 2))
    interactive(False)

    # TODO: try cartographic projections?
    plt.clf()
    fig=plt.figure()
    pc=plt.pcolormesh(lons, lats, sig, cmap='hot') # ,levels=levs)
    plt.colorbar(pc)
    if(accflag.max()>1e-3):
        plt.contour(lons, lats, accflag, levels=[0.5], colors='w',linestyles='dotted') #,levels=levs)
#    plt.contour(lons, lats, sig, colors='w') #,levels=levs)
    plt.quiver(lons[::skx, ::sky],
        lats[::skx, ::sky],
        vx[::skx, ::sky], vy[::skx, ::sky],
        pivot='mid',
        units='x',
        linewidth=1.0,
        color='k',
        scale=20.0,
    )
#    if((latrange == None) & (lonrange == None)):
        #        plt.ylim(-85.,85.)
#    else:
#        plt.ylim(latrange[0], latrange[1])
#        plt.xlim(lonrange[0], lonrange[1])
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    if t != None:
        plt.title('$t={:8.3f}$ms'.format(t), loc='left')
    fig.set_size_inches(8, 5)
    plt.savefig(outdir+'/snapshot.png')
    plt.savefig(outdir+'/snapshot.eps')
    plt.close()
    # drawing poles:
    nlons=np.size(lons)
    tinyover=1./np.double(nlons)
    theta=90.-lats
    plt.clf()
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    #    wnorth=np.where(lats>0.)
    tinyover=old_div(1.,np.double(nlons))
    pc=ax.contourf(lons*np.pi/180.*(tinyover+1.), theta, tb,cmap='hot',levels=levs)
    #    if(accflag.max()>1e-3):
    #        ax.contour(lons*np.pi/180.*(tinyover+1.), theta, accflag,colors='w',levels=[0.5])
#    ax.grid(color='w')
    plt.colorbar(pc,ax=ax)
    ax.set_rticks([30., 60.])
    ax.grid(color='w')
    ax.set_rmax(70.)
    if t != None:
       # plt.title(r'North pole, $t={:8.3f}$ms'.format(t), loc='left') #, t='+str(nstep))
        plt.text(90., 95.,r'North pole, $t={:8.3f}$ms'.format(t), fontsize=18)
    else:
        plt.title('North pole', loc='left')
#    ax.axis('equal') # never! it ruins polar coordinates!
    plt.tight_layout(pad=3)
    fig.set_size_inches(5, 4)
    plt.savefig(outdir+'/northpole.eps')
    plt.savefig(outdir+'/northpole.png')
    plt.close()
    plt.clf()
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    #    wnorth=np.where(lats>0.)
#    tinyover=0./np.double(nlons)
    pc=ax.contourf(lons*np.pi/180.*(tinyover+1.), 180.*(tinyover+1.)-theta, tb,cmap='hot', levels=levs) #,color='w')
    #    if(accflag.max()>1e-3):
    #        ax.contour(lons*np.pi/180.*(tinyover+1.), 180.*(1.+tinyover)-theta, accflag,colors='w',levels=[0.5])
    plt.colorbar(pc,ax=ax)
    ax.set_rticks([30., 60.])
    ax.grid(color='w')
    ax.set_rmax(70.)
    #    ax.axis('equal') ruins polar scale
    fig.set_size_inches(5, 4)
    if t != None:
        # plt.title(r'South pole, $t={:8.3f}$ms'.format(t), loc='left')
        plt.text(90., 95.,r'South pole, $t={:8.3f}$ms'.format(t), fontsize=18)
    else:
        plt.title('South pole', loc='left')
    plt.tight_layout(pad=3)
    plt.savefig(outdir+'/southpole.eps')
    plt.savefig(outdir+'/southpole.png')
    plt.close()
# post-factum visualizations from the ascii output of snapplot:
def postmaps(infile):
    lines = np.loadtxt(infile+".dat", comments="#", delimiter=" ", unpack=False)
    lats=lines[:,0] ;   lons=lines[:,1] ; sigma=lines[:,2] ; energy=lines[:,5]
    ug=lines[:,3] ; vg=lines[:,4] ; diss=lines[:,6] ; accflag=lines[:,7]
    nlats=np.size(np.unique(lats)) ;   nlons=np.size(np.unique(lons))
    lats=np.reshape(lats,[nlats, nlons]) ;   lons=np.reshape(lons,[nlats, nlons])
    sigma=np.reshape(sigma,[nlats, nlons]) ;   energy=np.reshape(energy,[nlats, nlons])
    ug=np.reshape(ug,[nlats, nlons]) ;   vg=np.reshape(vg,[nlats, nlons])
    diss=np.reshape(diss,[nlats, nlons]) ;   accflag=np.reshape(accflag,[nlats, nlons])
    #    print(lats[0,:].std())
    vv=np.sqrt(ug**2+vg**2)
    snapplot(lons, lats, sigma, accflag, energy/sigma, ug/vv.mean()*100., -vg/vv.mean()*100., [2,2], outdir=os.path.dirname(infile))
    
# general framework for a post-processed map of some quantity q
def somemap(lons, lats, q, outname):
    wnan=np.where(np.isnan(q))
    nnan=np.size(wnan)
    nlevs=30
    levs=np.linspace(q.min(), q.max(), nlevs)
    print(outname+" somemap: "+str(nnan)+"NaN points out of "+str(np.size(q)))
    plt.ioff()
    plt.clf()
    fig=plt.figure()
    plt.contourf(lons, lats, q, cmap='hot',levels=levs)
    plt.colorbar()
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    fig.set_size_inches(8, 5)
    plt.savefig(outname)
    plt.close()
#
def someplot(x, qlist, xname='', yname='', prefix='out/', title='', postfix='plot',
             fmt=None, ylog=False):
    '''
    a very general one-dimensional plot for several quantities
    x is an array, qlist is a list of arrays
    fmt are also a list of the colours and markers/linestyles to plot the lines with
    '''
    nq=np.shape(qlist)[0]
    if(fmt == None):
        fmt=np.repeat('k,', nq)
    plt.clf()
    for k in np.arange(nq):
        plt.plot(x, qlist[k], fmt[k])
    if(ylog):
        plt.yscale('log')
    plt.xlabel(xname) ;   plt.ylabel(yname) ; plt.title(title)
    plt.savefig(prefix+postfix+'.eps')
    plt.savefig(prefix+postfix+'.png')
    plt.close()
    
# general 1D-plot of several quantities as functions of time
def sometimes(tar, qlist, fmt=None, prefix='out/', title='', ylog=True):
    someplot(tar, qlist, xname='$t$, ms', prefix=prefix, title=title, postfix='curves', fmt=fmt, ylog=ylog)
    
########################################################################
# post-processing of remotely produced light curves and spectra
def pdsplot(infile="out/pdstots_diss", omega=None):
    lines = np.loadtxt(infile+".dat", comments="#", delimiter=" ", unpack=False)
    freq1=lines[:,0] ; freq2=lines[:,1]
    fc=(freq1+freq2)/2. # center of frequency interval
    f=lines[:,2] ; df=lines[:,3] # replace df with quantiles!
    nf=np.size(f)
    wfin=np.where(np.isfinite(f))
    fmin=f[wfin].min() ; fmax=f[wfin].max()
    plt.clf()
    if(omega != None):
        plt.plot([omega/2./np.pi,omega/2./np.pi], [fmin,fmax], 'b')
        plt.plot([omega/2./np.pi*0.5,omega/2./np.pi*0.5], [fmin,fmax], 'g', linestyle='dotted')
        plt.plot([omega/2./np.pi*1.5,omega/2./np.pi*1.5], [fmin,fmax], 'g', linestyle='dotted')
        plt.plot([2.*omega/2./np.pi,2.*omega/2./np.pi], [fmin,fmax], 'b', linestyle='dotted')
        plt.plot([3.*omega/2./np.pi,3.*omega/2./np.pi], [fmin,fmax], 'b', linestyle='dotted')
        plt.plot([4.*omega/2./np.pi,4.*omega/2./np.pi], [fmin,fmax], 'b', linestyle='dotted')
    for kf in np.arange(nf):
        plt.plot([freq1[kf], freq2[kf]], [f[kf], f[kf]], color='k')
        plt.plot([fc[kf], fc[kf]], [f[kf]-df[kf], f[kf]+df[kf]], color='k')
        
    plt.xlabel(r'$f$, Hz')
    plt.ylabel(r'PDS, relative units')
    plt.xscale('log') ;    plt.yscale('log')
    plt.savefig(infile+'.png')
    plt.savefig(infile+'.eps')
    plt.close()
#
def twopdss(file1, file2):
    '''
    plots a difference between two PDSs with identical frequency grid
    file2 > file1
    '''
    lines = np.loadtxt(file1+".dat", comments="#", delimiter=" ", unpack=False)
    freq1=lines[:,0] ; freq2=lines[:,1]  
    fc=(freq1+freq2)/2. # center of frequency interval
    f1=lines[:,2] ; df1=lines[:,3] ;  nf=np.size(f1) # replace df with quantiles!
    lines = np.loadtxt(file2+".dat", comments="#", delimiter=" ", unpack=False)
    f2=lines[:,2] ; df2=lines[:,3] # replace df with quantiles!
    medrat=np.median((f2/f1)[np.isfinite(f2/f1)])
    print("median ratio "+str(medrat))
    plt.clf()
    for kf in np.arange(nf):
        plt.plot([freq1[kf], freq2[kf]], [f2[kf]-f1[kf]*medrat, f2[kf]-f1[kf]*medrat], color='k')
        plt.plot([freq1[kf], freq2[kf]], [f1[kf], f1[kf]], color='r')
        plt.plot([freq1[kf], freq2[kf]], [f2[kf], f2[kf]], color='g')
        plt.plot([fc[kf], fc[kf]], [f2[kf]-f1[kf]*medrat-df1[kf]*medrat-df2[kf], f2[kf]-f1[kf]*medrat+df1[kf]*medrat+df2[kf]], color='k')
    plt.xlabel(r'$f$, Hz')
    plt.ylabel(r'$\Delta$PDS, relative units')
    plt.xscale('log') ;    plt.yscale('log')
    plt.savefig(file2+'_diff.png')
    plt.savefig(file2+'_diff.eps')
    plt.close()
    
#
def dynsplot(infile="out/pds_diss", omega=None):
    '''
    plots dynamical spectrum using timing.py ascii output 
    '''
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
    fig=plt.figure()
    #  plt.contourf(t, fc, f2, cmap='hot')
    plt.pcolor(t2, binfreq2, np.log(f2ma), cmap='hot', vmin=np.log(pmin), vmax=np.log(pmax)) # tcenter2, binfreq2 should be corners
    # plt.contourf(tc, fc, np.log(f2), cmap='hot')
    #    plt.colorbar()
    #    plt.plot([t.min(), t.min()],[omega/2./np.pi,omega/2./np.pi], 'r')
    #    plt.plot([t.min(), t.max()],[2.*omega/2./np.pi,2.*omega/2./np.pi], 'r')
    if(omega != None):
        plt.plot([t2.min(), t2.max()],[omega/2./np.pi,omega/2./np.pi], 'w')
        plt.plot([t2.min(), t2.max()],[2.*omega/2./np.pi,2.*omega/2./np.pi], 'w',linestyle='dotted')
    plt.ylim(freq2.min(), freq2.max()/2.)
    plt.yscale('log')
    plt.ylabel('$f$, Hz', fontsize=20)
    plt.xlabel('$t$, s', fontsize=20)
    plt.tick_params(labelsize=18, length=3, width=1., which='minor')
    plt.tick_params(labelsize=18, length=6, width=2., which='major')
    fig.set_size_inches(8, 4)
    fig.tight_layout()
    plt.savefig(infile+'.png')
    plt.savefig(infile+'.eps')
    plt.close()

# makes the t-phi and t-th plots for a selected pair of quantities
def timangle(tar, lats, lons, qth, qphi, prefix='out/',omega=None):
    '''
    plots time+theta and time+phi 2D maps for quantity qth,qphi (first is a function of theta, second depends on phi)
    '''
    # input angle mesh may be 2d or 1d:
    slats=np.shape(lats)
    if(np.size(slats)>1):
        latsmean=lats.mean(axis=1)
    else:
        latsmean=lats
    slons=np.shape(lons)    
    if(np.size(slons)>1):
        lonsmean=lons.mean(axis=0)
    else:
        lonsmean=lons
    plt.clf()
    plt.contourf(tar, latsmean*180./np.pi-90., qth, levels=np.linspace(qth.min(), qth.max(), 30), cmap='hot')
    plt.colorbar()
    plt.xlabel('$t$, ms')
    plt.ylabel('latitude')
    plt.savefig(prefix+'_tth.eps')
    plt.savefig(prefix+'_tth.png')
    plt.clf()
    fig=plt.figure()
    plt.contourf(tar, lonsmean*180./np.pi, qphi, levels=np.linspace(qphi.min(), qphi.max(), 30),cmap='hot')
    if(omega != None):
        plt.plot(tar, (omega*tar % (2.*np.pi))*180./np.pi, color='k')
    #    plt.colorbar()
    plt.ylim(0.,360.)
    plt.xlabel('$t$, ms', fontsize=20)
    plt.ylabel('longitude', fontsize=20)
    plt.tick_params(labelsize=18, length=3, width=1., which='minor')
    plt.tick_params(labelsize=18, length=6, width=2., which='major')
    fig.set_size_inches(8, 4)
    fig.tight_layout()
    plt.savefig(prefix+'_tphi.eps')
    plt.savefig(prefix+'_tphi.png')
        
# a wrapper for timangle
def plot_timangle(prefix='out/'):
    '''
    plot for a timangle output
    '''
    lines1 = np.loadtxt(prefix+"tth.dat", comments="#", delimiter=" ", unpack=False)
    t=lines1[:,0] ;  lats=lines1[:,1] ; flats=lines1[:,2]
    lines2 = np.loadtxt(prefix+"tphi.dat", comments="#", delimiter=" ", unpack=False)
    lons=lines2[:,1] ; flons=lines2[:,2]
    #    outdir=os.path.dirname(prefix)
    t=np.unique(t)  ; ulons=np.unique(lons) ; ulats=np.unique(lats)
    flats=np.reshape(flats, [np.size(t), np.size(ulats)])
    lats=np.reshape(lats, [np.size(t), np.size(ulats)])
    flons=np.reshape(flons, [np.size(t), np.size(ulons)])
    lons=np.reshape(lons, [np.size(t), np.size(ulons)])
    #    print(lons[:,0])
    timangle(t, ulats, ulons, np.transpose(flats), np.transpose(flons), prefix=prefix+'plots')

# vdKlis's plot: frequency as a function of flux
def FFplot(prefix='out/'):
    '''
    makes a flux-frequency correlation plot 
    '''
    lines_freq = np.loadtxt(prefix+"freqmax.dat", comments="#", delimiter=" ", unpack=False)
    time=lines_freq[:,0] ;  freq=lines_freq[:,1] ;  dfreq=lines_freq[:,2]
    lines_flux = np.loadtxt(prefix+"binflux.dat", comments="#", delimiter=" ", unpack=False)
    flux=lines_flux[:,1] ;  dflux=lines_flux[:,2]

    plt.clf()
    plt.errorbar(flux, freq, xerr=dflux, yerr=dfreq, fmt='.k')
    plt.xscale('log')  ;   plt.yscale('log')
    plt.xlabel(r'$L_{\rm obs}$, $10^{37}$erg\,s$^{-1}$')
    plt.ylabel(r'$f_{\rm peak}$, Hz')
    plt.savefig(prefix+'FFplot.eps')
    plt.savefig(prefix+'FFplot.png')
    plt.close()

# plot of a saved ascii map produced by plotnth
def plot_saved(infile):
    '''
    plots maps from a saved, aliased ascii map produced by plotnth (infile)
    '''
    
    lines = np.loadtxt(infile, comments="#", delimiter=" ", unpack=False)
    lats = lines[:,0] ; lons = lines[:,1] ; sig = lines[:,2]; qminus = lines[:,8]
    divg = lines[:,3] ; vortg = lines[:,4]
    
    ulons=np.unique(lons) ; ulats=np.unique(lats)
    lons=np.reshape(lons, [np.size(ulats), np.size(ulons)])
    lats=np.reshape(lats, [np.size(ulats), np.size(ulons)])
    sig=np.reshape(sig, [np.size(ulats), np.size(ulons)])
    qminus=np.reshape(qminus, [np.size(ulats), np.size(ulons)])
    vortg=np.reshape(vortg, [np.size(ulats), np.size(ulons)])

    somemap(lons, lats, np.log(sig), infile+"_sig.png")
    somemap(lons, lats, qminus, infile+"_qminus.png")
    somemap(lons, lats, vortg, infile+"_vort.png")
    
def multiplot_saved(prefix, skip=0):

    flist0 = np.sort(glob.glob(prefix+"[0-9].dat"))
    flist1 = np.sort(glob.glob(prefix+"[0-9][0-9].dat"))
    flist2 = np.sort(glob.glob(prefix+"[0-9][0-9][0-9].dat"))
    flist3 = np.sort(glob.glob(prefix+"[0-9][0-9][0-9][0-9].dat"))
    flist=np.concatenate((flist0, flist1, flist2, flist3))
    print(flist)
    nlist = np.size(flist)
    outdir=os.path.dirname(prefix)
    
    for k in np.arange(nlist-skip)+skip:
        plot_saved(flist[k])
        print(outdir+'/sig{:05d}'.format(k)+".png")
        os.system("mv "+flist[k]+"_sig.png"+" "+outdir+'/sig{:05d}'.format(k)+".png")
        os.system("mv "+flist[k]+"_qminus.png"+" "+outdir+'/q{:05d}'.format(k)+".png")
        os.system("mv "+flist[k]+"_vort.png"+" "+outdir+'/v{:05d}'.format(k)+".png")

        
# multiplot_saved('titania/out_twist/runcombine.hdf5_map', skip=7792)
# multiplot_saved('titania/out_NA/runcombine.hdf5_map', skip=0)
# multiplot_saved('titania/out512/run.hdf5_map', skip=0)
