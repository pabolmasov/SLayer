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

# main real-time visualization routine:
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
        lonsDeg = 180./np.pi*lons-180.
        latsDeg = 180./np.pi*lats

    nlons=np.size(lons) ; nlats=np.size(lats)
    lats1d=np.unique(lats)
    clats=np.sin(lats1d)
    print("visualize: accreted fraction from "+str(accflag.min())+" to "+str(accflag.max()))
    
    vorm=np.fabs(vortg-2.*cf.omega*np.sin(lats)).max()

    mdot=cf.sigplus * 4. * np.pi * np.sin(cf.latspread) * cf.rsphere**2 *np.sqrt(4.*np.pi)
    if(cf.tturnon>0.):
        mdot*=(1.-np.exp(-t/cf.tturnon))
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
    # internal temperature
    tbottom=50.59*((1.-beta)*energy*cf.sigmascale/cf.mass1)**0.25
    visualizeMap(axs[1], 
                 lonsDeg, latsDeg, 
                 tbottom, 
                 tbottom.min(), tbottom.max(), 
                 title=r'$T_{\rm bottom}$, keV')
    
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
                 title=r'tracer')
#    axs[6].plot([(np.pi/2.-np.arctan(angmoy/vangmo))*180./np.pi], [np.arcsin(angmoz/angmox)*180./np.pi], 'or')
    #Q^-
    teff=(qminus*cf.sigmascale/cf.mass1)**0.25*3.64 # effective temperature in keV
    visualizeMap(axs[7], 
                 lonsDeg, latsDeg, 
                 teff, 
                 teff.min(), teff.max(),  
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
             ,latrange=None, lonrange=None):
    # longitudes, latitudes, density field, accretion flag, velocity fields, alias for velocity output
    if((latrange == None) | (lonrange == None)):
        skx=sks[0] ; sky=sks[1]
    else:
        skx=2 ; sky=2

    wpoles=np.where(np.fabs(lats)<90.)
    s0=tb[wpoles].min() ; s1=tb[wpoles].max()
    #    s0=0.1 ; s1=10. # how to make a smooth estimate?
    nlev=30
    levs=(s1-s0)*((np.arange(nlev)-0.5)/np.double(nlev-1))+s0
    interactive(False)

    plt.clf()
    fig=plt.figure()
    plt.pcolormesh(lons, lats, tb, cmap='plasma') # ,levels=levs)
    plt.colorbar()
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
    if((latrange == None) & (lonrange == None)):
        plt.ylim(-85.,85.)
    else:
        plt.ylim(latrange[0], latrange[1])
        plt.xlim(lonrange[0], lonrange[1])
    plt.xlabel('longitude')
    plt.ylabel('latitude')
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
    ax.pcolormesh(lons*np.pi/180.*(tinyover+1.), theta, tb,cmap='plasma') #,levels=levs)
#    if(accflag.max()>1e-3):
#        ax.contour(lons*np.pi/180.*(tinyover+1.), theta, accflag,colors='w',levels=[0.5])
    ax.set_rticks([30., 60.])
    ax.set_rmax(90.)
    plt.title('  N') #, t='+str(nstep))
    plt.tight_layout()
    fig.set_size_inches(4, 4)
    plt.savefig(outdir+'/northpole.eps')
    plt.savefig(outdir+'/northpole.png')
    plt.close()
    plt.clf()
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    #    wnorth=np.where(lats>0.)
#    tinyover=0./np.double(nlons)
    ax.contourf(lons*np.pi/180.*(tinyover+1.), 180.*(tinyover+1.)-theta, tb,cmap='plasma' ,levels=levs)
#    if(accflag.max()>1e-3):
#        ax.contour(lons*np.pi/180.*(tinyover+1.), 180.*(1.+tinyover)-theta, accflag,colors='w',levels=[0.5])
    ax.set_rticks([30., 60.])
    ax.set_rmax(90.)
    plt.tight_layout(pad=2)
    fig.set_size_inches(4, 4)
    plt.title('  S') #, t='+str(nstep))
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
    print(outname+" somemap: "+str(nnan)+"NaN points out of "+str(np.size(q)))
    plt.clf()
    fig=plt.figure()
    plt.contourf(lons, lats, q,cmap='plasma') #,levels=levs)
    plt.colorbar()
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    fig.set_size_inches(8, 5)
    plt.savefig(outname)
    plt.close()
# general 1D-plot of several quantities as functions of time
def sometimes(tar, qlist, col=None, linest=None, prefix='out/', title=''):
    nq=np.shape(qlist)[0]
    if(col == None):
        col=np.repeat('k', nq)
    if(linest == None):
        linest=np.repeat('solid', nq)
#    print("sometimes: "+str(np.size(qlist))+", "+str(np.size(col))+", "+str(np.size(linest)))
    plt.clf()
    for k in np.arange(nq):
        plt.plot(tar, qlist[k], color=col[k], linestyle=linest[k])
    plt.yscale('log')
    plt.xlabel('$t$, s')
    plt.ylabel(title)
    plt.savefig(prefix+'curves.eps')
    plt.savefig(prefix+'curves.png')
    plt.close()
    
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
    #  plt.contourf(t, fc, f2, cmap='plasma')
    plt.pcolor(t2, binfreq2, np.log(f2ma), cmap='plasma', vmin=np.log(pmin), vmax=np.log(pmax)) # tcenter2, binfreq2 should be corners
    # plt.contourf(tc, fc, np.log(f2), cmap='plasma')
    #    plt.colorbar()
    #    plt.plot([t.min(), t.min()],[omega/2./np.pi,omega/2./np.pi], 'r')
    #    plt.plot([t.min(), t.max()],[2.*omega/2./np.pi,2.*omega/2./np.pi], 'r')
    if(omega != None):
        plt.plot([t2.min(), t2.max()],[omega/2./np.pi,omega/2./np.pi], 'w')
        plt.plot([t2.min(), t2.max()],[2.*omega/2./np.pi,2.*omega/2./np.pi], 'w',linestyle='dotted')
    plt.ylim(freq2.min(), freq2.max()/2.)
    plt.yscale('log')
    plt.ylabel('$f$, Hz')
    plt.xlabel('$t$, s')
    plt.savefig(infile+'.png')
    plt.savefig(infile+'.eps')
    plt.close()

#
def timangle(tar, lats, lons, qth, qphi, prefix='out/',omega=None):
    '''
    plots time+theta and time+phi 2D maps for quantity qth,qphi (first is a function of theta, second depends on phi)
    '''
    latsmean=lats.mean(axis=1)
    lonsmean=lons.mean(axis=0)
    plt.clf()
    plt.contourf(tar, latsmean*180./np.pi-90., qth, levels=np.linspace(qth.min(), qth.max(), 30))
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('latitude')
    plt.savefig(prefix+'_tth.eps')
    plt.savefig(prefix+'_tth.png')
    plt.clf()
    fig=plt.figure()
    plt.contourf(tar, lonsmean*180./np.pi, qphi, levels=np.linspace(qphi.min(), qphi.max(), 30))
    if(omega != None):
        plt.plot(tar, (omega*tar % (2.*np.pi))*180./np.pi, color='k')
    plt.colorbar()
    plt.ylim(0.,360.)
    plt.xlabel('$t$')
    plt.ylabel('longitude')
    fig.set_size_inches(10, 4)
    plt.savefig(prefix+'_tphi.eps')
    plt.savefig(prefix+'_tphi.png')
        
