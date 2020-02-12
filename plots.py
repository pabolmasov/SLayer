from __future__ import print_function
from __future__ import division
# module for all the visualization tools & functions

from builtins import str
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
from matplotlib import interactive, use
from matplotlib.colors import BoundaryNorm

import glob

plt.ioff()
use('Agg')

##################################################

def visualizePoles(ax, angmo):
    # axes and angular momentum components (3-tuple)
    x,y,z=angmo
    polelon = np.arctan2(y,x)
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
    ax.set_xlim(-180,180) ; ax.set_ylim(-90,90)
    plt.colorbar(pc, ax=ax)
    # ax.axis('equal')


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
    sigma = [(sk/2.), (sk/2.)]
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
    ax.set_xlim(-180,180) ; ax.set_ylim(-90,90)

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
    kenergy=trapz((sig*(ug**2+vg**2)).mean(axis=1), x=clats)/2.
#    (sig*engy).sum()*4.*np.pi/np.double(nlons*nlats)*rsphere**2
    angmoz=trapz((sig*ug*np.cos(lats)).mean(axis=1), x=clats)*cf.rsphere
    angmox=trapz((sig*(vg*np.sin(lons)-ug*np.sin(lats)*np.cos(lons))).mean(axis=1), x=clats)*cf.rsphere
    angmoy=trapz((sig*(vg*np.cos(lons)-ug*np.sin(lats)*np.sin(lons))).mean(axis=1), x=clats)*cf.rsphere
    vangmo=np.sqrt(angmox**2+angmoy**2+angmoz**2) # total angular momentum 

    print("t = "+str(t))
    print("simulation running with ktrunc = "+str(cf.ktrunc)+", ktrunc_diss = "+str(cf.ktrunc_diss)+", Ndiss = "+str(cf.ndiss))

    cspos = ((press/sig)+np.fabs((press/sig)))/2.
    
    #vorticity
    visualizeMap(axs[0], 
                 lonsDeg, latsDeg, 
                 vortg-2.*cf.omega*np.sin(lats), 
                 -vorm*1.1, vorm*1.1, 
                 title="$\Delta \omega$")
    # internal temperature
    tbottom=339.6*((1.-beta)*sig*(cf.sigmascale/1e8)/cf.mass1/cf.rsphere**2)**0.25
    # 50.59*((1.-beta)*energy*cf.sigmascale/cf.mass1)**0.25
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
    dunorm = (du/vabs)
    dvnorm = (dv/vabs)

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

    wpoles=np.where(np.fabs(lats)>30.)
    s0=tb.min() ; s1=tb.max()
    #    s0=0.1 ; s1=10. # how to make a smooth estimate?
    nlev=30
    levs=(s1-s0)*((np.arange(nlev)-0.5)/np.double(nlev-1))+s0
#    levs=np.unique(np.round(levs, 2))
    interactive(False)

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
    plt.xlim(0., 360.)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    if t != None:
        plt.title('$t={:8.3f}$ms'.format(t), loc='left')
    fig.set_size_inches(8, 5)
    plt.savefig(outdir+'/snapshot.png')
    plt.savefig(outdir+'/snapshot.eps')
    plt.close()
    # drawing poles:
    somepoles(lons*np.pi/180., lats*np.pi/180., tb, outdir+'/polemap', t=t)
   
###########################################################################
# post-factum visualizations from the ascii output of snapplot:
#    
# general framework for a post-processed map of some quantity q
def somemap(lons, lats, q, outname, latrange = None, title = None):
    wnan=np.where(np.isnan(q))
    nnan=np.size(wnan)
    nlevs=30
    dl = int(np.ceil(np.log10(abs(q).max())))
    levs=np.round(np.linspace(q.min(), q.max(), nlevs), 2-dl)
    cmap = plt.get_cmap('hot')
    norm = BoundaryNorm(levs, ncolors=cmap.N, clip=True)
    print(outname+" somemap: "+str(nnan)+"NaN points out of "+str(np.size(q)))
    #    plt.ioff()
    plt.clf()
    fig=plt.figure()
    plt.pcolormesh(lons*180./np.pi, -lats*180./np.pi, q, cmap=cmap, norm=norm)
    plt.colorbar()
    if(latrange is not None):
        plt.ylim(latrange[0],latrange[1])
    plt.tick_params(labelsize=14, length=3, width=1., which='minor')
    plt.tick_params(labelsize=14, length=6, width=2., which='major')    
    plt.xlabel('longitude, deg',fontsize=16)
    plt.ylabel('latitude, deg', fontsize=16)
    if title is not None:
        plt.title(title)
    fig.set_size_inches(5, 4)
    fig.tight_layout()
    plt.savefig(outname, dpi = 100)
    plt.close()
    #
def somepoles(lons, lats, q, outname, t = None):
    '''
    draws the maps of the polar regions
    '''
    wpoles = np.where(np.fabs(lats)>(np.pi/6.)) ; nlevs=30
    levs = np.linspace(q.min(), q.max(), nlevs)
#    print(levs)
    theta = np.pi/2.-lats
    plt.clf()
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    pc=ax.contourf(lons, theta*180./np.pi, q, cmap='hot', levels=levs)
    plt.colorbar(pc,ax=ax)
    ax.set_rticks([30., 60.])
    ax.grid(color='w')
    ax.set_rmax(70.)
    if t != None:
        plt.text(90., 95.,r'North pole, $t={:8.3f}$ms'.format(t), fontsize=18)
    else:
        plt.title('North pole', loc='left')
    plt.tight_layout(pad=3)
    fig.set_size_inches(5, 4)
#    plt.savefig(outname+'_north.eps')
    plt.savefig(outname+'_north.png')
    plt.close()
    plt.clf()
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    pc=ax.contourf(lons, 180.-theta*180./np.pi, q,cmap='hot', levels=levs) #,color='w')
    plt.colorbar(pc,ax=ax)
    ax.set_rticks([30., 60.])
    ax.grid(color='w')
    ax.set_rmax(70.)
    #    ax.axis('equal') ruins polar scale
    fig.set_size_inches(5, 4)
    if t != None:
        plt.text(90., 95.,r'South pole, $t={:8.3f}$ms'.format(t), fontsize=18)
    else:
        plt.title('South pole', loc='left')
    plt.tight_layout(pad=3)
    fig.set_size_inches(5, 4)
    fig.tight_layout()
    #    plt.savefig(outname+'_south.eps')
    plt.savefig(outname+'_south.png')
    plt.close()
    print(outname+'_north.png ; '+outname+'_south.png')
    
def plot_somemap(infile, nco, latrange = None):
    '''
    plots a map from a lons -- lats -- ... ascii data file
    '''
    lines = np.loadtxt(infile+".dat", comments="#", delimiter=" ", unpack=False)
    lats = lines[:,1] ; lons = lines[:,0]
    q = lines[:,nco]
    ulons = np.unique(lons) ; ulats = np.unique(lats)
    qt = np.transpose(np.reshape(q, [np.size(ulons),np.size(ulats)]))
    somemap(ulons, ulats, qt, infile+"_"+str(nco-2), latrange = latrange)

def crosses(x, dx, y, dy, xlabel='', ylabel='', outfilename = 'crossplot'):
    '''
    plots a two-dimensional plot with error bars
    '''
    plt.clf()
    fig=plt.figure()
    plt.errorbar(x, y, xerr=dx, yerr=dy, fmt='k.')
    plt.xlabel(xlabel, fontsize=20) ; plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(labelsize=18, length=3, width=1., which='minor')
    plt.tick_params(labelsize=18, length=6, width=2., which='major')
    fig.set_size_inches(5, 4)
    fig.tight_layout()
    plt.savefig(outfilename+'.png')
    plt.savefig(outfilename+'.eps')
    plt.close()
    
def someplot(x, qlist, xname='', yname='', prefix='out/', title='', postfix='plot',
             fmt=None, ylog=False, latmode=False):
    '''
    a very general one-dimensional plot for several quantities
    x is an array, qlist is a list of arrays
    fmt are also a list of the colours and markers/linestyles to plot the lines with
    '''
    nq=np.shape(qlist)[0]
    if(fmt == None):
        fmt=np.repeat('k,', nq)
    plt.clf()
    fig, ax = plt.subplots()
    for k in np.arange(nq):
        ax.plot(x, qlist[k], fmt[k])
    if(ylog):
        plt.yscale('log')
    plt.xlabel(xname, fontsize=18) ;   plt.ylabel(yname, fontsize=18) ; plt.title(title)
    ax.tick_params(labelsize=16, length=3, width=1., which='minor')
    ax.tick_params(labelsize=16, length=6, width=2., which='major')
    if(latmode):
        ax.set_xticks([-90,-60,-30, 0, 30, 60, 90])
    fig.tight_layout()
    plt.savefig(prefix+postfix+'.eps')
    plt.savefig(prefix+postfix+'.png')
    plt.close()
    print(prefix+postfix)
    
# general 1D-plot of several quantities as functions of time
def sometimes(tar, qlist, fmt=None, prefix='out/', title='', ylog=True, yname = None):
    someplot(tar, qlist, xname='$t$, ms', prefix=prefix, title=title, postfix='curves',
             fmt=fmt, ylog=ylog, yname=yname)

def plot_sometimes(infile="out/lcurve", ylog=True, tfilter = None, yname = None):
    lines = np.loadtxt(infile+".dat", comments="#", delimiter=" ", unpack=False)
    tar = lines[:,0] ; f = lines[:,1]
    if(tfilter == None):
        someplot(tar, [f], xname='$t$, ms', prefix=infile, postfix='_c', ylog=ylog, yname = yname)
    else:
        wfil = (tar<tfilter[1]) & (tar > tfilter[0])
        someplot(tar[wfil], [f[wfil]], xname='$t$, ms', prefix=infile, postfix='_c', ylog=ylog, fmt = ['k-'], yname = yname)
    
########################################################################
# post-processing of remotely produced light curves and spectra
def pdsplot(infile="out/pdstots_diss", omega=None, freqrange = None):
    lines = np.loadtxt(infile+".dat", comments="#", delimiter=" ", unpack=False)
    freq1=lines[:,0] ; freq2=lines[:,1]
    fc=(freq1+freq2)/2. # center of frequency interval
    f=lines[:,2] ; df=lines[:,3] # replace df with quantiles!
    nf=np.size(f)
    wfin=np.where(np.isfinite(f))
    fmin=f[wfin].min() ; fmax=f[wfin].max()
    plt.clf()
    if(omega is not None):
        so=np.size(omega)
        if(so<=1):
            plt.plot([omega/2./np.pi,omega/2./np.pi], [fmin,fmax], 'b')
        else:
            for ko in np.arange(so):
                plt.plot([omega[ko]/2./np.pi,omega[ko]/2./np.pi], [fmin,fmax], 'b')

    for kf in np.arange(nf):
        plt.plot([freq1[kf], freq2[kf]], [f[kf], f[kf]], color='k')
        plt.plot([fc[kf], fc[kf]], [f[kf]-df[kf], f[kf]+df[kf]], color='k')

    if(freqrange is not None):
        plt.xlim(freqrange[0], freqrange[1])
    plt.xlabel(r'$f$, Hz')
    plt.ylabel(r'fractional PDS')
    # plt.xscale('log')
    plt.yscale('log')
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
def dynsplot(infile="out/pds_diss", omega=None, binnorm=True):
    '''
    plots dynamical spectrum using timing.py ascii output 
    '''
    lines = np.loadtxt(infile+".dat", comments="#", delimiter=" ", unpack=False)
    freq1=lines[:,2] ; freq2=lines[:,3] ;  tar1=lines[:,0] ;  tar2=lines[:,1] 
    fc = np.copy(freq1+freq2)/2. # center of frequency interval
    tc = np.copy(tar1+tar2)/2.
    f=lines[:,4] ; df=lines[:,5] # replace df with quantiles!
    tun=np.unique(tar1) ; fun=np.unique(freq1)
    ntimes=np.size(tun) ;    nbins=np.size(fun)
  #  print("ntimes = "+str(ntimes))
  #  print("nbins = "+str(nbins))
    t2=np.zeros([ntimes+1, nbins+1], dtype=np.double)
    binfreq2=np.zeros([ntimes+1, nbins+1], dtype=np.double)
    f2=np.transpose(np.reshape(f, [nbins, ntimes]))
    df2=np.transpose(np.reshape(df, [nbins, ntimes]))
    tc=np.transpose(np.reshape(tc, [nbins, ntimes]))
    fc=np.transpose(np.reshape(fc, [nbins, ntimes]))
    for kt in np.arange(ntimes+1):
        if(kt<ntimes):
            t2[kt,:-1]= tun[kt] #tun[kt]-(tun[kt+1]-tun[kt])/2.
        else:
            t2[kt,:-1] = tar2.max()
        binfreq2[kt,:-1]=fun[:]
    t2[:-1,-1]=tun[:] ; t2[-1,-1]=tar2.max()
    f2ma=ma.masked_invalid(f2)
    if(binnorm):
        f2tot=f2ma.sum(axis=1)
        for kt in np.arange(ntimes):
            f2ma[kt,:]/=f2tot[kt]
    binfreq2[ntimes-1,:-1]=fun[:] ;  binfreq2[ntimes,:-1]=fun[:]
    binfreq2[:,-1]=freq2.max()
    w=np.isfinite(df2)&(df2>0.)
    p = np.log10(f2ma*fc**2)
    pmin=(f2ma*fc**2).min() ; pmax=(f2ma*fc**2).max()
    #    print(binfreq2.min(),binfreq2.max())
    print("T = "+str(t2.min())+" "+str(t2.max()))
    #    ii=input('T')
    #    print(f2)
    cmap = plt.get_cmap('hot')
    nlevs = 20
    levs = np.round(np.linspace(p.min(), p.max(), nlevs), 2)
    norm = BoundaryNorm(levs, ncolors=cmap.N, clip=True)
    
    plt.clf()
    fig=plt.figure()
    #    plt.contourf(tc, fc, f2, cmap='hot', nlevels=100)
    plt.pcolormesh(t2, binfreq2, p, cmap='hot', norm=norm) 
    # plt.pcolor(t2, binfreq2, p) #, vmin=np.log(pmin), vmax=np.log(pmax)) # t2, binfreq2 should be corners
    # plt.contourf(tc, fc, np.log(f2), cmap='hot')
    plt.colorbar()
    #    plt.plot([t.min(), t.min()],[omega/2./np.pi,omega/2./np.pi], 'r')
    #    plt.plot([t.min(), t.max()],[2.*omega/2./np.pi,2.*omega/2./np.pi], 'r')
    if(omega is not None):
        so=np.size(omega)
        if(so <= 1):
            plt.plot([t2.min(), t2.max()],[omega/2./np.pi,omega/2./np.pi], 'w')
        else:
           for ko in np.arange(so):
               plt.plot([t2.min(), t2.max()],[omega[ko]/2./np.pi,omega[ko]/2./np.pi], 'w')
        #  plt.plot([t2.min(), t2.max()],[2.*omega/2./np.pi,2.*omega/2./np.pi], 'w',linestyle='dotted')
    plt.ylim(freq2.min(), freq2.max()/2.)
    plt.xlim(t2.min(), t2.max())
    plt.yscale('log')
    plt.ylabel('$f$, Hz', fontsize=20)
    plt.xlabel('$t$, s', fontsize=20)
    plt.tick_params(labelsize=18, length=3, width=1., which='minor')
    plt.tick_params(labelsize=18, length=6, width=2., which='major')
    fig.set_size_inches(9, 4)
    fig.tight_layout()
    plt.savefig(infile+'.png')
    plt.savefig(infile+'.eps')
    plt.close()

# makes the t-phi and t-th plots for a selected pair of quantities
def timangle(tar, lats, lons, qth, qphi, prefix='out/',omega=None, nolon=False):
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
    fig=plt.figure()
    plt.contourf(tar, latsmean*180./np.pi-90., qth, levels=np.linspace(qth.min(), qth.max(), 30), cmap='hot')
    plt.colorbar()
    plt.xlabel('$t$, ms')
    plt.ylabel('latitude, deg')
    fig.set_size_inches(8, 4)
    fig.tight_layout()
    plt.savefig(prefix+'_tth.eps')
    plt.savefig(prefix+'_tth.png')
    plt.close()
    if not(nolon):
        plt.clf()
        fig=plt.figure()
        plt.contourf(tar, lonsmean*180./np.pi, qphi, levels=np.linspace(qphi.min(), qphi.max(), 30),cmap='hot')
        if(omega != None):
            tnorm=omega*tar
            for k in np.arange(np.floor(tnorm.max())):
                plt.plot(tar, tnorm*180./np.pi-360.*k, color='k')
        #    plt.colorbar()
        plt.ylim(0.,360.)
        plt.xlabel('$t$, ms', fontsize=20)
        plt.ylabel('longitude, deg', fontsize=20)
        plt.tick_params(labelsize=18, length=3, width=1., which='minor')
        plt.tick_params(labelsize=18, length=6, width=2., which='major')
        fig.set_size_inches(8, 4)
        fig.tight_layout()
        plt.savefig(prefix+'_tphi.eps')
        plt.savefig(prefix+'_tphi.png')
        plt.close()
    # extreme values:
    nt = np.size(tar)
    fext = open(prefix+"_tthmm.dat", 'w')
    for kt in np.arange(nt):
        fext.write(str(tar[kt])+" "+str(qth[:,kt].min())+" "+str(qth[:,kt].max())+"\n")
    fext.close()
        
# a wrapper for timangle
def plot_timangle(prefix='out/', trange = None, nolon = False):
    '''
    plot for a timangle output
    '''
    lines1 = np.loadtxt(prefix+"tth.dat", comments="#", delimiter=" ", unpack=False)
    t1=lines1[:,0] ;  lats=lines1[:,1] ; flats=lines1[:,2]
    if(not nolon):
        lines2 = np.loadtxt(prefix+"tphi.dat", comments="#", delimiter=" ", unpack=False)
        t2=lines2[:,0] ; lons=lines2[:,1] ; flons=lines2[:,2]
    if(trange is not None):
        wt1 = np.where((t1 > trange[0]) & (t1<trange[1]))
        t1=t1[wt1] ; lats=lats[wt1] ; flats=flats[wt1]
        if(not nolon):
            wt2 = np.where((t2 > trange[0]) & (t2<trange[1]))
            flons=flons[wt2] ; lons=lons[wt2]
    #    outdir=os.path.dirname(prefix)
    t=np.unique(t1)  ; ulats=np.unique(lats)
    flats=np.reshape(flats, [np.size(t), np.size(ulats)])
    lats=np.reshape(lats, [np.size(t), np.size(ulats)])
    if (not nolon):
        ulons=np.unique(lons)
        flons=np.reshape(flons, [np.size(t), np.size(ulons)])
        lons=np.reshape(lons, [np.size(t), np.size(ulons)])
    else:
        ulons = ulats ; flons = flats
    #    print(lons[:,0])
    timangle(t*1e3, ulats, ulons, np.transpose(flats), np.transpose(flons), prefix=prefix+'plots', nolon=nolon)

# vdKlis's plot: frequency as a function of flux
def FFplot(prefix='out/'):
    '''
    makes a flux-frequency correlation plot 
    '''
    lines_freq = np.loadtxt(prefix+"_ffreq.dat", comments="#", delimiter=" ", unpack=False)
    time=lines_freq[:,0] ; flux = lines_freq[:,1] ; dflux = lines_freq[:,2]
    freq=lines_freq[:,3] ;  dfreq=lines_freq[:,4]
    #     lines_flux = np.loadtxt(prefix+"binflux.dat", comments="#", delimiter=" ", unpack=False)
    #    flux=lines_flux[:,1] ;  dflux=lines_flux[:,2]

    plt.clf()
    plt.errorbar(flux, freq, xerr=dflux, yerr=dfreq, fmt='.k')
    plt.xscale('log')  ;   plt.yscale('log')
    plt.xlabel(r'$L_{\rm obs}$, $10^{37}$erg\,s$^{-1}$')
    plt.ylabel(r'$f_{\rm peak}$, Hz')
    plt.savefig(prefix+'FFplot.eps')
    plt.savefig(prefix+'FFplot.png')
    plt.close()

# plot of a saved ascii map produced by plotnth
def plot_saved(infile, latrange = None):
    '''
    plots maps from a saved, aliased ascii map produced by plotnth (infile)
    '''
    
    lines = np.loadtxt(infile, comments="#", delimiter=" ", unpack=False)
    lats = lines[:,0] ; lons = lines[:,1] ; sig = lines[:,2]; qminus = lines[:,8]
    divg = lines[:,3] ; vortg = lines[:,4]
    
    ulons=np.unique(lons) ; ulats=np.unique(lats)
    lons=np.reshape(lons, [np.size(ulats), np.size(ulons)])
    lats=np.reshape(lats, [np.size(ulats), np.size(ulons)])
    vortg=np.reshape(vortg, [np.size(ulats), np.size(ulons)])
    sig=np.reshape(sig, [np.size(ulats), np.size(ulons)])
    qminus=np.reshape(qminus, [np.size(ulats), np.size(ulons)])

    # somwhow the plot gets overturned
    lats = -lats

    print(np.shape(vortg))
    print(np.shape(sig))
    # sigma is initially in sigmascales
    somemap(lons, lats, vortg, infile+"_vort.png", latrange = latrange,
            title = r'$\omega$, ${\rm s}^{-1}$')
    somemap(lons, lats, qminus, infile+"_qminus.png", latrange = latrange,
            title = r'$\varkappa Q^{-} / cg$')
    somemap(lons, lats, sig, infile+"_sig.png", latrange = latrange,
            title = r'$\Sigma$, g\,cm$^{-2}$')
    if(qminus.max()>qminus.min()):
        somepoles(lons, lats, qminus, infile)
    
def multiplot_saved(prefix, skip=0, step=1):

    flist0 = np.sort(glob.glob(prefix+"[0-9].dat"))
    flist1 = np.sort(glob.glob(prefix+"[0-9][0-9].dat"))
    flist2 = np.sort(glob.glob(prefix+"[0-9][0-9][0-9].dat"))
    flist3 = np.sort(glob.glob(prefix+"[0-9][0-9][0-9][0-9].dat"))
    flist4 = np.sort(glob.glob(prefix+"[0-9][0-9][0-9][0-9][0-9].dat"))
    flist = np.concatenate((flist0, flist1, flist2, flist3, flist4))
    print(flist)
    #    ff=input('f')
    nlist = np.size(flist)
    outdir=os.path.dirname(prefix)
    
    for k in np.arange(skip, nlist, step):
        plot_saved(flist[k])
        print(outdir+'/sig{:05d}'.format(k)+".png")
        os.system("mv "+flist[k]+"_sig.png"+" "+outdir+'/sig{:05d}'.format(k)+".png")
        os.system("mv "+flist[k]+"_qminus.png"+" "+outdir+'/q{:05d}'.format(k)+".png")
        os.system("mv "+flist[k]+"_vort.png"+" "+outdir+'/v{:05d}'.format(k)+".png")
        os.system("mv "+flist[k]+"_north.png"+" "+outdir+'/n{:05d}'.format(k)+".png")
        os.system("mv "+flist[k]+"_south.png"+" "+outdir+'/s{:05d}'.format(k)+".png")

#
def plot_meanmap(infile = "out/meanmap_phavg"):
    '''
    designed for meanmap_phavg
    draws Reynolds's stress and other quantities averaged over multiple frames
    '''
    lines = np.loadtxt(infile+".dat", comments="#", delimiter=" ", unpack=False)
    lats = lines[:,0] ; sig = lines[:,1]; energy = lines[:,2]
    ug = lines[:,3] ; vg = lines[:,4] ;  csq = lines[:,5]
    cuv = lines[:,6]; dcuv = lines[:,7]; aniso = lines[:,8]
    
    someplot(90.-lats*180./np.pi, [cuv, -cuv, ug*vg, -ug*vg,  csq, abs(cuv)+dcuv, abs(cuv)-dcuv], xname=r'latitude, deg', yname=r'$\langle\Delta u \Delta v\rangle$', prefix=infile, title='', postfix='plot', fmt=['k-', 'k--', 'b-', 'b--', 'r:', 'k:', 'k:'], ylog=True, latmode=True)

    
