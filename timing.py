import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import time
import pylab
import h5py

#proper LaTeX support and decent fonts in figures 
rc('font',**{'family':'serif','serif':['Times']})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

# calculates the light curve and the power density spectrum
# it's much cheaper to read the datafile once and compute multiple data points
def fluxest(filename, lat0, lon0):
    # file name, the polar angle and the azimuth of the viewpoint 
    f = h5py.File(filename,'r')
    params=f["params"]
    nlons=params.attrs["nlons"] ; nlats=params.attrs["nlats"] ; omega=params.attrs["omega"] ; rsphere=params.attrs["rsphere"] ; tscale=params.attrs["tscale"] 
    lons1d = (2.*np.pi/nlons)*np.arange(nlons)
    clats1d = 2.*np.arange(nlats)/np.double(nlats)-1.
    lons,lats = np.meshgrid(lons1d, np.arccos(clats1d))
    dlons=2.*np.pi/np.size(lons1d) ; dlats=2./np.double(nlats)
    cosa=np.cos(lats)*np.cos(lat0)+np.sin(lats)*np.sin(lat0)*np.cos(lons-lon0)
    cosa=(cosa+np.fabs(cosa))/2. # only positive viewing angle

    keys=f.keys()
    nsize=np.size(keys)-1 # last key contains parameters
    flux=np.zeros(nsize)  ;  mass=np.zeros(nsize) ;  tar=np.zeros(nsize)
    flc=open('out/lcurve.dat', 'w')
    for k in np.arange(nsize):
        data=f[keys[k]]
        sig=data["sig"][:] ; diss=data["diss"][:]
        tar[k]=data.attrs["t"]
        flux[k]=(diss*cosa).sum()*dlons*dlats
        mass[k]=(sig*cosa).sum()*dlons*dlats
        flc.write(str(tar[k])+' '+str(flux[k])+' '+str(mass[k])+"\n")
        print str(tar[k])+' '+str(flux[k])+' '+str(mass[k])+"\n"
    f.close() ; flc.close() 
    tar*=tscale

    # linear trend:
    m,b =  np.polyfit(tar, mass, 1) # for "visible mass"
    md,bd =  np.polyfit(tar, flux, 1) # for dissipation
    
    plt.clf()
    plt.plot(tar, (flux-flux.min())/(flux.max()-flux.min()), color='k')
    plt.plot(tar, (mass-mass.min())/(mass.max()-mass.min()), color='r')
    plt.plot(tar, (tar*m+b-mass.min())/(mass.max()-mass.min()), color='g')
    plt.plot(tar, (tar*md+bd-flux.min())/(flux.max()-flux.min()), color='b')
    plt.xlabel('$t$')
    plt.ylabel('flux, relative units')
    plt.savefig('out/lcurve.eps')

    flux-=md*tar+bd ; mass-=m*tar+b # subtraction of linear trends
    tmean=tar.mean() ;     tspan=tar.max()-tar.min()

    fsp=np.fft.rfft(flux/flux.std()) ;   fspm=np.fft.rfft(mass/mass.std())
 #   fsp=np.fft.fftshift(fsp) ;    fspm=np.fft.fftshift(fspm)
    pds=np.abs(fsp)**2  ;   pdsm=np.abs(fspm)**2
    freq = np.fft.rfftfreq(nsize, tspan/np.double(nsize)) # frequency grid
    #  a good idea is also to make a binning
    omegadisk=2.*np.pi/rsphere**1.5*0.9/tscale
    omega/=tscale
    wpos=np.where((pds*pdsm)>0.)
    print omega, omegadisk
    plt.clf()
    plt.plot(freq[wpos], pds[wpos], 'k')
    plt.plot(freq[wpos], pdsm[wpos], 'b')
    plt.plot([omega/2./np.pi,omega/2./np.pi], [pds[wpos].min(),pds[wpos].max()], 'r')
    plt.plot([omegadisk/2./np.pi,omegadisk/2./np.pi], [pds[wpos].min(),pds[wpos].max()], 'g')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1./tspan, np.double(nsize)/tspan)
    plt.xlabel('$|f|$, s$^{-1}$')
    plt.ylabel('PDS, relative units')
    plt.savefig('out/PDS.eps')
    
    fpds=open('out/pds.dat', 'w')
    for k in np.arange(np.size(freq)):
        fpds.write(str(freq[k])+' '+str(pds[k])+"\n")
    fpds.close()
    
