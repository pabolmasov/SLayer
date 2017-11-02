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
    nsize=size(keys)
    flux=np.zeros(nsize)  ;  mass=np.zeros(nsize) ;  tar=np.zeros(nsize)
    flc=open('out/lcurve.dat', 'w')
    for k in np.arange(nsize):
        data=f["cycle_"+str(nsteps[k]).rjust(6, '0')]
        sig=data["sig"][:] ; sig=data["diss"][:]
        tar[k]=data.attrs["t"]
        fluxtmp=(diss*cosa).sum()*dlons*dlats
        mass[k]=(sig*cosa).sum()*dlons*dlats
        flux[k]=fluxtmp/mass[k]
        flc.write(str(tar[k])+' '+str(flux[k])+' '+str(mass[k])+"\n")
        print str(tar[k])+' '+str(flux[k])+' '+str(mass[k])+"\n"
    f.close() ; flc.close() 

    tar*=tscale

    plt.clf()
    plt.plot(tar, flux, color='k')
    plt.plot(tar, mass, color='r')
    plt.xlabel('$t$')
    plt.ylabel('flux, relative units')
    plt.savefig('lcurve.eps')

    tmean=tar.mean() ;     tspan=tar.max()-tar.min()

    fsp=np.fft.fft(flux/flux.mean()-1.)
    fsp=np.fft.fftshift(fsp)
    pds=np.abs(fsp)**2
    freq = np.fft.fftfreq(nsize, tspan/np.double(nsize)) # frequency grid
    #  a good idea is also to make a binning

    omegadisk=2.*np.pi/rsphere**1.5*0.9
    
    plt.clf()
    plt.plot(np.fabs(freq), pds, ',k')
    plt.plot([omega/2./pi,omega/2./pi], [pds.min(), pds.max()], ',r')
    plt.plot([omegadisk/2./pi,omegadisk/2./pi], [pds.min(), pds.max()], ',g')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1./tspan, np.double(tspan)/tspan)
    plt.xlabel('$|f|$, s$^{-1}$')
    plt.ylabel('PDS, relative units')
    plt.savefig('PDS.eps')
    
    fpds=open('pds.dat', 'w')
    for k in np.arange(nsize):
        fpds.write(str(freq[k])+' '+str(pds[k])+"\n")
    fpds.close()
    
