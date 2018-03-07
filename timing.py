import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import time
import pylab
import h5py

from conf import ifplot, kappa

#proper LaTeX support and decent fonts in figures 
rc('font',**{'family':'serif','serif':['Times']})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

# calculates the light curve and the power density spectrum
# it's much cheaper to read the datafile once and compute multiple data points
def fluxest(filename, lat0, lon0, nbins=10, ntimes=10, nfilter=None, nlim=None):
    """
    fluxest(<file name>, <viewpoint latitude, rad>, <viewpoint longitude, rad>, <keywords>)
    keywords:
    nbins -- number of bins in spectral space for PDS averaging
    ntimes -- number of temporal intervals for PDS calculation (for dynamic spectral analysis)
    nfilter
    """
    f = h5py.File(filename,'r')
    params=f["params"]
    nlons=params.attrs["nlons"] ; nlats=params.attrs["nlats"] ; omega=params.attrs["omega"] ; rsphere=params.attrs["rsphere"] ; tscale=params.attrs["tscale"] 
    lons1d = (2.*np.pi/nlons)*np.arange(nlons)
    clats1d = 2.*np.arange(nlats)/np.double(nlats)-1.
    lons,lats = np.meshgrid(lons1d, np.arccos(clats1d))
    dlons=2.*np.pi/np.size(lons1d) ; dlats=2./np.double(nlats)
    cosa=np.cos(lats)*np.cos(lat0)+np.sin(lats)*np.sin(lat0)*np.cos(lons-lon0)
    cosa=(cosa+np.fabs(cosa))/2. # only positive viewing angle
#    cosa=np.double(cosa>0.8)
    
    keys=f.keys()
    if(nfilter):
        if(nlim):
            keys=keys[nfilter:nlim] # filtering out first nfilter points
        else:
            keys=keys[nfilter:]
    else:
        if(nlim):
            keys=keys[:nlim] # filtering out everything after nlim

    #    keys=keys[4000:]
    nsize=np.size(keys)-1 # last key contains parameters
    print str(nsize)+" points from "+str(keys[0])+" to "+str(keys[-2])
    flux=np.zeros(nsize)  ;  mass=np.zeros(nsize) ;  tar=np.zeros(nsize) ; newmass=np.zeros(nsize)
    flc=open('out/lcurve.dat', 'w')
    for k in np.arange(nsize):
        data=f[keys[k]]
        sig=data["sig"][:] ; diss=data["diss"][:] ; accflag=data["accflag"][:]
        energy=data["energy"][:] ; beta=data["beta"][:]
        press = energy* 3. * (1.-beta/2.)
        tar[k]=data.attrs["t"]
        flux[k]=(press*(1.-beta)/(sig*kappa+1.)*cosa).sum()*dlons*dlats
        mass[k]=(sig*cosa).sum()*dlons*dlats
        newmass[k]=(sig*cosa*accflag).sum()*dlons*dlats
        flc.write(str(tar[k])+' '+str(flux[k])+' '+str(mass[k])+"\n")
#        print str(tar[k])+' '+str(flux[k])+' '+str(mass[k])+"\n"
    f.close() ; flc.close() 
    tar*=tscale

    wnan=np.where(np.isnan(flux))
    if(np.size(wnan)>0):
        print str(np.size(wnan))+" NaN points"
        ii=war_input('?')
    # linear trend:
    m,b =  np.polyfit(tar, mass, 1) # for "visible mass"
    mn,bn =  np.polyfit(tar, newmass, 1) # for "visible accreted mass"
    md,bd =  np.polyfit(tar, flux, 1) # for dissipation

    if(ifplot):
        plt.clf()
        plt.plot(tar, (flux-flux.min())/(flux.max()-flux.min()), color='k')
        plt.plot(tar, (mass-mass.min())/(mass.max()-mass.min()), color='r')
        plt.plot(tar, (newmass-newmass.min())/(newmass.max()-newmass.min()), color='g')
        plt.plot(tar, (tar*m+b-mass.min())/(mass.max()-mass.min()), color='r', linestyle='dashed')
        plt.plot(tar, (tar*mn+bn-newmass.min())/(newmass.max()-newmass.min()), color='g', linestyle='dashed')
        plt.plot(tar, (tar*md+bd-flux.min())/(flux.max()-flux.min()), color='k', linestyle='dashed')
        plt.xlabel('$t$')
        plt.ylabel('flux, relative units')
        plt.savefig('out/lcurve.eps')

    flux-=md*tar+bd ; mass-=m*tar+b; newmass-=mn*tar+bn # subtraction of linear trends
    tmean=tar.mean() ;     tspan=tar.max()-tar.min()
    freq1=1./tspan*np.double(ntimes)/2. ; freq2=freq1*np.double(nsize)/np.double(ntimes)
    
    # binning:
    binfreq=(freq2/freq1)**(np.arange(nbins+1)/np.double(nbins))*freq1
    binfreq[0]=0.

    print "frequencies ", binfreq
    binfreqc=(binfreq[:-1]+binfreq[1:])/2. ;   binfreqs=(-binfreq[:-1]+binfreq[1:])/2.
    pdsbin=np.zeros([ntimes, nbins]) ; pdsbinm=np.zeros([ntimes, nbins]) ; pdsbinn=np.zeros([ntimes, nbins])
    dpdsbin=np.zeros([ntimes, nbins]) ; dpdsbinm=np.zeros([ntimes, nbins]) ; dpdsbinn=np.zeros([ntimes, nbins])

    # dynamical spectra:
    #    fsp=np.zeros([ntimes, nsize]) ;fspm=np.zeros([ntimes, nsize]) ; fspn=np.zeros([ntimes, nsize])
    tcenter=np.zeros(ntimes, dtype=np.double)
    t2=np.zeros([ntimes+1, nbins+1], dtype=np.double)
    binfreq2=np.zeros([ntimes+1, nbins+1], dtype=np.double)
    t2[ntimes,:]=tar.max() ; binfreq2[ntimes,:]=binfreq
    for kt in np.arange(ntimes):
        tbegin=tar.min()+tspan*np.double(kt)/np.double(ntimes); tend=tar.min()+tspan*np.double(kt+1)/np.double(ntimes)
        tcenter[kt]=(tbegin+tend)/2.   ;     t2[kt,:]=tbegin;     binfreq2[kt,:]=binfreq
        wwindow=np.where((tar>=tbegin)&(tar<tend))
        wsize=np.size(wwindow)
        fsp=np.fft.rfft(flux[wwindow]/flux.std()) ;   fspm=np.fft.rfft(mass[wwindow]/mass.std())
        fspn=np.fft.rfft(newmass[wwindow]/newmass.std())
        pds=np.abs(fsp)**2  ;   pdsm=np.abs(fspm)**2 ;   pdsn=np.abs(fspn)**2
        freq = np.fft.rfftfreq(wsize, tspan/np.double(nsize)) # frequency grid (different for all the time bins)
        print "frequencies from "+str(freq.min())+" to "+str(freq.max())
        print "compare to "+str(binfreq[0])+" and "+str(binfreq[-1])
#        ii=raw_input('/')
        for kb in np.arange(nbins):
            freqrange=np.where((freq>=binfreq[kb])&(freq<binfreq[kb+1]))
#            if(np.size(freqrange)<=1):
#                print "kb = "+str(kb)
#                print "kt = "+str(kt)
            pdsbin[kt,kb]=pds[freqrange].mean()   ;     pdsbinm[kt,kb]=pdsm[freqrange].mean()   ;   pdsbinn[kt,kb]=pdsn[freqrange].mean()
            dpdsbin[kt,kb]=pds[freqrange].std()   ;     dpdsbinm[kt,kb]=pdsm[freqrange].std()   ;   dpdsbinn[kt,kb]=pdsn[freqrange].std()
    
    
    # let us also make a Fourier of the whole series:
    pdsbin_total=np.zeros([nbins]) ; pdsbinm_total=np.zeros([nbins]) ; pdsbinn_total=np.zeros([nbins])
    dpdsbin_total=np.zeros([nbins]) ; dpdsbinm_total=np.zeros([nbins]) ; dpdsbinn_total=np.zeros([nbins])
    fsp=np.fft.rfft(flux/flux.std()) ;  fspm=np.fft.rfft(mass/mass.std())
    fspn=np.fft.rfft(newmass/newmass.std())
    pds=np.abs(fsp)**2  ;  pdsm=np.abs(fspm)**2 ;   pdsn=np.abs(fspn)**2
    freq = np.fft.rfftfreq(nsize, tspan/np.double(nsize)) # frequency grid (total)
    for kb in np.arange(nbins):
        freqrange=np.where((freq>=binfreq[kb])&(freq<binfreq[kb+1]))
        pdsbin_total[kb]=pds[freqrange].mean()   ;     pdsbinm_total[kb]=pdsm[freqrange].mean()   ;   pdsbinn_total[kb]=pdsn[freqrange].mean()
        dpdsbin_total[kb]=pds[freqrange].std()   ;     dpdsbinm_total[kb]=pdsm[freqrange].std()
        dpdsbinn_total[kb]=pdsn[freqrange].std()

    if(ifplot):
        omegadisk=2.*np.pi/rsphere**1.5*0.9/tscale
        omega/=tscale
        wfin=np.where(np.isfinite(pdsbin_total))
        print omega, omegadisk
        print "pdsbin from "+str(pdsbin_total[wfin].min())+" tot "+str(pdsbin_total[wfin].max())
        pmin=pdsbin_total[wfin].min() ; pmax=pdsbin_total[wfin].max()
        # colour plot:
        plt.clf()
        plt.pcolormesh(t2, binfreq2, np.log(pdsbin), cmap='jet',vmin=np.log(pmin), vmax=np.log(pmax)) # tcenter2, binfreq2 should be corners
        # plt.contourf(tcenter2, binfreqc2, np.log(pdsbin))
        plt.plot([tar.min(), tar.max()],[omega/2./np.pi,omega/2./np.pi], 'r')
        plt.plot([tar.min(), tar.max()],[2.*omega/2./np.pi,2.*omega/2./np.pi], 'r')
        plt.yscale('log')
        plt.ylabel('$|f|$, s$^{-1}$')
        plt.xlabel('$t$, ms')
        plt.savefig('out/dynPDS.eps')

        # integral power density spectra
        plt.clf()
        plt.plot([omega/2./np.pi,omega/2./np.pi], [pdsbin_total.min(),pdsbin_total.max()], 'b')
        plt.plot([2.*omega/2./np.pi,2.*omega/2./np.pi], [pmin,pmax], 'b', linestyle='dotted')
        plt.plot([3.*omega/2./np.pi,3.*omega/2./np.pi], [pmin,pmax], 'b', linestyle='dotted')
        plt.plot([omegadisk/2./np.pi,omegadisk/2./np.pi], [pmin,pmax], 'm')
        plt.errorbar(binfreqc, pdsbin_total, yerr=dpdsbin_total, xerr=binfreqs, color='k', fmt='.')
        plt.errorbar(binfreqc, pdsbinm_total, yerr=dpdsbinm_total, xerr=binfreqs, color='r', fmt='.')
        plt.errorbar(binfreqc, pdsbinn_total, yerr=dpdsbinn_total, xerr=binfreqs, color='g', fmt='.')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1./tspan, np.double(nsize)/tspan)
        plt.ylim(pmin*0.5, pmax*2.)
        plt.xlabel('$|f|$, s$^{-1}$')
        plt.ylabel('PDS, relative units')
        plt.savefig('out/PDS.eps')

    # ascii output, total:
    fpdstots=open('out/pdstots_diss.dat', 'w')
    fpdstots_mass=open('out/pdstots_mass.dat', 'w')
    fpdstots_newmass=open('out/pdstots_newmass.dat', 'w')
    for k in np.arange(nbins):
        fpdstots.write(str(binfreq[k])+' '+str(binfreq[k+1])+' '+str(pdsbin_total[k])+' '+str(dpdsbin_total[k])+"\n")
        fpdstots_mass.write(str(binfreq[k])+' '+str(binfreq[k+1])+' '+str(pdsbinm_total[k])+' '+str(dpdsbinm_total[k])+"\n")
        fpdstots_newmass.write(str(binfreq[k])+' '+str(binfreq[k+1])+' '+str(pdsbinn_total[k])+' '+str(dpdsbinn_total[k])+"\n")
    fpdstots.close() ;   fpdstots_mass.close() ;   fpdstots_newmass.close()
  
    # ascii output, dynamical spectrum:
    fpds=open('out/pds_diss.dat', 'w')
    fpdsm=open('out/pds_mass.dat', 'w')
    fpdsn=open('out/pds_newmass.dat', 'w')
    # time -- frequency -- PDS
    for k in np.arange(nbins):
        for kt in np.arange(ntimes):
            fpds.write(str(tcenter[kt])+' '+str(binfreq[k])+' '+str(binfreq[k+1])+' '+str(pdsbin[kt,k])+' '+str(dpdsbin[kt,k])+"\n")
            fpdsn.write(str(tcenter[kt])+' '+str(binfreq[k])+' '+' '+str(binfreq[k+1])+str(pdsbinn[kt,k])+' '+str(dpdsbinn[kt,k])+"\n")
            fpdsm.write(str(tcenter[kt])+' '+str(binfreq[k])+' '+' '+str(binfreq[k+1])+str(pdsbinm[kt,k])+' '+str(dpdsbinm[kt,k])+"\n")
    fpds.close() ;   fpdsm.close() ;   fpdsn.close()
    
