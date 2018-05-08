from __future__ import print_function
from __future__ import division
from builtins import str
from past.utils import old_div
import numpy as np
import time
import h5py
from spharmt import Spharmt 

from scipy.integrate import trapz

from conf import ifplot, kappa, sigmascale

if(ifplot):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import rc
    import pylab

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
    # NSmass=params.attrs["mass"]
    print(type(nlons))
    NSmass=1.4
    x = Spharmt(int(nlons),int(nlats),int(old_div(nlons,3)),rsphere,gridtype='gaussian')
    lons1d = x.lons
    clats1d = np.sin(x.lats) # 2.*np.arange(nlats)/np.double(nlats)-1.
    lons,lats = np.meshgrid(lons1d, np.arccos(clats1d))
    dlons=2.*np.pi/np.size(lons1d) ; dlats=old_div(2.,np.double(nlats))
    cosa=np.cos(lats)*np.cos(lat0)+np.sin(lats)*np.sin(lat0)*np.cos(lons-lon0)
    cosa=old_div((cosa+np.fabs(cosa)),2.) # only positive viewing angle
#    cosa=np.double(cosa>0.8)
    
    keys=list(f.keys())
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
    print(str(nsize)+" points from "+str(keys[0])+" to "+str(keys[-2]))
    mass_total=np.zeros(nsize) ; energy_total=np.zeros(nsize)
    flux=np.zeros(nsize)  ;  mass=np.zeros(nsize) ;  tar=np.zeros(nsize) ; newmass=np.zeros(nsize)
    kenergy=np.zeros(nsize) ;  thenergy=np.zeros(nsize) ; meancs=np.zeros(nsize)
    kenergy_u=np.zeros(nsize) ;  kenergy_v=np.zeros(nsize)
    for k in np.arange(nsize):
        data=f[keys[k]]
        sig=data["sig"][:] ; diss=data["diss"][:] ; accflag=data["accflag"][:]
        ug=data["ug"][:] ;  vg=data["vg"][:]
        energy=data["energy"][:] ; beta=data["beta"][:]
#        print np.shape(energy)
        press = energy* 3. * (1.-old_div(beta,2.))
        tar[k]=data.attrs["t"]
        print("entrance "+keys[k]+", dimensions "+str(np.shape(energy)))
        flux[k]=trapz((press*(1.-beta)/(sig*kappa+1.)*cosa).sum(axis=1), x=clats1d)*dlons
        mass_total[k]=trapz(sig.sum(axis=1), x=-clats1d)*dlons
        mass[k]=trapz((sig*cosa).sum(axis=1), x=-clats1d)*dlons
        newmass[k]=trapz((accflag*sig*cosa).sum(axis=1), x=-clats1d)*dlons
        kenergy[k]=trapz(((ug**2+vg**2)*sig).sum(axis=1), x=-clats1d)*dlons/2.
        kenergy_u[k]=trapz(((ug**2)*sig).sum(axis=1), x=-clats1d)*dlons/2.
        kenergy_v[k]=trapz((vg**2*sig).sum(axis=1), x=-clats1d)*dlons/2.
        thenergy[k]=trapz(energy.sum(axis=1), x=-clats1d)*dlons
        csqmap=press/sig*(4.+beta)/3.
        meancs[k]=np.sqrt(csqmap.mean())
    f.close() 
    tar*=tscale 
    # mass consistency:
    mass_total *= rsphere**2*NSmass**2*2.18082e-2*(sigmascale/1e8) # 10^{20}g
    mass *= rsphere**2*NSmass**2*2.18082e-2*(sigmascale/1e8) # 10^{20}g
    newmass *= rsphere**2*NSmass**2*2.18082e-2*(sigmascale/1e8) # 10^{20}g
    meanmass=mass_total.mean() ; stdmass=mass_total.std()
    print("M = "+str(meanmass)+"+/-"+str(stdmass)+" X 10^{20} g")
    flux *= 1.4690e12*rsphere**2*NSmass**2*(sigmascale/1e8)  # 10^37 erg/s apparent luminosity
    kenergy *= rsphere**2*NSmass**2*19.6002e3*(sigmascale/1e8) # 10^{35} erg
    kenergy_u *= rsphere**2*NSmass**2*19.6002e3*(sigmascale/1e8) # 10^{35} erg
    kenergy_v *= rsphere**2*NSmass**2*19.6002e3*(sigmascale/1e8) # 10^{35} erg
    thenergy *= rsphere**2*NSmass**2*19.6002e3*(sigmascale/1e8)  # 10^{35} erg
    wnan=np.where(np.isnan(flux))
    if(np.size(wnan)>0):
        print(str(np.size(wnan))+" NaN points")
        ii=war_input('?')
    # linear trend:
    m,b =  np.polyfit(tar, mass, 1) # for "visible mass"
    mn,bn =  np.polyfit(tar, newmass, 1) # for "visible accreted mass"
    md,bd =  np.polyfit(tar, flux, 1) # for dissipation

    # ascii output:
    flc=open('out/lcurve.dat', 'w')
    fmc=open('out/mcurve.dat', 'w')    
    fec=open('out/ecurve.dat', 'w')
    flc.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    fec.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    fmc.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    flc.write("#   time, s     effective luminosity, 10^37erg/s    one-sided mass, 10^20 g\n")
    fec.write("#   time, s     kinetic energy, 10^35erg    thermal energy, 10^35 erg  kinetic energy along phi, 10^35 erg  kinetic energy along theta, 10^35erg \n")
    
    for k in np.arange(nsize):
        flc.write(str(tar[k])+' '+str(flux[k])+' '+str(mass[k])+"\n") 
        fec.write(str(tar[k])+' '+str(kenergy[k])+' '+str(thenergy[k])+' '+str(kenergy_u[k])+' '+str(kenergy_v[k])+"\n")
        fmc.write(str(tar[k])+' '+str(mass_total[k])+"\n")
        flc.flush() ; fmc.flush()
    flc.close() ; fmc.close() ; fec.close()
    print("total energy changed from "+str(kenergy[0]+thenergy[0])+" to "+str(kenergy[-1]+thenergy[-1])+"\n")

    if(ifplot): # move to plots.py!
        plt.clf()
        plt.plot(tar, mass_total, color='k')
        plt.plot(tar, mass, color='r')
        plt.plot(tar, newmass, color='g')
        plt.xlabel('$t$')
        plt.ylabel('mass, $10^{20}$g')
        plt.savefig('out/mcurve.eps')
        plt.clf()
        plt.plot(tar, kenergy+thenergy, color='k')
        plt.plot(tar, thenergy, color='r')
        plt.plot(tar, kenergy, color='b')
        plt.plot(tar, kenergy_v, color='b',  linestyle='dotted')
        plt.plot(tar, kenergy_u, color='b',  linestyle='dashed')
        #        plt.plot(tar, np.exp(2.*tar*omega/tscale)*0.00003, color='g')
        plt.ylim(thenergy.min()/2., kenergy.max()*1.5)
        plt.xlabel('$t$')
        plt.ylabel('energy, $10^{35}$erg')
        plt.yscale('log')
        plt.savefig('out/ecurve.eps')
        plt.clf()
        plt.plot(tar, old_div((flux-flux.min()),(flux.max()-flux.min())), color='k')
        plt.plot(tar, old_div((mass-mass.min()),(mass.max()-mass.min())), color='r')
        plt.plot(tar, old_div((newmass-newmass.min()),(newmass.max()-newmass.min())), color='g')
        plt.plot(tar, old_div((tar*m+b-mass.min()),(mass.max()-mass.min())), color='r', linestyle='dashed')
        plt.plot(tar, old_div((tar*mn+bn-newmass.min()),(newmass.max()-newmass.min())), color='g', linestyle='dashed')
        plt.plot(tar, old_div((tar*md+bd-flux.min()),(flux.max()-flux.min())), color='k', linestyle='dashed')
        plt.xlabel('$t$')
        plt.ylabel('flux, relative units')
        plt.savefig('out/lcurve.eps')

    flux-=md*tar+bd ; mass-=m*tar+b; newmass-=mn*tar+bn # subtraction of linear trends
    tmean=tar.mean() ;     tspan=tar.max()-tar.min()
    freq1=1./tspan*np.double(ntimes)/2. ; freq2=freq1*np.double(nsize)/np.double(ntimes)
    
    # sound waves (dynamic eigenfrequency):
    fsound=meancs/rsphere/2./np.pi/tscale*np.sqrt(2.)

    # binning:
    binfreq=(old_div(freq2,freq1))**(old_div(np.arange(nbins+1),np.double(nbins)))*freq1
    binfreq[0]=0.
    binfreqc=old_div((binfreq[:-1]+binfreq[1:]),2.) ;   binfreqs=old_div((-binfreq[:-1]+binfreq[1:]),2.)
    pdsbin=np.zeros([ntimes, nbins]) ; pdsbinm=np.zeros([ntimes, nbins]) ; pdsbinn=np.zeros([ntimes, nbins])
    dpdsbin=np.zeros([ntimes, nbins]) ; dpdsbinm=np.zeros([ntimes, nbins]) ; dpdsbinn=np.zeros([ntimes, nbins])
    nbin=np.zeros([ntimes, nbins]) ; nbinm=np.zeros([ntimes, nbins]) ; nbinn=np.zeros([ntimes, nbins])
    # dynamical spectra:
    #    fsp=np.zeros([ntimes, nsize]) ;fspm=np.zeros([ntimes, nsize]) ; fspn=np.zeros([ntimes, nsize])
    tcenter=np.zeros(ntimes, dtype=np.double)
    t2=np.zeros([ntimes+1, nbins+1], dtype=np.double)
    binfreq2=np.zeros([ntimes+1, nbins+1], dtype=np.double)
    t2[ntimes,:]=tar.max() ; binfreq2[ntimes,:]=binfreq
    for kt in np.arange(ntimes):
        tbegin=tar.min()+tspan*np.double(kt)/np.double(ntimes); tend=tar.min()+tspan*np.double(kt+1)/np.double(ntimes)
        tcenter[kt]=old_div((tbegin+tend),2.)   ;     t2[kt,:]=tbegin;     binfreq2[kt,:]=binfreq
        wwindow=np.where((tar>=tbegin)&(tar<tend))
        wsize=np.size(wwindow)
        fstd=flux.std()
        if(fstd<=0.):
            fstd=1.
        fsp=np.fft.rfft(old_div(flux[wwindow],fstd)) ;   fspm=np.fft.rfft(old_div(mass[wwindow],mass.std()))
        fspn=np.fft.rfft(old_div(newmass[wwindow],newmass.std()))
        pds=np.abs(fsp)**2  ;   pdsm=np.abs(fspm)**2 ;   pdsn=np.abs(fspn)**2
        freq = np.fft.fftfreq(wsize, old_div(tspan,np.double(nsize))) # frequency grid (different for all the time bins)
        print("frequencies from "+str(freq.min())+" to "+str(freq.max()))
        print("compare to "+str(binfreq[0])+" and "+str(binfreq[-1]))
#        ii=raw_input('/')
        for kb in np.arange(nbins):
            freqrange=np.where((freq>=binfreq[kb])&(freq<binfreq[kb+1]))
#            if(np.size(freqrange)<=1):
#                print "kb = "+str(kb)
#                print "kt = "+str(kt)
            pdsbin[kt,kb]=pds[freqrange].mean()   ;     pdsbinm[kt,kb]=pdsm[freqrange].mean()   ;   pdsbinn[kt,kb]=pdsn[freqrange].mean()
            dpdsbin[kt,kb]=pds[freqrange].std()   ;     dpdsbinm[kt,kb]=pdsm[freqrange].std()   ;   dpdsbinn[kt,kb]=pdsn[freqrange].std()    
            nbin[kt,kb]=np.size(pds[freqrange]) ; nbinm[kt,kb]=np.size(pdsm[freqrange]) ; nbinn[kt,kb]=np.size(pdsn[freqrange])

    # let us also make a Fourier of the whole series:
    pdsbin_total=np.zeros([nbins]) ; pdsbinm_total=np.zeros([nbins]) ; pdsbinn_total=np.zeros([nbins])
    dpdsbin_total=np.zeros([nbins]) ; dpdsbinm_total=np.zeros([nbins]) ; dpdsbinn_total=np.zeros([nbins])
    fsp=np.fft.rfft(old_div(flux,fstd)) ;  fspm=np.fft.rfft(old_div(mass,mass.std()))
    fspn=np.fft.rfft(old_div(newmass,newmass.std()))
    pds=np.abs(fsp)**2  ;  pdsm=np.abs(fspm)**2 ;   pdsn=np.abs(fspn)**2
    freq = np.fft.fftfreq(nsize, old_div(tspan,np.double(nsize))) # frequency grid (total)
    for kb in np.arange(nbins):
        freqrange=np.where((freq>=binfreq[kb])&(freq<binfreq[kb+1]))
        pdsbin_total[kb]=pds[freqrange].mean()   ;     pdsbinm_total[kb]=pdsm[freqrange].mean()   ;   pdsbinn_total[kb]=pdsn[freqrange].mean()
        dpdsbin_total[kb]=pds[freqrange].std()   ;     dpdsbinm_total[kb]=pdsm[freqrange].std()
        dpdsbinn_total[kb]=pdsn[freqrange].std()

#    print(pds)
    if(ifplot):
        omegadisk=2.*np.pi/rsphere**1.5*0.9/tscale
        omega/=tscale
        wfin=np.where(np.isfinite(pdsbin_total))
        print(omega, omegadisk)
        print("pdsbin from "+str(pdsbin_total[wfin].min())+" tot "+str(pdsbin_total[wfin].max()))
        pmin=pdsbin_total[wfin].min() ; pmax=pdsbin_total[wfin].max()
        # colour plot:
        plt.clf()
        plt.pcolormesh(t2, binfreq2, np.log(pdsbin), cmap='jet',vmin=np.log(pmin), vmax=np.log(pmax)) # tcenter2, binfreq2 should be corners
        # plt.contourf(tcenter2, binfreqc2, np.log(pdsbin))
        plt.plot([tar.min(), tar.max()],[omega/2./np.pi,omega/2./np.pi], 'r')
        plt.plot([tar.min(), tar.max()],[2.*omega/2./np.pi,2.*omega/2./np.pi], 'r')
        plt.yscale('log')
        plt.ylabel('$f$, Hz')
        plt.xlabel('$t$, ms')
        plt.savefig('out/dynPDS.eps')

        # integral power density spectra
        plt.clf()
        plt.plot([omega/2./np.pi,omega/2./np.pi], [pdsbin_total.min(),pdsbin_total.max()], 'b')
        plt.plot([fsound, fsound], [pdsbin_total.min(),pdsbin_total.max()], 'g', linewidth=1)
        plt.plot([fsound*2., fsound*2.], [pdsbin_total.min(),pdsbin_total.max()], 'g', linewidth=0.5, linestyle='dotted')
        plt.plot([2.*omega/2./np.pi,2.*omega/2./np.pi], [pmin,pmax], 'b', linestyle='dotted')
        plt.plot([3.*omega/2./np.pi,3.*omega/2./np.pi], [pmin,pmax], 'b', linestyle='dotted')
        plt.plot([omegadisk/2./np.pi,omegadisk/2./np.pi], [pmin,pmax], 'm')
        plt.errorbar(binfreqc, pdsbin_total, yerr=dpdsbin_total, xerr=binfreqs-freq1/2.*(np.arange(nbins)<=0.), color='k') #, fmt='.') # we need asymmetric error bars, otherwise they are incorrectly shown
        plt.errorbar(binfreqc, pdsbinm_total, yerr=dpdsbinm_total, xerr=binfreqs-freq1/2.*(np.arange(nbins)<=0.), color='r') #, fmt='.')
        plt.errorbar(binfreqc, pdsbinn_total, yerr=dpdsbinn_total, xerr=binfreqs-freq1/2.*(np.arange(nbins)<=0.), color='g') #, fmt='.')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(old_div(1.,tspan), old_div(np.double(nsize),tspan))
        plt.ylim(pmin*0.5, pmax*2.)
        plt.xlabel('$f$, Hz')
        plt.ylabel('PDS, relative units')
        plt.savefig('out/PDS.eps')

    # ascii output, total:
    fpdstots=open('out/pdstots_diss.dat', 'w')
    fpdstots_mass=open('out/pdstots_mass.dat', 'w')
    fpdstots_newmass=open('out/pdstots_newmass.dat', 'w')
    fpdstots.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    fpdstots.write("# flux variability \n")
    fpdstots_mass.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    fpdstots_mass.write("# mass (one-sided) variability \n")
    fpdstots_newmass.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    fpdstots_newmass.write("# newly accreted mass (one-sided) variability \n")
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
    fpds.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    fpds.write("# flux variability \n")
    fpds.write("# time -- frequency1 -- frequency2 -- PDS -- dPDS \n")
    fpdsn.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    fpds.write("# newly accreted mass (one-sided) variability \n")
    fpdsn.write("# time -- frequency1 -- frequency2 -- PDS -- dPDS \n")
    fpds.write("# total mass (one-sided) variability \n")
    fpdsm.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    fpdsm.write("# time -- frequency1 -- frequency2 -- PDS -- dPDS -- N \n")
    for k in np.arange(nbins):
        for kt in np.arange(ntimes):
            fpds.write(str(tcenter[kt])+' '+str(binfreq[k])+' '+str(binfreq[k+1])+' '+str(pdsbin[kt,k])+' '+str(dpdsbin[kt,k])+" "+str(nbin[kt,k])+"\n")
            fpdsn.write(str(tcenter[kt])+' '+str(binfreq[k])+' '+str(binfreq[k+1])+' '+str(pdsbinn[kt,k])+' '+str(dpdsbinn[kt,k])+" "+str(nbinn[kt,k])+"\n")
            fpdsm.write(str(tcenter[kt])+' '+str(binfreq[k])+' '+str(binfreq[k+1])+' '+str(pdsbinm[kt,k])+' '+str(dpdsbinm[kt,k])+" "+str(nbin[kt,k])+"\n")
    fpds.close() ;   fpdsm.close() ;   fpdsn.close()
    
fluxest('out/run.hdf5', 1.5, 0., nbins=20)
