from __future__ import print_function
from __future__ import division
from builtins import str
from past.utils import old_div
import os.path
import numpy as np
import time
import h5py
from spharmt import Spharmt 

from scipy.integrate import trapz

# TODO: global parameters should be read from hdf5 rather than taken from conf
from conf import ifplot

if(ifplot):
    import plots
    
# calculates the light curve and the power density spectrum
# it's much cheaper to read the datafile once and compute multiple data points
def fluxest(filename, lat0, lon0, nbins=10, ntimes=10, nfilter=None, nlim=None):
    """
    fluxest(<file name>, <viewpoint latitude, rad>, <viewpoint longitude, rad>, <keywords>)
    keywords:
    nbins -- number of bins in spectral space for PDS averaging
    ntimes -- number of temporal intervals for PDS calculation (for dynamic spectral analysis)
    nfilter
    TODO: variables from conf.py and stored in the hdf5 file blend together; ideally, we should get without refences to conf.py 
    """
    outdir=os.path.dirname(filename)
    print("writing output to "+outdir)
    f = h5py.File(filename,'r')
    params=f["params"]
    # loading global parameters from the hdf5 file
    nlons=params.attrs["nlons"] ; nlats=params.attrs["nlats"]
    omega=params.attrs["omega"] ; rsphere=params.attrs["rsphere"]
    tscale=params.attrs["tscale"] ; mass1=params.attrs["NSmass"]
    sigmascale=params.attrs["sigmascale"]; sigplus=params.attrs["sigplus"]
    overkepler=params.attrs["overkepler"]; tfric=params.attrs["tfric"]
    tdepl=params.attrs["tdepl"] ; mdotfinal=params.attrs["mdotfinal"]
    # NSmass=params.attrs["mass"]
#    print(type(nlons))
    x = Spharmt(int(nlons),int(nlats),int(old_div(nlons,3)),rsphere,gridtype='gaussian')
    lons1d = x.lons
    clats1d = np.sin(x.lats) # 2.*np.arange(nlats)/np.double(nlats)-1.
    lons,lats = np.meshgrid(lons1d, np.arccos(clats1d))
    dlons=2.*np.pi/np.size(lons1d) ; dlats=old_div(2.,np.double(nlats))
    cosa=np.cos(lats)*np.cos(lat0)+np.sin(lats)*np.sin(lat0)*np.cos(lons-lon0)
    cosa=(cosa+np.fabs(cosa))/2. # viewing angle positive (visible) or zero (invisible side)
    
    keys=list(f.keys())
    if(nfilter):
        if(nlim):
            keys=keys[nfilter:nlim] # filtering out first nfilter points
        else:
            keys=keys[nfilter:]
    else:
        if(nlim):
            keys=keys[:nlim] # filtering out everything after nlim

    nsize=np.size(keys)-1 # last key contains parameters
    print(str(nsize)+" points from "+str(keys[0])+" to "+str(keys[-2]))
    mass_total=np.zeros(nsize) ; energy_total=np.zeros(nsize)
    newmass_total=np.zeros(nsize) 
    flux=np.zeros(nsize)  ;  lumtot=np.zeros(nsize) ;  heattot=np.zeros(nsize)
    mass=np.zeros(nsize) ;  tar=np.zeros(nsize) ; newmass=np.zeros(nsize)
    kenergy=np.zeros(nsize) ;  thenergy=np.zeros(nsize) ; meancs=np.zeros(nsize)
    kenergy_u=np.zeros(nsize) ;  kenergy_v=np.zeros(nsize)
    angmoz_new=np.zeros(nsize)  ;  angmoz_old=np.zeros(nsize)
    maxdiss=np.zeros(nsize) ;    mindiss=np.zeros(nsize)
    sigmaver=np.zeros([nlats, nsize])  ;   sigmaver_lon=np.zeros([nlons, nsize])
    omeaver=np.zeros([nlats, nsize])  ;   omeaver_lon=np.zeros([nlons, nsize])
    tbottom=np.zeros(nsize) ; teff=np.zeros(nsize)
    tbottommax=np.zeros(nsize) ; tbottommin=np.zeros(nsize)
    omegamean=np.zeros(nsize) ;   mdot=np.zeros(nsize)
    for k in np.arange(nsize):
        data=f[keys[k]]
        sig=data["sig"][:] ; diss=data["diss"][:] ; accflag=data["accflag"][:]
        ug=data["ug"][:] ;  vg=data["vg"][:]
        energy=data["energy"][:] ; beta=data["beta"][:]
        qplus=data["qplus"][:] ;  qminus=data["qminus"][:]
        #        print np.shape(energy)
        press = energy* 3. * (1.-beta/2.)
        tar[k]=data.attrs["t"] ; mdot[k]=data.attrs['mdot']
        print("entrance "+keys[k]+", dimensions "+str(np.shape(energy)))
        flux[k]=trapz((qminus*cosa).sum(axis=1), x=-clats1d)*dlons
        lumtot[k]=data.attrs['lumtot']
        heattot[k]=data.attrs['heattot']
        mass_total[k]=data.attrs['mass']
        newmass_total[k]=data.attrs['newmass']
        mass[k]=trapz((sig*cosa).sum(axis=1), x=-clats1d)*dlons
        newmass[k]=trapz((accflag*sig*cosa).sum(axis=1), x=-clats1d)*dlons
        kenergy[k]=trapz(((ug**2+vg**2)*sig).sum(axis=1), x=-clats1d)*dlons/2.
        kenergy_u[k]=trapz(((ug**2)*sig).sum(axis=1), x=-clats1d)*dlons/2.
        kenergy_v[k]=trapz((vg**2*sig).sum(axis=1), x=-clats1d)*dlons/2.
        thenergy[k]=trapz(energy.sum(axis=1), x=-clats1d)*dlons
        angmoz_new[k]=trapz((sig*ug*np.sin(lats)*accflag).sum(axis=1), x=-clats1d)*dlons
        angmoz_old[k]=trapz((sig*ug*np.sin(lats)*(1.-accflag)).sum(axis=1), x=-clats1d)*dlons
        csqmap=press/sig*(4.+beta)/3. ;    meancs[k]=np.sqrt(csqmap.mean())
        tbottom[k]=(50.59*((1.-beta)*energy*sigmascale/mass1)**0.25).mean()
        tbottommin[k]=(50.59*((1.-beta)*energy*sigmascale/mass1)**0.25).min()
        tbottommax[k]=(50.59*((1.-beta)*energy*sigmascale/mass1)**0.25).max()
        teff[k]=(qminus.mean()*sigmascale/mass1)**0.25*3.64
        maxdiss[k]=diss.max() ;     mindiss[k]=diss.min()
        sigmaver[:,k]=(sig).mean(axis=1)/sig.mean()
        omeaver[:,k]=(sig*ug/np.sin(lats)).mean(axis=1)/sig.mean() /rsphere
        omeaver_lon[:,k]=(sig*ug/np.sin(lats)).mean(axis=0)/sig.mean() /rsphere
        sigmaver_lon[:,k]=(sig).mean(axis=0)/sig.mean()
        omegamean[k]=trapz((ug/np.sin(lats)*sig).sum(axis=1), x=-clats1d)*dlons / mass_total[k] /rsphere
    f.close() 
    tar*=tscale 
    # mass consistency:
    mass_total *= rsphere**2*mass1**2*2.18082e-2*(sigmascale/1e8) # 10^{20}g
    newmass_total *= rsphere**2*mass1**2*2.18082e-2*(sigmascale/1e8) # 10^{20}g
    mass *= rsphere**2*mass1**2*2.18082e-2*(sigmascale/1e8) # 10^{20}g
    newmass *= rsphere**2*mass1**2*2.18082e-2*(sigmascale/1e8) # 10^{20}g
    mdot *= rsphere**2*mass1**2*(sigmascale/1e8) * 0.00702374 # Msun/yr # check!
    meanmass=mass_total.mean() ; stdmass=mass_total.std()
    print("M = "+str(meanmass)+"+/-"+str(stdmass)+" X 10^{20} g")
    angmoz_new *= rsphere**3*mass1**3* 0.9655 * (sigmascale/1e8) # X 10^{26} erg * s
    angmoz_old *= rsphere**3*mass1**3* 0.9655 * (sigmascale/1e8) # X 10^{26} erg * s
    flux *= 0.3979*rsphere**2*mass1*sigmascale  # 10^37 erg/s apparent luminosity
    lumtot *= 0.3979*rsphere**2*mass1*sigmascale  # 10^37 erg/s total luminosity
    kenergy *= rsphere**2*mass1**2*19.6002e3*(sigmascale/1e8) # 10^{35} erg
    kenergy_u *= rsphere**2*mass1**2*19.6002e3*(sigmascale/1e8) # 10^{35} erg
    kenergy_v *= rsphere**2*mass1**2*19.6002e3*(sigmascale/1e8) # 10^{35} erg
    thenergy *= rsphere**2*mass1**2*19.6002e3*(sigmascale/1e8)  # 10^{35} erg
    omegamean/=tscale ; omeaver/=tscale ; omeaver_lon/=tscale
    wnan=np.where(np.isnan(flux))
    if(np.size(wnan)>0):
        print(str(np.size(wnan))+" NaN points")
        ii=war_input('?')
    # linear trend:
    m,b =  np.polyfit(tar, mass, 1) # for "visible mass"
    mn,bn =  np.polyfit(tar, newmass, 1) # for "visible accreted mass"
    md,bd =  np.polyfit(tar, flux, 1) # for dissipation

    # ascii output:
    flc=open(outdir+'/lcurve'+str(lat0)+'.dat', 'w')
    fmc=open(outdir+'/mcurve'+str(lat0)+'.dat', 'w')    
    fec=open(outdir+'/ecurve'+str(lat0)+'.dat', 'w')
    fdc=open(outdir+'/dcurve.dat', 'w')
    flc.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    fec.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    fmc.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    flc.write("#   time, s     effective luminosity, 10^37erg/s    total luminosity, 10^37 erg/s\n")
    fmc.write("#   time, s     effective mass, 10^20g    total mass, 10^20g \n")
    fec.write("#   time, s     kinetic energy, 10^35erg    thermal energy, 10^35 erg  kinetic energy along phi, 10^35 erg  kinetic energy along theta, 10^35erg \n")
    fdc.write("#   time, s     maximal neg. dissipation    maximal dissipation \n")
    
    for k in np.arange(nsize):
        flc.write(str(tar[k])+' '+str(flux[k])+' '+str(lumtot[k])+"\n") 
        fec.write(str(tar[k])+' '+str(kenergy[k])+' '+str(thenergy[k])+' '+str(kenergy_u[k])+' '+str(kenergy_v[k])+"\n")
        fmc.write(str(tar[k])+' '+str(mass[k])+' '+str(mass_total[k])+"\n")
        fdc.write(str(tar[k])+' '+str(maxdiss[k])+' '+str(-mindiss[k])+"\n")
        flc.flush() ; fmc.flush()
    flc.close() ; fmc.close() ; fec.close() ;  fdc.close()
    print("total energy changed from "+str(kenergy[0]+thenergy[0])+" to "+str(kenergy[-1]+thenergy[-1])+"\n")

    if(ifplot): 
        plots.timangle(tar*1e3, lats, lons, np.log(sigmaver),
                       np.log(sigmaver_lon), prefix=outdir+'/sig', omega=omega)
        plots.timangle(tar*1e3, lats, lons, omeaver,
                       np.log(omeaver_lon), prefix=outdir+'/ome')
        plots.sometimes(tar*1e3, [mdot, mdot*0.+mdotfinal], fmt=['k.', 'r-']
                        , prefix=outdir+'/mdot', title='mass accretion rate')
        plots.sometimes(tar*1e3, [maxdiss, -mindiss], fmt=['k', 'r'],
                        prefix=outdir+'/disslimits', title='dissipation limits')
        omegadisk=2.*np.pi/rsphere**1.5*0.9/tscale
        feq=(tdepl*tscale)*mdot/mass_total*630322
        omegaeq=(omega/tscale+omegadisk/feq)*(1.+feq)/(2.*np.pi)
        print("feq = "+str(feq))
        plots.sometimes(tar, [omegamean/(2.*np.pi), tar*0.+omega/tscale/(2.*np.pi), tar*0.+omegadisk/(2.*np.pi), omegaeq], fmt=['k', 'r', 'b', 'g'],
                        prefix=outdir+'/omega', title=r'frequency')
        if(sigplus>0.):
            plots.sometimes(tar, [mass_total, newmass_total, mass, newmass], fmt=['k-', 'k:', 'r-', 'g-']
                       #     , linest=['solid', 'dotted', 'solid', 'solid']
                            , prefix=outdir+'/m', title=r'mass, $10^{20}$g')
            plots.sometimes(tar, [newmass_total/mass_total], title='mass fraction', prefix=outdir+'/mfraction')
        plots.sometimes(tar, [kenergy+thenergy, thenergy, kenergy, kenergy_v, kenergy_u]
                        , fmt=['k-', 'r-', 'b-', 'b:', 'b--']
                        #, linest=['solid', 'solid', 'solid', 'dotted', 'dashed']
                        , title=r'energy, $10^{35}$erg', prefix=outdir+'/e')
        plots.sometimes(tar, [flux, lumtot, heattot], fmt=['k-', 'r-', 'r--'] 
                        #           , linest=['solid', 'solid', 'dashed']
                        , title=r'apparent luminosity, $10^{37}$erg s$^{-1}$', prefix=outdir+'/l')
        if(sigplus>0.):
            plots.sometimes(tar, [angmoz_new, angmoz_old, angmoz_new+angmoz_old]
                            , fmt=['-b', '-r', '-k'], title=r'angular momentum, $10^{26} {\rm g \,cm^2\, s^{-1}}$'
                            , prefix=outdir+'/angmoz', ylog=True)
        else:
            plots.sometimes(tar, [angmoz_new+angmoz_old], fmt=['-k']
                            , title=r'angular momentum, $10^{26} {\rm g \,cm^2\, s^{-1}}$'
                            , prefix=outdir+'/angmoz', ylog=False)
            
        plots.sometimes(tar, [tbottom, teff, tbottommin, tbottommax], fmt=['k-', 'r-', 'k:', 'k:']
                        # , linest=['solid', 'solid', 'dotted', 'dotted']
                        , title='$T$, keV', prefix=outdir+'/t')
        print("last Teff = "+str(teff[-1]))
        print("last Tb = "+str(tbottom[-1]))

    rawflux=flux
    flux-=md*tar+bd ; mass-=m*tar+b; newmass-=mn*tar+bn # subtraction of linear trends
    tmean=tar.mean() ;     tspan=tar.max()-tar.min()
    freq1=1./tspan*np.double(ntimes)/2. ; freq2=freq1*np.double(nsize)/np.double(ntimes)

    # binning:
    binfreq=(freq2-freq1)*((np.arange(nbins+1)/np.double(nbins)))+freq1
    # (freq2/freq1)**((np.arange(nbins+1)/np.double(nbins)))*freq1
    binfreq[0]=0.
    binfreqc=(binfreq[:-1]+binfreq[1:])/2. ;   binfreqs=(-binfreq[:-1]+binfreq[1:])/2.
    pdsbin=np.zeros([ntimes, nbins]) ; pdsbinm=np.zeros([ntimes, nbins]) ; pdsbinn=np.zeros([ntimes, nbins])
    dpdsbin=np.zeros([ntimes, nbins]) ; dpdsbinm=np.zeros([ntimes, nbins]) ; dpdsbinn=np.zeros([ntimes, nbins])
    nbin=np.zeros([ntimes, nbins]) ; nbinm=np.zeros([ntimes, nbins]) ; nbinn=np.zeros([ntimes, nbins])
    # dynamical spectra:
    #    fsp=np.zeros([ntimes, nsize]) ;fspm=np.zeros([ntimes, nsize]) ; fspn=np.zeros([ntimes, nsize])
    tcenter=np.zeros(ntimes, dtype=np.double)
    freqmax_diss=np.zeros(ntimes, dtype=np.double)
    dfreqmax_diss=np.zeros(ntimes, dtype=np.double)
    freqmax_mass=np.zeros(ntimes, dtype=np.double)
    dfreqmax_mass=np.zeros(ntimes, dtype=np.double)
    t2=np.zeros([ntimes+1, nbins+1], dtype=np.double)
    binfreq2=np.zeros([ntimes+1, nbins+1], dtype=np.double)
    t2[ntimes,:]=tar.max() ; binfreq2[ntimes,:]=binfreq
    binflux=np.zeros(ntimes, dtype=np.double) ;   binstd=np.zeros(ntimes, dtype=np.double)
    for kt in np.arange(ntimes):
        tbegin=tar.min()+tspan*np.double(kt)/np.double(ntimes); tend=tar.min()+tspan*np.double(kt+1)/np.double(ntimes)
        tcenter[kt]=(tbegin+tend)/2.   ;     t2[kt,:]=tbegin;     binfreq2[kt,:]=binfreq
        wwindow=np.where((tar>=tbegin)&(tar<tend))
        binflux[kt]=rawflux[wwindow].mean()   ;     binstd[kt]=rawflux[wwindow].std()
        wsize=np.size(wwindow)
        fsp=np.fft.rfft(flux[wwindow]/flux.std()) ;   fspm=np.fft.rfft(mass[wwindow]/mass.std())
        fspn=np.fft.rfft(newmass[wwindow]/mass.std()) # note we normalize for total mass variation; if sigplus===0, does not make sense but does not invoke errors
        freq = np.fft.rfftfreq(wsize, tspan/np.double(nsize)) # frequency grid (different for all the time bins)
        pds=np.abs(fsp*freq)**2  ;   pdsm=np.abs(fspm*freq)**2 ;   pdsn=np.abs(fspn*freq)**2
        
        for kb in np.arange(nbins):
            freqrange=np.where((freq>=binfreq[kb])&(freq<binfreq[kb+1]))
            pdsbin[kt,kb]=pds[freqrange].mean()   ;     pdsbinm[kt,kb]=pdsm[freqrange].mean()   ;   pdsbinn[kt,kb]=pdsn[freqrange].mean()
            dpdsbin[kt,kb]=pds[freqrange].std()   ;     dpdsbinm[kt,kb]=pdsm[freqrange].std()   ;   dpdsbinn[kt,kb]=pdsn[freqrange].std()    
            nbin[kt,kb]=np.size(pds[freqrange]) ; nbinm[kt,kb]=np.size(pdsm[freqrange]) ; nbinn[kt,kb]=np.size(pdsn[freqrange])
        # searching for the maximum in the PDS
        ston = 10. # signal-to-noize ratio
        kbmax=(pdsbin[kt,:]).argmax()
        freqmax_diss[kt]=binfreqc[kbmax]
        dfreqmax_diss[kt]=binfreqs[kbmax]
        kbmax=(pdsbinm[kt,:]).argmax()
        freqmax_mass[kt]=binfreqc[kbmax]
        dfreqmax_mass[kt]=binfreqs[kbmax]
    # let us also make a Fourier of the whole series:
    pdsbin_total=np.zeros([nbins]) ; pdsbinm_total=np.zeros([nbins]) ; pdsbinn_total=np.zeros([nbins])
    dpdsbin_total=np.zeros([nbins]) ; dpdsbinm_total=np.zeros([nbins]) ; dpdsbinn_total=np.zeros([nbins])
    fsp=np.fft.rfft(flux/flux.std()) ;  fspm=np.fft.rfft(mass/mass.std())
    fspn=np.fft.rfft(newmass/mass.std())
        
    pds=np.abs(fsp)**2  ;  pdsm=np.abs(fspm)**2 ;   pdsn=np.abs(fspn)**2
    freq = np.fft.rfftfreq(nsize, tspan/np.double(nsize)) # frequency grid (total)
    for kb in np.arange(nbins):
        freqrange=np.where((freq>=binfreq[kb])&(freq<binfreq[kb+1]))
        pdsbin_total[kb]=pds[freqrange].mean()   ;     pdsbinm_total[kb]=pdsm[freqrange].mean()   ;   pdsbinn_total[kb]=pdsn[freqrange].mean()
        dpdsbin_total[kb]=pds[freqrange].std()   ;     dpdsbinm_total[kb]=pdsm[freqrange].std()
        dpdsbinn_total[kb]=pdsn[freqrange].std()

    # we will need mean fluxes for vdKlis's plots:
    fbinfluxes=open(outdir+'/diss_binflux.dat', 'w')
    fbinfluxes.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    fbinfluxes.write("#  time -- mean flux -- flux std\n")
    for k in np.arange(ntimes):
        fbinfluxes.write(str(tcenter[k])+" "+str(binflux[k])+" "+str(binstd[k])+"\n")
    fbinfluxes.close()
    os.system('cp '+outdir+'/diss_binflux.dat '+outdir+'/mass_binflux.dat')
    # maximum in the dynamical spetcrum for vdKlis's plots:
    ffmax_diss=open(outdir+'/diss_freqmax.dat', 'w')
    ffmax_mass=open(outdir+'/mass_freqmax.dat', 'w')
    ffmax_diss.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    ffmax_diss.write("#  time -- freqmax -- dfreqmax\n")
    ffmax_mass.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    ffmax_mass.write("#  time -- freqmax -- dfreqmax\n")
    for k in np.arange(ntimes):
        if(np.isfinite(freqmax_diss[k])):
            ffmax_diss.write(str(tcenter[k])+" "+str(freqmax_diss[k])+" "+str(dfreqmax_diss[k])+"\n")
        if(np.isfinite(freqmax_mass[k])):
            ffmax_mass.write(str(tcenter[k])+" "+str(freqmax_mass[k])+" "+str(dfreqmax_mass[k])+"\n")
    ffmax_diss.close() ; ffmax_mass.close()
    
    # ascii output, total:
    fpdstots=open(outdir+'/pdstots_diss.dat', 'w')
    fpdstots_mass=open(outdir+'/pdstots_mass.dat', 'w')
    fpdstots_newmass=open(outdir+'/pdstots_newmass.dat', 'w')
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
    os.system('cp '+outdir+'/pdstots_diss.dat '+outdir+'/pdstots_diss'+str(lat0)+'.dat')
    os.system('cp '+outdir+'/pdstots_mass.dat '+outdir+'/pdstots_mass'+str(lat0)+'.dat')
    # ascii output, t-th/phi diagrams
    ftth=open(outdir+'/tth.dat', 'w')
    ftphi=open(outdir+'/tphi.dat', 'w')
    # time -- theta -- <Sigma>, time is aliased
    ftth.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    ftphi.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    ftth.write("# time -- th (rad) -- <Sigma>_phi \n")
    ftphi.write("# time -- phi (rad) -- <Sigma>_theta \n")
    kalias=5
    for k in np.arange(nsize):    
        if k % kalias ==0:
            for kth in np.arange(nlats):
                ftth.write(str(tar[k])+" "+str(lats[kth,0])+" "+str(sigmaver[kth, k])+"\n")
            for kphi in np.arange(nlons):
                ftphi.write(str(tar[k])+" "+str(lons[0,kphi])+" "+str(sigmaver_lon[kphi, k])+"\n")
    ftth.close() ; ftphi.close()
    # ascii output, dynamical spectrum:
    fpds=open(outdir+'/pds_diss.dat', 'w')
    fpdsm=open(outdir+'/pds_mass.dat', 'w')
    fpdsn=open(outdir+'/pds_newmass.dat', 'w')
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
    
    if(ifplot):
#        omega/=tscale
        wfin=np.where(np.isfinite(pdsbin_total))
#        print(omega, omegadisk)
        print("pdsbin from "+str(pdsbin_total[wfin].min())+" tot "+str(pdsbin_total[wfin].max()))
        pmin=pdsbin_total[wfin].min() ; pmax=pdsbin_total[wfin].max()
        # colour plot:
        plots.dynsplot(infile=outdir+"/pds_diss")
        plots.dynsplot(infile=outdir+"/pds_mass")
        plots.dynsplot(infile=outdir+"/pds_newmass")
        plots.pdsplot(infile=outdir+"/pdstots_diss")
        plots.pdsplot(infile=outdir+"/pdstots_mass")
        plots.pdsplot(infile=outdir+"/pdstots_newmass")

####################################################################################################
def meanmaps(filename, n1, n2):
    '''
    makes time(frame)-averaged pictures from hdf5 file filename, frames n1 to n2
    '''
    outdir=os.path.dirname(filename)
    print("writing output in "+outdir)
    f = h5py.File(filename,'r')
    params=f["params"]
    nlons=params.attrs["nlons"] ; nlats=params.attrs["nlats"] ; omega=params.attrs["omega"] ; rsphere=params.attrs["rsphere"] ; tscale=params.attrs["tscale"]
    # NSmass=params.attrs["mass"]
    print(type(nlons))
    x = Spharmt(int(nlons),int(nlats),int(nlons/3.),rsphere,gridtype='gaussian')
    lons1d = x.lons
    clats1d = np.sin(x.lats) # 2.*np.arange(nlats)/np.double(nlats)-1.
    lons,lats = np.meshgrid(lons1d, np.arccos(clats1d))
    #    print(np.shape(lons))
    sigmean=np.zeros([nlats, nlons], dtype=np.double)
    energymean=np.zeros([nlats, nlons], dtype=np.double)
    ugmean=np.zeros([nlats, nlons], dtype=np.double) ;   vgmean=np.zeros([nlats, nlons], dtype=np.double)
    ugdisp=np.zeros([nlats, nlons], dtype=np.double) ;   vgdisp=np.zeros([nlats, nlons], dtype=np.double)
    uvcorr=np.zeros([nlats, nlons], dtype=np.double)
    qmmean=np.zeros([nlats, nlons], dtype=np.double) ; qpmean=np.zeros([nlats, nlons], dtype=np.double)
    keys=list(f.keys())
    for k in np.arange(n2-n1)+n1:
        print("reading data entry "+keys[k])
        data=f[keys[k]]
        sig=data["sig"][:] ; diss=data["diss"][:] ; accflag=data["accflag"][:]
        ug=data["ug"][:] ;  vg=data["vg"][:]
        energy=data["energy"][:] ; beta=data["beta"][:]
        qplus=data["qplus"][:] ;  qminus=data["qminus"][:]
        sigmean+=sig ; energymean+=energy ; ugmean+=ug ; vgmean+=vg ; ugdisp+=ug**2 ; vgdisp+=vg**2
        qmmean+=qminus ; qpmean+=qplus
    f.close()
    sigmean/=np.double(n2-n1) ; energymean/=np.double(n2-n1) ; ugmean/=np.double(n2-n1); vgmean/=np.double(n2-n1)
    qmmean/=np.double(n2-n1) ; qpmean/=np.double(n2-n1)
    ugdisp=(ugdisp/np.double(n2-n1)-ugmean**2) ; vgdisp=(vgdisp/np.double(n2-n1)-vgmean**2)
    uvcorr=uvcorr/np.double(n2-n1)-ugmean*vgmean

    # ascii output:
    fout=open(outdir+'/meanmap.dat', 'w')
    for kx in np.arange(nlons):
        for ky in np.arange(nlats):
            fout.write(str(lons[ky,kx])+" "+str(lats[ky,kx])+" "+str(sigmean[ky,kx])+" "+str(energymean[ky,kx])+" "+str(ugmean[ky,kx])+" "+str(vgmean[ky,kx])+" "+str(ugdisp[ky,kx])+" "+str(vgdisp[ky,kx])+" "+str(uvcorr[ky,kx])+"\n")
    fout.close()
    if(ifplot):
        plots.somemap(lons, lats, sigmean, outdir+"/mean_sigma.png")
        plots.somemap(lons, lats, energymean, outdir+"/mean_energy.png")
        plots.somemap(lons, lats, uvcorr/np.sqrt(vgdisp*ugdisp), outdir+"/mean_uvcorr.png")
        plots.somemap(lons, lats, (vgdisp-ugdisp)/(vgdisp+ugdisp), outdir+"/mean_anisotropy.png")
       
fluxest('out/run.hdf5', np.pi/2., 0., ntimes=5, nbins=30)
# meanmaps('out/run.hdf5', 1000, 2000)
