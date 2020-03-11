from __future__ import print_function
from __future__ import division
from builtins import str
import os.path
import numpy as np
import time
import h5py
from spharmt import Spharmt 

from scipy.integrate import trapz
from scipy.interpolate import interp1d

from conf import ifplot

if(ifplot):
    import plots
    
# calculates the light curve and the power density spectrum
# it's much cheaper to read the datafile once and compute multiple data points
def lightcurves(filename, lat0, lon0, iftth = True, nmax = None):
    """
    lightcurves(<file name>, <viewpoint latitude, rad>, <viewpoint longitude, rad>)
    TODO: variables from conf.py and stored in the hdf5 file blend together; ideally, we should get without refences to conf.py 
    """
    outdir=os.path.dirname(filename)
    print("writing output to "+outdir)
    f = h5py.File(filename,'r', libver='latest')
    params=f["params"]
    # loading global parameters from the hdf5 file
    nlons=params.attrs["nlons"] ; nlats=params.attrs["nlats"]
    omega=params.attrs["omega"] ; rsphere=params.attrs["rsphere"]
    tscale=params.attrs["tscale"] ; mass1=params.attrs["NSmass"]
    csqmin=params.attrs["csqmin"] 
    sigmascale=params.attrs["sigmascale"]; sigplus=params.attrs["sigplus"]
    overkepler=params.attrs["overkepler"]; tfric=params.attrs["tfric"]
    tdepl=params.attrs["tdepl"] ; mdotfinal=params.attrs["mdotfinal"]
    mu=0.6
    cssqscale=2.89591e-06 * sigmascale**0.25 / mu * mass1**0.25
    # NSmass=params.attrs["mass"]
    #    print(type(nlons))
    x = Spharmt(int(nlons),int(nlats),int(np.double(nlons)/3.),rsphere,gridtype='gaussian') # coordinate mesh
    lons1d = x.lons ; lats1d = x.lats
    clats1d = np.sin(x.lats) 
    lons,lats = np.meshgrid(lons1d, np.arccos(clats1d))
    dlons=2.*np.pi/np.size(lons1d) ; dlats=(2./np.double(nlats))
    # viewing angle:
    cosa=np.cos(lats)*np.cos(lat0)+np.sin(lats)*np.sin(lat0)*np.cos(lons-lon0)
    cosa=(cosa+np.fabs(cosa))/2. # viewing angle positive (visible) or zero (invisible side)
    
    keys=list(f.keys())

    nsize=np.size(keys)-1 # last key contains parameters
    if nmax is not None:
        nsize = np.minimum(nsize, nmax+1)
    print(str(nsize)+" points from "+str(keys[0])+" to "+str(keys[-2]))
    #    ipoints = input("keys")
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
    rxyaver=np.zeros([nlats, nsize]) # longitudinally-averaged Reynolds stress
    tbaver=np.zeros([nlats, nsize]) # longitudinally-averaged bottom temp
    tbottommean=np.zeros(nsize) ; teff=np.zeros(nsize)
    tbottommax=np.zeros(nsize) ; tbottommin=np.zeros(nsize)
    omegamean=np.zeros(nsize) ;   mdot=np.zeros(nsize)
    
    for k in np.arange(nsize):
        data=f[keys[k]]
        sig=data["sig"][:] ; diss=data["diss"][:] ; accflag=data["accflag"][:]
        sig_phiav = sig.mean(axis=1)
        sig_thetav = trapz(sig, x=-clats1d, axis=0)*dlons/2.
        ug=data["ug"][:] ;  vg=data["vg"][:]
        energy=data["energy"][:] ; beta=data["beta"][:]
        qplus=data["qplus"][:] ;  qminus=data["qminus"][:]
        print("beta from "+str(beta.min())+"  to "+str(beta.max()))
        press = energy / (3. * (1.-beta/2.))
        uvcorr =  ug*vg
        for klat in np.arange(nlats):
            uvcorr[klat,:] = (ug[klat,:]-ug[klat,:].mean()) * (vg[klat,:]-vg[klat,:].mean())
        dimsequal = (np.shape(sig)[0] == nlats) & (np.shape(sig)[1] == nlons)
        if dimsequal:
            x1= x
            lons1 = lons ; lats1 = lats
            dlons1 = dlons ; dlats1 = dlats
            lons1d1 = lons1d ; lats1d1 = lats1d
            clats1d1 = clats1d
            cosa1 = cosa
        else:
            # if the sizes of individual entries are unequal
            #            print("dimensions changed to "+str(np.shape(sig)))
            nlats1, nlons1 = np.shape(sig)
            x1 = Spharmt(int(nlons1),int(nlats1),int(np.double(nlons1)/3.),rsphere,gridtype='gaussian')
            lons1d1 = x1.lons ; lats1d1 = x1.lats
            clats1d1 = np.sin(x1.lats) # 2.*np.arange(nlats)/np.double(nlats)-1.
            lons1,lats1 = np.meshgrid(lons1d1, np.arccos(clats1d1))
            dlons1=2.*np.pi/np.size(lons1d1) ; dlats1=(2./np.double(nlats1))
            cosa1=np.cos(lats1)*np.cos(lat0)+np.sin(lats1)*np.sin(lat0)*np.cos(lons1-lon0)
            cosa1=(cosa1+np.fabs(cosa1))/2. # viewing angle positive (visible) or zero (invisible side)

        tar[k]=data.attrs["t"] ; mdot[k]=data.attrs['mdot']
        print("entrance "+keys[k]+", dimensions "+str(np.shape(energy)))
        flux[k]=trapz((qminus*cosa1).sum(axis=1), x=-clats1d1)*dlons1
        lumtot[k]=data.attrs['lumtot']
        heattot[k]=data.attrs['heattot']
        mass_total[k]=trapz(sig_phiav, x=-clats1d)*dlons
        newmass_total[k]=trapz((sig*accflag).sum(axis=1), x=-clats1d)*dlons
        mass[k]=trapz((sig*cosa1).sum(axis=1), x=-clats1d1)*dlons1
        sigmean = mass_total[k]/4./np.pi # mean surface density
        newmass[k]=trapz((accflag*sig*cosa1).sum(axis=1), x=-clats1d1)*dlons1
        kenergy[k]=trapz(((ug**2+vg**2)*sig).sum(axis=1), x=-clats1d1)*dlons1/2.
        kenergy_u[k]=trapz(((ug**2)*sig).sum(axis=1), x=-clats1d1)*dlons1/2.
        kenergy_v[k]=trapz((vg**2*sig).sum(axis=1), x=-clats1d1)*dlons1/2.
        thenergy[k]=trapz(energy.sum(axis=1), x=-clats1d1)*dlons1
        angmoz_new[k]=trapz((sig*ug*np.sin(lats1)*accflag).sum(axis=1), x=-clats1d1)*dlons1
        angmoz_old[k]=trapz((sig*ug*np.sin(lats1)*(1.-accflag)).sum(axis=1), x=-clats1d1)*dlons1
        csqmap=press/sig*(4.+beta)/3. ;    meancs[k]=np.sqrt(csqmap.mean())
        tbottom=339.6*((1.-beta)*sig*(sigmascale/1e8)/mass1/rsphere**2)**0.25
        tbottommean[k]=tbottom.mean()
        tbottommin[k]=tbottom.min()
        tbottommax[k]=tbottom.max()
        teff[k]=(qminus.mean()*sigmascale/mass1)**0.25*3.64
        maxdiss[k]=diss.max() ;     mindiss[k]=diss.min()
        # for tth/tphi plots:
        if(dimsequal):
            tbaver[:,k] = (tbottom*sig).mean(axis=1)/sig_phiav # Sigma-weighed
            rxyaver[:,k] = (uvcorr*sig).mean(axis=1)/sig_phiav
            sigmaver[:,k]=(sig).mean(axis=1)/sigmean
            omeaver[:,k]=(sig*ug/np.sin(lats1)).mean(axis=1)/sig_phiav /rsphere
            omeaver_lon[:,k]=(sig*ug/np.sin(lats1)).mean(axis=0)/sig.mean(axis=0) /rsphere
            sigmaver_lon[:,k]=sig_thetav/sigmean
            # trapz(sig, x=-clats1d, axis=0)/2.
            # (sig).mean(axis=0)/sigmean
        else:
            # if dimensions are not equal, and we need to interpolate
            #            print("size (lats1) = "+str(np.size(lats1)))
            averfun = interp1d(lats1d1, (tbottom*sig).mean(axis=1)/sig_phiav, kind='linear')
            tbaver[:,k] = averfun(lats1d)
            averfun = interp1d(lats1d1, (uvcorr*sig).mean(axis=1)/sig_phiav, kind='linear')
            rxyaver[:,k] = averfun(lats1d)
            averfun = interp1d(lats1d1, (sig).mean(axis=1)/sig_phiav, kind='linear')
            sigmaver[:,k] = averfun(lats1d)
            averfun = interp1d(lats1d1, (sig*ug/np.sin(lats1)).mean(axis=1)/sig_phiav /rsphere, kind='linear')
            omeaver[:,k] = averfun(lats1d)
            averfun = interp1d(lons1d1, sig_thetav/sigmean, kind='linear')
            sigmaver_lon[:,k] = averfun(lons1d)
            averfun = interp1d(lons1d1, trapz(sig*ug/np.sin(lats1), x=-clats1d,axis=0)/sig_thetav /rsphere, kind='linear')
            omeaver_lon[:,k] = averfun(lons1d)
        omegamean[k]=trapz((ug/np.sin(lats1)*sig).sum(axis=1), x=-clats1d1)*dlons1 / mass_total[k] /rsphere
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
    flux *= 4.*0.3979*rsphere**2*mass1*sigmascale  # 10^37 erg/s apparent luminosity
    lumtot *= 4.*0.3979*rsphere**2*mass1*sigmascale  # 10^37 erg/s total luminosity
    heattot *= 4.*0.3979*rsphere**2*mass1*sigmascale  # 10^37 erg/s total heating
    qnstot = 4.*0.3979*rsphere**2*mass1*sigmascale * 4.*np.pi * (csqmin/cssqscale)**4  # heating from the NS surface
    kenergy *= rsphere**2*mass1**2*19.6002e3*(sigmascale/1e8) # 10^{35} erg
    kenergy_u *= rsphere**2*mass1**2*19.6002e3*(sigmascale/1e8) # 10^{35} erg
    kenergy_v *= rsphere**2*mass1**2*19.6002e3*(sigmascale/1e8) # 10^{35} erg
    thenergy *= rsphere**2*mass1**2*19.6002e3*(sigmascale/1e8)  # 10^{35} erg
    omegamean/=tscale ; omeaver/=tscale ; omeaver_lon/=tscale
    wnan=np.where(np.isnan(flux))
    if(np.size(wnan)>0):
        print(str(np.size(wnan))+" NaN points")
        ii=war_input('?')

    # ascii output:
    flc=open(outdir+'/lcurve'+str(lat0)+'.dat', 'w')
    fmc=open(outdir+'/mcurve'+str(lat0)+'.dat', 'w')    
    fec=open(outdir+'/ecurve'+str(lat0)+'.dat', 'w')
    fdc=open(outdir+'/dcurve.dat', 'w')
    flc.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    fec.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    fmc.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
    flc.write("#   time, s     effective luminosity, 10^37erg/s    total luminosity, 10^37 erg/s   total heating, 10^37 erg/s\n")
    fmc.write("#   time, s     effective mass, 10^20g    total mass, 10^20g \n")
    fec.write("#   time, s     kinetic energy, 10^35erg    thermal energy, 10^35 erg  kinetic energy along phi, 10^35 erg  kinetic energy along theta, 10^35erg \n")
    fdc.write("#   time, s     maximal neg. dissipation    maximal dissipation \n")
    for k in np.arange(nsize):
        flc.write(str(tar[k])+' '+str(flux[k])+' '+str(lumtot[k])+' '+str(heattot[k])+"\n") 
        fec.write(str(tar[k])+' '+str(kenergy[k])+' '+str(thenergy[k])+' '+str(kenergy_u[k])+' '+str(kenergy_v[k])+"\n")
        fmc.write(str(tar[k])+' '+str(mass[k])+' '+str(mass_total[k])+"\n")
        fdc.write(str(tar[k])+' '+str(maxdiss[k])+' '+str(-mindiss[k])+"\n")
        flc.flush() ; fmc.flush()
    flc.close() ; fmc.close() ; fec.close() ;  fdc.close()
    print("total energy changed from "+str(kenergy[0]+thenergy[0])+" to "+str(kenergy[-1]+thenergy[-1])+"\n")

    if iftth:
        # ascii output, t-th/phi diagrams
        ftth_sigma=open(filename+'_sig_tth.dat', 'w')
        ftphi_sigma=open(filename+'_sig_tphi.dat', 'w')
        ftth_omega=open(filename+'_omega_tth.dat', 'w')
        ftphi_omega=open(filename+'_omega_tphi.dat', 'w')
        frey=open(filename+'_rxy_tth.dat', 'w')
        ftb=open(filename+'_tb_tth.dat', 'w')
        # time -- theta -- <Sigma>, time is aliased
        ftth_sigma.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
        ftth_omega.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
        ftb.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
        frey.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
        ftphi_sigma.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
        ftphi_omega.write("#  generated by fluxest, lat0="+str(lat0)+"rad, lon0="+str(lon0)+"rad \n")
        ftth_sigma.write("# time -- th (rad) -- <Sigma>_phi \n")
        ftth_omega.write("# time -- th (rad) -- <Omega>_phi \n")
        ftb.write("# time -- th (rad) -- <T_b>_phi \n")
        frey.write("# time -- th (rad) -- <DU DV>_phi \n")
        ftphi_sigma.write("# time -- phi (rad) -- <Sigma>_theta \n")
        ftphi_omega.write("# time -- phi (rad) -- <Omega>_theta \n")
        kalias=5
        for k in np.arange(nsize):    
            if k % kalias ==0:
                for kth in np.arange(nlats):
                    ftth_sigma.write(str(tar[k])+" "+str(lats[kth,0])+" "+str(sigmaver[kth, k])+"\n")
                    ftth_omega.write(str(tar[k])+" "+str(lats[kth,0])+" "+str(omeaver[kth, k])+"\n")
                    frey.write(str(tar[k])+" "+str(lats[kth,0])+" "+str(rxyaver[kth, k])+"\n")
                    ftb.write(str(tar[k])+" "+str(lats[kth,0])+" "+str(tbaver[kth, k])+"\n")
                for kphi in np.arange(nlons):
                    ftphi_sigma.write(str(tar[k])+" "+str(lons[0,kphi])+" "+str(sigmaver_lon[kphi, k])+"\n")
                    ftphi_omega.write(str(tar[k])+" "+str(lons[0,kphi])+" "+str(omeaver_lon[kphi, k])+"\n")
        ftth_sigma.close() ; ftth_omega.close() ; frey.close() ; ftphi_sigma.close() ; ftphi_omega.close() ; ftb.close()
    
    # plots:
    if(ifplot):
        if iftth:
            plots.timangle(tar*1e3, lats, lons, np.log(sigmaver),
                           np.log(sigmaver_lon), prefix=outdir+'/sig', omega=omega/1e3/tscale)
            plots.timangle(tar*1e3, lats, lons, omeaver,
                           np.log(omeaver_lon), prefix=outdir+'/ome')
            plots.timangle(tar*1e3, lats, lons, rxyaver,
                           np.log(omeaver_lon), prefix=outdir+'/rxy', nolon=True)
            plots.timangle(tar*1e3, lats, lons, tbaver,
                           np.log(omeaver_lon), prefix=outdir+'/tb', nolon=True)
        plots.sometimes(tar*1e3, [mdot, mdot*0.+mdotfinal], fmt=['k.', 'r-']
                        , prefix=outdir+'/mdot', title='mass accretion rate')
        plots.sometimes(tar*1e3, [maxdiss, -mindiss], fmt=['k', 'r'],
                        prefix=outdir+'/disslimits', title='dissipation limits')
        omegadisk = rsphere**(-1.5)*overkepler/tscale
        feq=(tdepl*tscale)*mdot/mass_total*630322.
        omegaeq=(omega/tscale+omegadisk/feq)*(1.+feq)/(2.*np.pi)
        print("feq = "+str(feq))
        plots.sometimes(tar*1e3, [omegamean/(2.*np.pi), tar*0.+omega/tscale/(2.*np.pi), tar*0.+omegadisk/(2.*np.pi), omegaeq], fmt=['k', 'r', 'b', 'g'],
                        prefix=outdir+'/omega', title=r'frequency')
        plots.sometimes(tar*1e3, [mass_total, newmass_total, mass_total-newmass_total, mdotfinal*tar*tscale*630322.]
                        , fmt=['k-', 'g-', 'r-', 'k:'], ylog=False
                        , prefix=outdir+'/m', title=r'mass, $10^{20}$g')
        if(sigplus>0.):
            plots.sometimes(tar*1e3, [newmass_total/mass_total], title='mass fraction', prefix=outdir+'/mfraction')

        plots.sometimes(tar*1e3, [kenergy+thenergy, thenergy, kenergy, kenergy_v, kenergy_u]
                        , fmt=['k-', 'r-', 'b-', 'b:', 'b--']
                        #, linest=['solid', 'solid', 'solid', 'dotted', 'dashed']
                        , title=r'energy, $10^{35}$erg', prefix=outdir+'/e')
        plots.sometimes(tar*1e3, [flux, lumtot, heattot, flux*0.+qnstot], fmt=['k-', 'r-', 'g--', 'k:'] 
                        #       what goes on with heattot???!!!
                        , title=r'apparent luminosity, $10^{37}$erg s$^{-1}$', prefix=outdir+'/l')
        if(sigplus>0.):
            plots.sometimes(tar*1e3, [angmoz_new, angmoz_old, angmoz_new+angmoz_old]
                            , fmt=['-b', '-r', '-k'], title=r'angular momentum, $10^{26} {\rm g \,cm^2\, s^{-1}}$'
                            , prefix=outdir+'/angmoz', ylog=True)
        else:
            plots.sometimes(tar*1e3, [angmoz_new+angmoz_old], fmt=['-k']
                            , title=r'angular momentum, $10^{26} {\rm g \,cm^2\, s^{-1}}$'
                            , prefix=outdir+'/angmoz', ylog=False)
            
        plots.sometimes(tar*1e3, [tbottommean, teff, tbottommin, tbottommax], fmt=['k-', 'r-', 'k:', 'k:']
                        # , linest=['solid', 'solid', 'dotted', 'dotted']
                        , title='$T$, keV', prefix=outdir+'/t')
        print("last Teff = "+str(teff[-1]))
        print("last Tb = "+str(tbottommean[-1]))

#################################################################################################
#
def specmaker(infile='out/lcurve', nbins = 10, logbinning = False, trange = None):
    '''
    makes a Fourier PDS out of a light curve
    '''
    lines = np.loadtxt(infile+".dat")
    t = lines[:,0] ; x = lines[:,1]
    if(trange is not None):
        w=np.where((t<trange[1])&(t>trange[0]))
        t=t[w] ; x=x[w]
    nt = np.size(t) ;    tspan = t.max() - t.min() 
    dt = tspan / np.double(nt) 
    pdsbin=np.zeros(nbins) ; dpdsbin=np.zeros(nbins)
    freq1 =1./tspan/2. ; freq2=freq1*np.double(nt)/2.
    # binning:
    if(logbinning):
        binfreq=(freq2/freq1)**((np.arange(nbins+1)/np.double(nbins)))*freq1
    else:
        binfreq=(freq2-freq1)*((np.arange(nbins+1)/np.double(nbins)))+freq1
    # what about removing some trend? linear or polynomial?
    m,b = np.polyfit(t, x, 1)
    x1=x-t*m-b
    fsp=2.*np.fft.rfft(x1)/x.sum() # this gives a fractional amplitude
    pds=np.abs(fsp)**2
    freq = np.fft.rfftfreq(nt, dt)
    for kb in np.arange(nbins):
        freqrange=(freq>=binfreq[kb])&(freq<binfreq[kb+1])
        pdsbin[kb]=pds[freqrange].sum() 
        dpdsbin[kb]=pds[freqrange].std()
        # /np.sqrt(np.double(freqrange.sum())-1.)
    # ascii output:
    fpds = open(infile+'_pdstot.dat', 'w')
    for k in np.arange(nbins):
        fpds.write(str(binfreq[k])+' '+str(binfreq[k+1])+' '+str(pdsbin[k])+' '+str(dpdsbin[k])+"\n")
    fpds.close()
    if(ifplot):
        plots.pdsplot(infile=infile+'_pdstot', omega = [2.*np.pi/0.003, 10733.7], freqrange=[200., 1700.])
        
#
def dynspec_maker(infile='out/lcurve', ntimes = 30, nbins = 150, logbinning = False, fmaxout = False):
    '''
    makes a dynamical spectrum from a light curve
    '''
    lines = np.loadtxt(infile+".dat")
    t = lines[:,0] ; x = lines[:,1]
    nt = np.size(t)
    tmin=t.min() ; tmax=t.max() ; tspan=tmax-tmin
    tbins = np.arange(ntimes+1)/np.double(ntimes)*tspan+tmin
    tcenter = (tbins[1:]+tbins[:-1])/2.
    dt = tspan / np.double(np.size(t))

    freq1 =1./tspan*np.double(ntimes)/2. ; freq2=freq1*np.double(nt)/np.double(ntimes)/2.
    pdsbin=np.zeros([ntimes, nbins]) ; dpdsbin=np.zeros([ntimes, nbins]); nbin = np.zeros([ntimes, nbins])
    # binning:
    if(logbinning):
        binfreq=(freq2/freq1)**((np.arange(nbins+1)/np.double(nbins)))*freq1
    else:
        binfreq=(freq2-freq1)*((np.arange(nbins+1)/np.double(nbins)))+freq1
        
    for k in np.arange(ntimes):
        wtime = np.where((t>tbins[k])*(t<tbins[k+1]))
        t1 = t[wtime] ; x1=x[wtime]
        m,b = np.polyfit(t1, x1, 1)
        #        x1 -= (m*t1+b)
        fsp = 2.*np.fft.rfft(x1-(m*t1+b))/x1.sum() # fractional amplitude
        pds = np.abs(fsp)**2
        freq = np.fft.rfftfreq(np.size(wtime), dt)
        for kb in np.arange(nbins):
            freqrange=(freq>=binfreq[kb])&(freq<binfreq[kb+1])
            pdsbin[k,kb]=pds[freqrange].sum()
            dpdsbin[k,kb]=pds[freqrange].std()
            nbin[k,kb] = (freqrange).sum()
            
    fpds=open(infile+'_pds.dat', 'w')
    fpds.write("# time1 -- time2 -- frequency1 -- frequency2 -- PDS -- dPDS \n")
    for k in np.arange(nbins):
        for kt in np.arange(ntimes):
            fpds.write(str(tbins[kt])+' '+str(tbins[kt+1])+' '+str(binfreq[k])+' '+str(binfreq[k+1])+' '+str(pdsbin[kt,k])+' '+str(dpdsbin[kt,k])+' '+str(nbin[kt,k])+"\n")
    fpds.close() 
    if(ifplot):
        plots.dynsplot(infile=infile+'_pds', omega = [2.*np.pi/0.003, 10733.7])
        # if we want a vdK plot
    if(fmaxout):
        xmeans = np.zeros(ntimes) ; dxmeans = np.zeros(ntimes)
        fmaxs = np.zeros(ntimes) ; dfmaxs = np.zeros(ntimes)
        fout=open(infile+'_ffreq.dat', 'w')
        fout.write("# t -- F -- rms(F) -- freq -- dfreq")
        for k in np.arange(ntimes):
            binfreqc=(binfreq[1:]+binfreq[:-1])/2.
            binfreqs=(binfreq[1:]-binfreq[:-1])/2.
            wmax = (pdsbin[k,:]*binfreqc * (binfreqc> 100.) * (binfreqc < 2000.)).argmax()
            wtime = np.where((t>tbins[k])*(t<tbins[k+1]))
            xmean = x[wtime].mean() ; dxmean = x[wtime].std()
            fout.write(str(tcenter[k])+" "+str(xmean)+" "+str(dxmean)+" "+str(binfreqc[wmax])+" "+str(binfreqs[wmax])+"\n")
            xmeans[k]=xmean ; dxmeans = dxmean
            fmaxs[k] = binfreqc[wmax] ;    dfmaxs[k] = binfreqs[wmax]
        fout.flush()
        fout.close()
        # PL fit:
        wfit = (fmaxs >400.)
        pfit = np.polyfit(np.log(xmeans[wfit]), np.log(fmaxs[wfit]), 1)
        print("PL fit: ln freq = a ln F + b")
        print("a = "+str(pfit[0]))
        print("b = "+str(pfit[1]))
        #        ii=input("!")
        if(ifplot):
            plots.crosses(xmeans, dxmeans*10., fmaxs, dfmaxs,
                          xlabel=r'$L_{\rm obs}$, ${\rm erg\,s^{-1}}$',
                          ylabel=r'$f_{\rm peak}$, Hz', outfilename = infile + '_ffreq')
            
        #
        
####################################################################################################
def meanmaps(filename, n1, n2):
    '''
    makes time(frame)-averaged pictures from hdf5 file filename, frames n1 to n2
    '''
    outdir=os.path.dirname(filename)
    print("writing output in "+outdir)
    f = h5py.File(filename,'r', libver='latest')
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
    uvgdisp=np.zeros([nlats, nlons], dtype=np.double)
    u2vmean=np.zeros([nlats, nlons], dtype=np.double)
    uv2mean=np.zeros([nlats, nlons], dtype=np.double)
  
    qmmean=np.zeros([nlats, nlons], dtype=np.double) ; qpmean=np.zeros([nlats, nlons], dtype=np.double)
    qmstd=np.zeros([nlats, nlons], dtype=np.double) ; qpstd=np.zeros([nlats, nlons], dtype=np.double)
    keys=list(f.keys())
    for k in np.arange(n2-n1)+n1:
        print("reading data entry "+keys[k])
        data=f[keys[k]]
        sig=data["sig"][:] ;    ug=data["ug"][:] ;  vg=data["vg"][:]
        energy=data["energy"][:] ; beta=data["beta"][:]
        qplus=data["qplus"][:] ;  qminus=data["qminus"][:]
        sigmean+=sig ; energymean+=energy ; ugmean+=ug ; vgmean+=vg ; ugdisp+=ug**2 ; vgdisp+=vg**2
        uvcorr += ug * vg
        uvgdisp += ug**2 * vg**2 ; u2vmean += ug**2 * vg ; uv2mean += ug * vg**2
        qmmean += qminus ; qpmean += qplus
        qmstd += qminus**2 ; qpstd += qplus**2
    f.close()
    sigmean/=np.double(n2-n1) ; energymean/=np.double(n2-n1) ; ugmean/=np.double(n2-n1); vgmean/=np.double(n2-n1)
    qmmean/=np.double(n2-n1) ; qpmean/=np.double(n2-n1)
    ugdisp=(ugdisp/np.double(n2-n1)-ugmean**2) ; vgdisp=(vgdisp/np.double(n2-n1)-vgmean**2)
    qmstd = (qmstd/np.double(n2-n1)-qmmean**2) ; qpstd = (qpstd/np.double(n2-n1)-qpmean**2)
    
    uvcorr=uvcorr/np.double(n2-n1)-ugmean*vgmean
    uvgdisp /= np.double(n2-n1) # <u^2v^2>
    u2vmean /= np.double(n2-n1) # <u^2v>
    uv2mean /= np.double(n2-n1) # <uv^2>
    # covariance dispersion
    duvcorr = uvgdisp + 3. * ugmean**2 * vgmean**2 + \
              ugmean**2 * vgdisp + ugdisp * vgmean**2 - \
              2. * ugmean * uv2mean - 2. * u2vmean * vgmean + \
              4. * ugmean * vgmean * uvcorr - uvcorr**2              
    
    print("maximal Reynolds stress "+str(uvcorr.max()))
    print("maximal speed of sound "+str((energymean/sigmean).max()))
    
    # ascii output:
    fout=open(outdir+'/meanmap.dat', 'w')
    fout.write("# lons -- lats -- sig -- energy -- ug -- vg -- Dug -- Dvg -- Cuv\n")
    for kx in np.arange(nlons):
        for ky in np.arange(nlats):
            fout.write(str(lons[ky,kx])+" "+str(lats[ky,kx])+" "+str(sigmean[ky,kx])+" "+str(energymean[ky,kx])+" "+str(ugmean[ky,kx])+" "+str(vgmean[ky,kx])+" "+str(ugdisp[ky,kx])+" "+str(vgdisp[ky,kx])+" "+str(uvcorr[ky,kx])+"\n")
    fout.close()
    if(ifplot):
        plots.somemap(lons, lats, qmstd, outdir+"/std_qminus.png")
        plots.somemap(lons, lats, qpstd, outdir+"/std_qplus.png")
        plots.somemap(lons, lats, qmmean, outdir+"/mean_qminus.png")
        plots.somemap(lons, lats, qpmean, outdir+"/mean_qplus.png")
        plots.somemap(lons, lats, sigmean, outdir+"/mean_sigma.png")
        plots.somemap(lons, lats, energymean, outdir+"/mean_energy.png")
        plots.somemap(lons, lats, uvcorr/np.sqrt(vgdisp*ugdisp), outdir+"/mean_uvcorr.png")
        plots.somemap(lons, lats, (vgdisp-ugdisp)/(vgdisp+ugdisp), outdir+"/mean_anisotropy.png")

    # azimuthal average:
    ulats = lats.mean(axis=1)
    sigmean_phavg = sigmean.mean(axis=1) ; energymean_phavg = energymean.mean(axis=1)
    qmmean_phavg = qmmean.mean(axis=1)  ;  qpmean_phavg = qpmean.mean(axis=1)
    qmstd_phavg =np.sqrt(qmstd.mean(axis=1)+qmmean.std(axis=1)**2)
    qpstd_phavg = np.sqrt(qpstd.mean(axis=1)+qpmean.std(axis=1)**2)
    ugmean_phavg = ugmean.mean(axis=1) ; vgmean_phavg = vgmean.mean(axis=1)
    uvcorr_phavg = uvcorr.mean(axis=1)+(ugmean*vgmean).mean(axis=1)-ugmean_phavg*vgmean_phavg
    duvcorr_phavg = np.sqrt(duvcorr.mean(axis=1)/np.double(n2-n1-1) + uvcorr.std(axis=1)**2/np.double(nlons-1))
    
    csq_phavg = energymean_phavg / sigmean_phavg * 4./9. # for radiation-pressure-dominated case!
    kappa_tmp1 = 2.*ugmean_phavg / np.tan(ulats)/rsphere
    kappa_tmp2 = ugmean_phavg * np.sin(ulats)/rsphere
    kappa = (kappa_tmp1[1:]+kappa_tmp1[:-1])/2. * (kappa_tmp2[1:]-kappa_tmp2[:-1])/(np.cos(ulats)[1:]-np.cos(ulats)[:-1])
    rossby = np.sqrt(csq_phavg[1:-1]/np.abs(kappa[1:]+kappa[:-1]))
    ekappa = np.sqrt(np.abs((kappa[:-1]+kappa[1:])/2.))/tscale/2./np.pi 
    # In general, Rossby radius is sensitive to epicyclic frequency
    # kappa^2 = 2.*Omega * cot(theta) * d/dtheta(Omega sin^2theta)
    aniso_phavg = ((vgdisp-ugdisp)/(vgdisp+ugdisp)).mean(axis=1)
    if(ifplot):
        plots.someplot(ulats, [omega*rsphere*np.sin(ulats), ugmean_phavg, vgmean_phavg], xname=r'$\theta$', yname='$u$, $v$', prefix=outdir+'/uvmeans', title='', postfix='plot', fmt=['k:','r:', 'k-'])
        plots.someplot(ulats, [uvcorr_phavg, -uvcorr_phavg, ugmean_phavg*vgmean_phavg, -ugmean_phavg*vgmean_phavg,  csq_phavg, uvcorr_phavg+duvcorr_phavg, uvcorr_phavg-duvcorr_phavg, -uvcorr_phavg+duvcorr_phavg, -uvcorr_phavg-duvcorr_phavg], xname=r'$\theta$', yname=r'$\langle\Delta u \Delta v\rangle$', prefix=outdir+'/uvcorr', title='', postfix='plot', fmt=['k-', 'k--', 'b-', 'b--', 'r:', 'k:', 'k:', 'k:', 'k:'], ylog=True)
        plots.someplot(ulats, [qmmean_phavg, qpmean_phavg, qmstd_phavg, qpstd_phavg], xname=r'$\theta$', yname="$Q^{\pm}$", prefix=outdir+'/qpm', fmt = ['k-', 'r-', 'k:', 'r:'], ylog=True)
        plots.someplot(ulats[1:-1], [rossby/rsphere, np.abs(ulats[1:-1]-np.pi/2.)], xname=r'$\theta$', yname=r'$R_{\rm Rossby}/R_*$', prefix=outdir+'/ro', fmt=['k-', 'r:'], ylog=True)
        plots.someplot(ulats[1:-1], [ekappa, (ugmean_phavg/rsphere/np.sin(ulats)/2./np.pi)[1:-1]/tscale], xname=r'$\theta$', yname=r'$f$, Hz', prefix=outdir+'/ekappa', fmt=['k-', 'r:'], ylog=True)
    fout=open(outdir+'/meanmap_phavg.dat', 'w')
    fout.write("# lats -- sig -- energy -- ug -- vg -- csq -- Cuv -- dCuv -- aniso\n")
    for k in np.arange(nlats):
        fout.write(str(ulats[k])+" "+str(sigmean_phavg[k])+" "+str(energymean_phavg[k])+" "+str(ugmean_phavg[k])+" "+str(vgmean_phavg[k])+" "+str(csq_phavg[k])+" "+str(uvcorr_phavg[k])+" "+str(duvcorr_phavg[k])+" "+str(aniso_phavg[k])+"\n")
    fout.close()    
    fout=open(outdir+'/meanmap_qphavg.dat', 'w')
    fout.write("# lats -- qmmean -- qpmean -- qmstd -- qpstd\n")
    for k in np.arange(nlats):
        fout.write(str(ulats[k])+" "+str(qmmean_phavg[k])+" "+str(qpmean_phavg[k])+" "+str(qmstd_phavg[k])+" "+str(qpstd_phavg[k])+"\n")
    fout.close()
    fout=open(outdir+'/meanmap_ro.dat', 'w')
    fout.write("# lats -- Ro/R* -- kappa \n")
    for k in np.arange(nlats-2)+1:
        fout.write(str(ulats[k])+" "+str(rossby[k-1]/rsphere)+" "+str(ekappa[k-1])+"\n")
    fout.close()
    
