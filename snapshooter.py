from __future__ import print_function
from __future__ import division
from builtins import str
import numpy as np
import shtns
import scipy.ndimage as nd
import time
from spharmt import Spharmt 
import os
import h5py

'''
Post-processing and various post-factum diagnostic outputs
'''
from conf import ifplot, rsphere, sigmascale, mass1

if(ifplot):
    import plots

def keyshow(filename):
    '''
    showing the list of keys (entries) in a given data file
    '''
    f = h5py.File(filename,'r', libver='latest')
    keys = list(f.keys())
    #    print(list(f.keys()))
    f.close()
    return keys

def plotnth(filename, nstep, derot = False, step = 1):
    '''
    plot a given time step of a given data file. To list the available nsteps (integer values), use keyshow(filename).
    If "ifplot" is off, makes an ascii map instead (useful for remote calculations)
    derot keyword compensates for NS rotation
    '''
    global rsphere
    outdir=os.path.dirname(filename)
    f = h5py.File(filename,'r', libver='latest')
    params=f["params"]
    nlons=params.attrs["nlons"] ; nlats=params.attrs["nlats"] ; omega=params.attrs["omega"] ; tscale=params.attrs["tscale"]
    x = Spharmt(int(nlons),int(nlats),int(np.double(nlons)/3.),rsphere,gridtype='gaussian')
    lons1d = x.lons # (2.*np.pi/np.double(nlons))*np.arange(nlons)
    clats1d = np.sin(x.lats) # 2.*np.arange(nlats)/np.double(nlats)-1.
    slats1d = np.cos(x.lats) # 2.*np.arange(nlats)/np.double(nlats)-1.
    lons,lats = np.meshgrid(lons1d, x.lats)
    rsphere=params.attrs["rsphere"] ; grav=params.attrs["grav"] # ; kappa=params.attrs["kappa"]
    omegaNS=params.attrs["omega"] ;   tscale=params.attrs["tscale"]
    data=f["cycle_"+str(nstep).rjust(6, '0')]
    vortg=data["vortg"][:] ; divg=data["divg"][:] ; ug=data["ug"][:] ; vg=data["vg"][:] ; t=data.attrs["t"]
    sig=data["sig"][:] ; energy=data["energy"][:] ; beta=data["beta"][:] ; diss=data["diss"][:] ; accflag=data["accflag"][:]
    qminus=data["qminus"][:]
    nth,nphi=np.shape(lats)
    if(derot):
        #        print("lons from "+str(lons.min())+" to "+str(lons.max()))
        lons = (lons-omegaNS * t) % (2.*np.pi)
        lonsort = lons[0,:].argsort()
        ug -= rsphere * omegaNS * np.cos(lats)
        vortg -= 2.* omegaNS * np.sin(lats)
        for kth in np.arange(nth):
            lons[kth, :] = lons[kth, lonsort]
            sig[kth, :] = sig[kth, lonsort]
            ug[kth, :] = ug[kth, lonsort] ;            vg[kth, :] = vg[kth, lonsort]
            divg[kth, :] = divg[kth, lonsort] ; vortg[kth, :] = vortg[kth, lonsort]
            energy[kth, :] = energy[kth, lonsort] ; qminus[kth, :] = qminus[kth, lonsort]
            accflag[kth, :] = accflag[kth, lonsort]
    lonsDeg=lons*180./np.pi ; latsDeg=lats*180./np.pi
    f.close()
    press=energy* 3. * (1.-beta/2.)
    # ascii output:
    fmap=open(filename+'_map'+str(nstep)+'.dat', 'w')
    fmap.write("# map with step = "+str(step)+"\n")
    fmap.write("# t="+str(t*tscale)+"\n")
    fmap.write("# format: lats lons sigma digv vortg ug vg E Q- accflag\n")
    for kth in np.arange(0,nth, step):
        for kphi in np.arange(0,nphi, step):
            fmap.write(str(lats[kth,kphi])+" "+str(lons[kth,kphi])+" "+str(sig[kth,kphi])+" "
                       +str(divg[kth,kphi])+" "+str(vortg[kth,kphi])+" "
                       +str(ug[kth,kphi])+" "+str(vg[kth,kphi])+" "
                       +str(energy[kth,kphi])+" "+str(qminus[kth,kphi])+" "
                       +str(accflag[kth,kphi])+"\n")
    fmap.flush()
    fmap.close()
    if(ifplot):
        # velocity
        xx=ug ; yy=-vg
        xxmean=xx.mean(axis=1) ;    yymean=yy.mean(axis=1)
        sigmean=sig.mean(axis=1)
        sig1=np.zeros(sig.shape, dtype=np.double)
        for k in np.arange(nlons):
            sig1[:,k]=sig[:,k] # -sigmean[:]
            #        xx[:,k]-=xxmean[:]
            #        yy[:,k]-=yymean[:]
        vv=np.sqrt(xx**2+yy**2)
        vvmax=vv.max()
        skx = 8 ; sky=8 # we do not need to output every point; these are the steps for the output in two dimensions
        xx = nd.filters.gaussian_filter(xx, np.double(skx)/2., mode='constant')*500./vvmax
        yy = nd.filters.gaussian_filter(yy, np.double(sky)/2., mode='constant')*500./vvmax
        tbottom=339.6*((1.-beta)*sig*(sigmascale/1e8)/mass1/rsphere**2)**0.25
        # (50.59*((1.-beta)*energy*sigmascale/mass1)**0.25)
        teff=(qminus*sigmascale/mass1)**0.25*3.64 # effective temperature in keV
        # vortg-2.*omegaNS*np.sin(lats)
        #        print(teff.min(), teff.max())
        plots.snapplot(lonsDeg, latsDeg, sig, accflag, teff, xx, yy, [skx,sky], outdir=outdir, t=t*tscale*1e3) # geographic maps
        # plots.snapplot(lonsDeg, latsDeg, sig, accflag, qminus, xx, yy, [skx,sky], outdir=outdir) # geographic maps
        gamma=4./3.
        j=ug*rsphere*np.cos(lats)
        geff=1./rsphere**2-(ug**2+vg**2)/rsphere
        hvert = 5./np.abs(geff) * press/sig # vertical thickness
        s=np.log(press/sig)-(1.-1./gamma)*np.log(geff)
        sgrad1, sgrad2 = x.getGrad(x.grid2sph(s))
        jgrad1, jgrad2 = x.getGrad(x.grid2sph(j))
        ograd1, ograd2 = x.getGrad(x.grid2sph(np.log(ug/rsphere/np.cos(lats))))
        kappasq = -2.*ug*np.sin(lats)/np.cos(lats)**2*jgrad2
        nsq = (ug/rsphere)**2*sgrad2
        laminst=(2.*np.pi/np.abs(ograd2)).mean(axis=1)
        dlaminst=(2.*np.pi/np.abs(ograd2)).std(axis=1)
        rcircle=2.*np.pi*rsphere*np.cos(x.lats)
        
        plots.someplot(x.lats*180./np.pi, [laminst, rcircle],
                       xname='latitude, deg', yname=r'$\lambda_{\rm KH}$', prefix=outdir+'/KH',
                       fmt=['k.', 'r-'], title='$t = {:6.2f}$\,ms'.format( t*tscale*1e3))
        plots.someplot(lats, [j], xname='lats', yname='$j$', prefix=outdir+'/jacc')
        plots.someplot(lats, [hvert*1.47676*mass1], xname='lats', yname='$H$, km', prefix=outdir+'/hvert', ylog=(hvert.max() > (hvert.min()*10.)))
        plots.someplot(lats, [kappasq + nsq, -(kappasq+nsq), kappasq, nsq], xname='lats',
                       yname=r'$\varkappa^2+N^2$',
                       prefix=outdir+'/kappa', ylog=True, fmt=['k,', 'g,', 'b,', 'r,'])

# multiple diagnostic maps for making movies
def multireader(infile, nrange = None, nframes = None, derot = False, step = 5):

    keys = keyshow(infile)
    print(keys)
    nsize=np.size(keys)-1 # last key contains parameters
    if(nrange == None):
        nmin = 1 ; nmax = nsize
    else:
        nmin, nmax = nrange
    nmax-=1
    outdir=os.path.dirname(infile)
    ndigits=np.long(np.ceil(np.log10(nmax))) # number of digits

    if(nframes == None):
        frames = np.linspace(nmin, nmax, dtype=int)
    else:
        frames =  np.linspace(nmin, nmax, nframes, dtype=int)
    
    for k in frames:
        plotnth(infile, k, derot = derot, step = step)
        if(ifplot):
            os.system('cp '+outdir+'/snapshot.png '+outdir+'/shot'+str(k).rjust(ndigits, '0')+'.png')
            os.system('cp '+outdir+'/polemap_north.png '+outdir+'/north'+str(k).rjust(ndigits, '0')+'.png')
            os.system('cp '+outdir+'/polemap_south.png '+outdir+'/south'+str(k).rjust(ndigits, '0')+'.png')
        print('shot'+str(k).rjust(ndigits, '0'))
