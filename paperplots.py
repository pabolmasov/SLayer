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
from matplotlib import interactive

import glob

from conf import rsphere, tscale
import plots

plt.ioff()
# compound and special plots for the paper

def twotwists():
    '''
    picture for the split-sphere test
    '''
    file1 = "titania/out_stwistLR/ecurve1.57079632679.dat"
    file2 = "titania/out_stwistHR/ecurve1.57079632679.dat"
    lines1 = np.loadtxt(file1, comments="#", delimiter=" ", unpack=False)
    lines2 = np.loadtxt(file2, comments="#", delimiter=" ", unpack=False)

    tar1=lines1[:,0] ; tar2=lines2[:,0]
    ev1=lines1[:,4] ; ev2=lines2[:,4]
    eth1=lines1[:,2] ; eth2=lines2[:,2]
    eu1=lines1[:,3] ; eu2=lines2[:,3]

    rsphere=6.04606
    twistscale=0.2
    pspin=0.01
    dvdr = 2.*np.pi/pspin
    
    plt.clf()
    fig=plt.figure()
    plt.plot(tar1, ev1, 'k:')
    plt.plot(tar2, ev2, 'k')
    plt.plot(tar1, eu1, 'r:')
    plt.plot(tar2, eu2, 'r')
    plt.plot(tar1, np.exp(0.67*(tar1-0.025)*dvdr), 'b--')
    plt.yscale('log')
    plt.xlabel('$t$, s', fontsize=18)
    plt.ylim(ev1[ev1>0.].min()+ev2[ev2>0.].min(), (eth1).max()+eth2.max())
    plt.ylabel('$E$, $10^{35}$erg', fontsize=18)
    plt.tick_params(labelsize=16, length=3, width=1., which='minor')
    plt.tick_params(labelsize=16, length=6, width=2., which='major')
    fig.set_size_inches(5, 4)
    fig.tight_layout()
    plt.savefig('forpaper/twostwists.png')
    plt.savefig('forpaper/twostwists.eps')
    plt.close()
    
def twoND():
    '''
    error growth for the no-accretion, rigid-body test (NDLR, NDHR)
    '''
    file1='titania/out_NDLR/rtest.dat'
    file2='titania/out_NDHR/rtest.dat'
    lines1 = np.loadtxt(file1, comments="#", delimiter=" ", unpack=False)
    lines2 = np.loadtxt(file2, comments="#", delimiter=" ", unpack=False)
    
    tar1=lines1[:,0] ; err1=lines1[:,1] ; serr1=lines1[:,2]
    tar2=lines2[:,0] ; err2=lines2[:,1] ; serr2=lines2[:,2]

    plt.clf()
    fig=plt.figure()
    plt.subplot(121)
    plt.plot(tar1, err1, 'k')
    plt.plot(tar2, err2, 'r')
    plt.yscale('log')
    plt.tick_params(labelsize=16, length=3, width=1., which='minor')
    plt.tick_params(labelsize=16, length=6, width=2., which='major')
    plt.ylabel('random error, $\Delta \Sigma/\Sigma$', fontsize=18)
    plt.xlabel('$t$, s', fontsize=18)
    plt.subplot(122)
    plt.plot(tar1, abs(serr1), 'k')
    plt.plot(tar2, abs(serr2), 'r')
    plt.plot([tar1[0]+0.1,tar1[0]+0.13], [1e-8, 1e-8], 'b')
    plt.yscale('log')
    plt.ylabel('systematic error, $\Delta \Sigma/\Sigma$', fontsize=16)
    plt.xlabel('$t$, s', fontsize=16)
    plt.tick_params(labelsize=14, length=3, width=1., which='minor')
    plt.tick_params(labelsize=14, length=6, width=2., which='major')
    fig.set_size_inches(5, 8)
    plt.tight_layout()
    plt.savefig('rtests.eps')
    plt.savefig('rtests.png')
    plt.close()
#
def threecurves(outdir = "titania/out_3LR/"):
    '''
    three light curves for the whistler plot
    '''
    
    file1 = outdir+"lcurve0.0.dat"
    file2 = outdir+"lcurve0.785398163397.dat"
    file3 = outdir+"lcurve1.57079632679.dat"
    lines = np.loadtxt(file1)
    t1=lines[:,0] ; l1=lines[:,1]
    lines = np.loadtxt(file2)
    t2=lines[:,0] ; l2=lines[:,1]
    lines = np.loadtxt(file3)
    t3=lines[:,0] ; l3=lines[:,1]

    plt.clf()
    fig = plt.figure()
    plt.plot(t1, l1, 'k-')
    plt.plot(t2, l2, 'g--')
    plt.plot(t3, l3, 'b:')    
    #    plt.yscale('log')
    plt.xlim(t1.min(), t1.max())
    plt.ylabel(r'$L_{\rm obs}$, $10^{37}{\rm \, erg \, s^{-1}}$', fontsize=20)
    plt.xlabel('$t$, s', fontsize=20)
    plt.tick_params(labelsize=18, length=3, width=1., which='minor')
    plt.tick_params(labelsize=18, length=6, width=2., which='major')
    #    plt.yscale('log')
    fig.set_size_inches(12, 4)
    fig.tight_layout()
    plt.savefig(outdir+'forpaper_3lc.png')
    plt.savefig(outdir+'forpaper_3lc.eps')
    plt.close()

#
def ekappa():
    '''
    plots epicyclic and rotation velocities from meanmap longitudinally-averaged data
    '''
    omega = 2.*np.pi/0.003 
    outdir = "titania/out_3LR/"
    infile = outdir + "meanmap_ro.dat"
    lines = np.loadtxt(infile)
    theta=lines[:,0] ; fe=lines[:,2]
    ugfile = outdir + "meanmap_phavg.dat"
    lines = np.loadtxt(ugfile)
    oloc=(lines[:,3])[1:-1]/rsphere/np.sin(theta)/tscale

    latDeg = 180./np.pi*(np.pi/2.-theta)
    plt.clf()
    fig = plt.figure()
    plt.plot(latDeg, fe, 'k-')
    plt.plot(latDeg, oloc/2./np.pi, 'r:')
    plt.plot(latDeg, oloc*0.+omega/2./np.pi, 'g--')
    plt.plot(latDeg, oloc*0.+2.*omega/2./np.pi, 'g--')
    plt.xlabel(r'latitude, deg', fontsize=16)
    plt.ylabel(r'$\varkappa_{\rm e}/2\pi$, Hz', fontsize=16)
    plt.tick_params(labelsize=14, length=3, width=1., which='minor')
    plt.tick_params(labelsize=14, length=6, width=2., which='major')
    fig.set_size_inches(6, 5)
    fig.tight_layout()
    plt.savefig("forpaper/ekappa.png")
    plt.savefig("forpaper/ekappa.eps")
    plt.close()
#
def fourPDS(outdir = '/home/pasha/SLayer/titania/out_8LR4s/'):
    infile1 = outdir + 'lcurve0.0_pdstot.dat'
    lines1 = np.loadtxt(infile1, unpack=True)
    freqstart1=lines1[0,:] ; freqend1=lines1[1,:]
    flux1=lines1[2,:] ; dflux1=lines1[3,:]

    infile2 = outdir + 'lcurve0.785398163397_pdstot.dat'
    lines2 = np.loadtxt(infile2, unpack=True)
    freqstart2=lines2[0,:] ; freqend2=lines2[1,:]
    flux2=lines2[2,:] ; dflux2=lines2[3,:]
   
    infile3 = outdir + 'lcurve1.57079632679_pdstot.dat'
    lines3 = np.loadtxt(infile3, unpack=True)
    freqstart3=lines3[0,:] ; freqend3=lines3[1,:]
    flux3=lines3[2,:] ; dflux3=lines3[3,:]

    infile4 = outdir + 'lcurve3.14159265359_pdstot.dat'
    lines4 = np.loadtxt(infile4, unpack=True)
    freqstart4=lines4[0,:] ; freqend4=lines4[1,:]
    flux4=lines4[2,:] ; dflux4=lines4[3,:]

    xlabel = r'$f$, Hz' ; ylabel = r'fractional PDS'
    
    freqmin = 200. ; freqmax = 1500.
    win = np.where((freqstart1> freqmin) * (freqend1 < freqmax))
    
    plt.clf()
    fig=plt.figure()
    plt.plot([1./0.003, 1./0.003], [flux4.min(), flux1.max()], color='gray')
    plt.errorbar((freqstart1+freqend1)/2., flux1, xerr=(-freqstart1+freqend1)/2., yerr=dflux1, fmt='ko')
    plt.errorbar((freqstart2+freqend2)/2., flux2, xerr=(-freqstart2+freqend2)/2., yerr=dflux2, fmt='rd')
    plt.errorbar((freqstart3+freqend3)/2., flux3, xerr=(-freqstart3+freqend3)/2., yerr=dflux3, fmt='g^')
    plt.errorbar((freqstart4+freqend4)/2., flux4, xerr=(-freqstart4+freqend4)/2., yerr=dflux4, fmt='bv')
    plt.xlim(freqmin, freqmax) ; plt.ylim(flux4[win].min()*0.5, flux1[win].max()*2.)
    plt.xlabel(xlabel, fontsize=18) ; plt.ylabel(ylabel, fontsize=18)
    plt.tick_params(labelsize=16, length=3, width=1., which='minor')
    plt.tick_params(labelsize=16, length=6, width=2., which='major')
    plt.yscale('log')
    fig.set_size_inches(6, 5)
    fig.tight_layout()
    plt.savefig(outdir+'pds4.png')
    plt.savefig(outdir+'pds4.eps')
    plt.close()
#
def threePDS(outdir = '/home/pasha/SLayer/titania/out_3LR/'):
    infile1 = outdir + 'lcurve0.0_pdstot.dat'
    lines1 = np.loadtxt(infile1, unpack=True)
    freqstart1=lines1[0,:] ; freqend1=lines1[1,:]
    flux1=lines1[2,:] ; dflux1=lines1[3,:]

    infile2 = outdir + 'lcurve0.785398163397_pdstot.dat'
    lines2 = np.loadtxt(infile2, unpack=True)
    freqstart2=lines2[0,:] ; freqend2=lines2[1,:]
    flux2=lines2[2,:] ; dflux2=lines2[3,:]
   
    infile3 = outdir + 'lcurve1.57079632679_pdstot.dat'
    lines3 = np.loadtxt(infile3, unpack=True)
    freqstart3=lines3[0,:] ; freqend3=lines3[1,:]
    flux3=lines3[2,:] ; dflux3=lines3[3,:]

    xlabel = r'$f$, Hz' ; ylabel = r'fractional PDS'
    
    freqmin = 200. ; freqmax = 1500.
    win = np.where((freqstart1> freqmin) * (freqend1 < freqmax))
    
    plt.clf()
    fig=plt.figure()
    plt.plot([1./0.003, 1./0.003], [flux1.min(), flux1.max()], color='b')
    plt.errorbar((freqstart1+freqend1)/2., flux1, xerr=(-freqstart1+freqend1)/2., yerr=dflux1, fmt='ko')
    plt.errorbar((freqstart2+freqend2)/2., flux2, xerr=(-freqstart2+freqend2)/2., yerr=dflux2, fmt='rd')
    plt.errorbar((freqstart3+freqend3)/2., flux3, xerr=(-freqstart3+freqend3)/2., yerr=dflux3, fmt='g^')
    plt.xlim(freqmin,freqmax) ; plt.ylim(flux1[win].min()*0.5, flux1[win].max()*10.)
    plt.xlabel(xlabel, fontsize=18) ; plt.ylabel(ylabel, fontsize=18)
    plt.tick_params(labelsize=16, length=3, width=1., which='minor')
    plt.tick_params(labelsize=16, length=6, width=2., which='major')
    plt.yscale('log')
    fig.set_size_inches(6, 5)
    fig.tight_layout()
    plt.savefig(outdir+'pds3.png')
    plt.savefig(outdir+'pds3.eps')
    plt.close()
#
def threecrosses(outdir = '/home/pasha/SLayer/titania/out_3LR/'):
    
    infile1 = outdir + 'lcurve0.0_ffreq.dat'
    lines1 = np.loadtxt(infile1, unpack=True)
    print(np.shape(lines1))
    flux1 = lines1[1, :] ; dflux1 = lines1[2, :] ; freq1 = lines1[3,:]; dfreq1 = lines1[4,:]
    infile2 = outdir + 'lcurve0.785398163397_ffreq.dat'
    lines2 = np.loadtxt(infile2, unpack=True)
    flux2 = lines2[1, :] ; dflux2 = lines2[2, :] ; freq2 = lines2[3,:]; dfreq2 = lines2[4,:]
    infile3 = outdir + 'lcurve1.57079632679_ffreq.dat'
    lines3 = np.loadtxt(infile3, unpack=True)
    flux3 = lines3[1, :] ; dflux3 = lines3[2, :] ; freq3 = lines3[3,:]; dfreq3 = lines3[4,:]

    xlabel=r'$L_{\rm obs}$, $10^{37}{\rm erg\,s^{-1}}$' ;   ylabel=r'$f_{\rm peak}$, Hz'
    plt.clf()
    fig=plt.figure()
    plt.plot([flux1.min(), flux1.max()], [1./0.003, 1./0.003], 'g:')
    plt.errorbar(flux1, freq1, xerr=dflux1, yerr=dfreq1, fmt='ko')
    plt.errorbar(flux2, freq2, xerr=dflux2, yerr=dfreq2, fmt='rd')
    plt.errorbar(flux3, freq3, xerr=dflux3, yerr=dfreq3, fmt='b^')
    plt.xlabel(xlabel, fontsize=20) ; plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(labelsize=18, length=3, width=1., which='minor')
    plt.tick_params(labelsize=18, length=6, width=2., which='major')
    plt.ylim(0.,1000.) ;    plt.xlim(0.,4.)
    fig.set_size_inches(6, 5)
    fig.tight_layout()
    plt.savefig(outdir+'ffreq3.png')
    plt.savefig(outdir+'ffreq3.eps')
    plt.close()

def qpmplot(outdir = 'titania/out_3LR/'):
    
    lats, qmmean, qpmean, qmstd, qpstd = np.loadtxt(outdir+'meanmap_qphavg.dat', unpack=True)
    latsDeg = 180./np.pi * (np.pi/2.-lats)
    
    plots.someplot(latsDeg, [qmmean/qmmean.max(), qmstd/qmmean.max()], xname=r'latitude, deg', yname="$Q^{-}$", prefix=outdir+'qminus', fmt = ['k-', 'r:'], ylog=False)
    plots.someplot(latsDeg, [qpmean/qpmean.max(), qpstd/qpmean.max()], xname=r'latitude, deg', yname="$Q^{+}$", prefix=outdir+'qplus', fmt = ['k-', 'r:'], ylog=False)
    
#############################################################
def teffsigma():
    '''
    a plot showing the physical processes setting an upper limit for surface density
    '''
    mass1 = 1.5 # solar masses
    rstar6 = 1.2 # radius in 10^6 cm

    s1 = 1e5 ; s2 = 1e11 ; ns=100
    sigma = (s2/s1)**(np.arange(ns)/np.double(ns-1))*s1 # surface density
    teffscale = 2.11602e7 /(mass1 * rstar6 ** 2 )**0.25
    t1 = 1e6 ; t2 = teffscale ; nt=101
    teff = (t2/t1)**(np.arange(nt)/np.double(nt-1))*t1 # effective temperature

    sigma2, teff2 = np.meshgrid(sigma, teff)

    # pressure ratio:
    beta = 1.-(teff2/teffscale)**4
    
    # internal temp:
    tint = 1.51462e9 * ((sigma2/1e8) * mass1/rstar6**2 * (1.-beta))**0.25
    # density:
    rho = 53079.9 * ((sigma2/1e8) * mass1/rstar6**2)**0.75 * beta / (1.-beta)**0.25
    # Fermi temperature:
    TF = 6.05411e8 / np.sqrt(mass1/rstar6**2) * np.sqrt(sigma2/1e8)
    # electrons become relativistic:
    Trele = 5.92986e9 
    # Coulomb coupling parameter:
    GammaC = (1.77359e5/tint) * (rho/1.)**(1./3.)

    print("maximal Coulomb coupling "+str(GammaC.max()))
    
    plt.clf()
    plt.contourf(sigma2, teff2, np.log10(rho))
    plt.colorbar()
    cs = plt.contour(sigma2, teff2, np.log10(tint), levels=np.arange(10), linestyles='dotted', colors='k')
    plt.clabel(cs, fmt = r'$10^{%d}$\,K', fontsize=18)
    plt.contour(sigma2, teff2, GammaC, colors='k', levels=[1.])
    plt.contour(sigma2, teff2, TF/tint, colors='w', levels=[1.])
    plt.contour(sigma2, teff2, tint/Trele, colors='r', levels=[1.])
    plt.xlabel(r'$\Sigma,\ {\rm g \, cm^{-2}}$')
    plt.ylabel(r'$T_{\rm eff}$, K')
    plt.xscale('log') ; plt.yscale('log')
    plt.savefig('limits.png')
    plt.close()

