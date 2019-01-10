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

import glob

plt.ioff()
# compound and special plots for the paper

def twotwists():
    '''
    picture for the split-sphere test
    '''
    file1 = "out_twist/ecurve1.5707963267948966.dat"
    file2 = "titania/out_twist/ecurve1.57079632679.dat"
    lines1 = np.loadtxt(file1, comments="#", delimiter=" ", unpack=False)
    lines2 = np.loadtxt(file2, comments="#", delimiter=" ", unpack=False)

    tar1=lines1[:,0] ; tar2=lines2[:,0]
    ev1=lines1[:,4] ; ev2=lines2[:,4]
    eth1=lines1[:,2] ; eth2=lines2[:,2]
    eu1=lines1[:,3] ; eu2=lines2[:,3]

    rsphere=6.04606
    twistscale=0.2
    pspin=0.03
    dvdr = 2.*np.pi/pspin
    
    plt.clf()
    fig=plt.figure()
    plt.plot(tar1, ev1, 'k:')
    plt.plot(tar2, ev2, 'k')
    plt.plot(tar1, eu1, 'r:')
    plt.plot(tar2, eu2, 'r')
    plt.plot(tar1, np.exp((tar1-0.05)*dvdr), 'b--')
    plt.yscale('log')
    plt.xlabel('$t$, s')
    plt.ylim(ev1[ev1>0.].min()+ev2[ev2>0.].min(), (eth1).max()+eth2.max())
    plt.ylabel('$E$, $10^{35}$erg')
    fig.set_size_inches(5, 4)
    plt.savefig('twotwists.png')
    plt.savefig('twotwists.eps')
    plt.close()
    
def twoND():
    '''
    error growth for the no-accretion, rigid-body test (NDLR, NDHR)
    '''
    file1='out_NDLR/rtest.dat'
    file2='titania/out_NDHR/rtest.dat'
    lines1 = np.loadtxt(file1, comments="#", delimiter=" ", unpack=False)
    lines2 = np.loadtxt(file2, comments="#", delimiter=" ", unpack=False)
    
    tar1=lines1[:,0] ; err1=lines1[:,1] ; serr1=lines1[:,2]
    tar2=lines2[:,0] ; err2=lines2[:,1] ; serr2=lines2[:,2]

    plt.clf()
    fig=plt.figure()
    plt.subplot(211)
    plt.plot(tar1, err1, '.k')
    plt.plot(tar2, err2, '.r')
    plt.ylabel('random error, $\Delta \Sigma/\Sigma$')
    plt.subplot(212)
    plt.plot(tar1, serr1, 'k')
    plt.plot(tar2, serr2, 'r')
    plt.ylabel('systematic error, $\Delta \Sigma/\Sigma$')
    plt.xlabel('$t$, s')
    fig.set_size_inches(4, 8)
    plt.tight_layout()
    plt.savefig('rtests.eps')
    plt.close()
    
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
