from __future__ import division
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import *
import time
import pylab
import h5py

from conf import mu, grav, mass1

#proper LaTeX support and decent fonts in figures 
rc('font',**{'family':'serif','serif':['Times']})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

def allview():

    bmin=1e-5 ; bmax=1e5 ; nb=100
    smin=1. ; smax=1e10 ; ns=29
    beta1d=(bmax/bmin)**(arange(nb)/double(nb-1))*bmin
    beta1d=beta1d/(1.-beta1d)
    sig1d=((smax/smin))**(np.arange(ns)/np.double(ns-1))*smin
    beta, sig=meshgrid(beta1d, sig1d)
    
    bcom=beta/(1.-beta)**0.25/(1.-(beta/2.))
    # speed of sound E/Sigma c^2
    etos=8.7e-6/mu*(grav*mass1*sig)**0.25/bcom
    rho=1.87066*mu*beta/(1.-beta)**0.25*(grav*sig)**0.75
    gammae=0.006347*(beta/(1.-beta))**(1./3.)
    
    plt.clf()
    plt.contourf(beta, sig, etos, nlevels=100)
    plt.contour(beta, sig, levels=[1.], colors='k')
    plt.contour(beta, sig, gammae, levels=[1.,10.,100.], colors='w', linestyles='dotted')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\Sigma {\rm \,g\,cm^{-2}}$')
    plt.savefig('allview.eps')
