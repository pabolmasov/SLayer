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

    bmin=1e-3 ; bmax=1.-bmin ; nb=1000
    smin=1e5 ; smax=1e10 ; ns=29
    beta1d=(bmax/bmin)**(arange(nb)/double(nb-1))*bmin
    #    beta1d=beta1d/(1.-beta1d)
    sig1d=((smax/smin))**(arange(ns)/double(ns-1))*smin
    beta, sig=meshgrid(beta1d, sig1d)
    
    bcom=beta/(1.-beta)**0.25/(1.-(beta/2.))
    # speed of sound E/Sigma c^2
    etos=8.7e-6/mu*(grav*mass1*sig)**0.25/bcom
    rho=1.87066*mu*beta/(1.-beta)**0.25*(grav*sig)**0.75
    gammae=0.006347*(beta/(1.-beta))**(1./3.)
    eden = 1.38772 * beta/(1.-beta)**0.25 *(grav*sig)**0.75 # in 1e24 cm^{-3}

    ef = 7.13593e-05 * eden**(2./3.) # in me*c**2
    kt = etos
    
    fluxratio = 1.80904e-06/sqrt(etos)*eden * (maximum(ef, kt)/ kt)**1.5
    
    plt.clf()
    plt.contourf(beta, sig, log10(rho))
    plt.colorbar()
    plt.contour(beta, sig, ef/kt, levels=[1.], colors='r')
    plt.contour(beta, sig, fluxratio, levels=[1.], colors='k')
    plt.contour(beta, sig, gammae, levels=[1.,10.,100.], colors='w', linestyles='dotted')
    #    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\Sigma,\, {\rm \,g\,cm^{-2}}$')
    plt.savefig('allview.eps')
    plt.savefig('allview.png')
