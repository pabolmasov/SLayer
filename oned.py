from __future__ import division
from builtins import str
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import time
import pylab

#proper LaTeX support and decent fonts in figures 
rc('font',**{'family':'serif','serif':['Times']})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

from conf import rsphere, overkepler


def levisol():
    '''
    old model without viscous energy transfer
    '''
    mdot1=0. ; mdot2=10. ; nmdot=5
    mdot=(mdot2-mdot1)*(np.arange(nmdot)/np.double(nmdot-1))+mdot1
    mdot = [0.2, 1., 5.]
    nmdot=np.size(mdot)
    Kcoeff=1.-overkepler**2
    th1=0. ; th2=(np.pi/2.) ;  nth=100
    th=np.arange(nth)/np.double(nth)*(th2-th1)+th1
    costh=np.cos(th) ; sinth=np.sin(th)
    cmap = matplotlib.cm.get_cmap('hot')
    plt.clf()
    fig=plt.figure()
    sub1=plt.subplot(211)
    sub2=plt.subplot(212)
    for k in np.arange(nmdot):
        v=np.sqrt(((1.-Kcoeff*np.exp(2./3.*mdot[k]/rsphere*costh))/rsphere))
        omega=(costh+(mdot[k]/3.*sinth**2-costh)*(1.-rsphere*v**2))/v/rsphere**2/sinth
        sub1.plot(th, v*np.sqrt(rsphere), color=cmap(np.double(k)/np.double(nmdot)),label='$\dot{m}='+str(mdot[k])+'$')
        sub2.plot(th, omega*rsphere**1.5*sinth, color=cmap(np.double(k)/np.double(nmdot)))
    #    sub1.legend()
    sub1.set_ylabel(r'$v/v_{\rm K}$')
    sub2.set_ylabel(r'$\omega\sin\theta/\Omega_{\rm K} $')
    sub2.set_xlabel(r'$\theta$, rad')
    plt.savefig('oned.eps')
    sub1.legend()
    plt.savefig('oned.png')

###########################################################
def omegastream_dth(sthsq, omega, mdot):
    return -sthsq/mdot * (1.-omega**2 * sthsq)/(1.-omega * sthsq)

def omegastream_N(sthsq, omega, mdot):
    return 1.-omega**2 * sthsq

def omegastream_D(sthsq, omega, mdot, omega0 = 1.):
    return omega0-omega * sthsq

def omegastream(mdot = 10.):

    th1 = np.pi/4. ; th2= np.pi/2. ; nth = 200
    o1 = 0.25 ; o2 = 1.25 ; no = 201

    theta = (np.arange(nth)+0.5)/np.double(nth)*(th2-th1)+th1
    omega = (o2-o1)*(np.arange(no)/np.double(no-1))+o1
    theta2, omega2 = np.meshgrid(theta, omega)

    print(theta2)
    omega0 = 0.8
    
    sthsq = np.sin(theta2)**2
    dome_N = omegastream_N(sthsq, omega2, mdot)
    dome_D = omegastream_D(sthsq, omega2, mdot, omega0=omega0)

    plt.clf()
    fig=plt.figure()
    plt.plot((np.pi/2.-theta)*180./np.pi, 1./np.sin(theta), color='r')
    plt.plot((np.pi/2.-theta)*180./np.pi, omega0/np.sin(theta)**2, color='r')
    plt.streamplot((np.pi/2.-theta2)*180./np.pi, omega2, dome_D, -sthsq*rsphere*dome_N/mdot * np.pi/180., color='k',integration_direction='backward')
    #    plt.plot((np.pi/2.-theta)*180./np.pi, omega0 - np.sqrt(2.*(1.-omega0**2)*((np.pi/2.-theta)/2.+0.25*np.sin(theta*2.))/mdot), color='g')
    plt.ylim(omega.min(),omega.max())
    #    plt.yscale('log')
    plt.xlabel(r'latitude, deg', fontsize=20) ; plt.ylabel(r'$w$', fontsize=16)
    plt.tick_params(labelsize=14, length=3, width=1., which='minor')
    plt.tick_params(labelsize=14, length=6, width=2., which='major')
    fig.set_size_inches(6, 5)
    fig.tight_layout()
    plt.savefig('dome.png')
    plt.savefig('dome.eps')
    plt.close('all')
