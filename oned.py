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

    mdot1=0. ; mdot2=10. ; nmdot=5
    mdot=(mdot2-mdot1)*(np.arange(nmdot)/np.double(nmdot-1))+mdot1
    Kcoeff=1.-overkepler**2
    th1=0. ; th2=np.pi/2. ;  nth=100
    th=np.arange(nth)/np.double(nth)*(th2-th1)+th1
    costh=np.cos(th) ; sinth=np.sin(th)
    cmap = matplotlib.cm.get_cmap('jet')
    plt.clf()
    fig=plt.figure()
    sub1=plt.subplot(211)
    sub2=plt.subplot(212)
    for k in np.arange(nmdot):
        v=np.sqrt((1.-Kcoeff*np.exp(2./3.*mdot[k]/rsphere*costh))/rsphere)
        omega=(costh+(mdot[k]/3.*sinth**2-costh)*(1.-rsphere*v**2))/v/rsphere**2/sinth
        sub1.plot(th, v*np.sqrt(rsphere), color=cmap(np.double(k)/np.double(nmdot)),label='$\dot{m}='+str(mdot[k])+'$')
        sub2.plot(th, omega*rsphere**1.5*sinth, color=cmap(np.double(k)/np.double(nmdot)))
    sub1.legend()
    sub1.set_ylabel(r'$v/v_{\rm K}$')
    sub2.set_ylabel(r'$\omega\sin\theta/\Omega_{\rm K} $')
    sub2.set_xlabel(r'$\theta$, rad')
    plt.savefig('oned.png')
