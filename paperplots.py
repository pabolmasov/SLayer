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

# compound and special plots for the paper

def twotwists():

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
    
