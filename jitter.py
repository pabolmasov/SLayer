from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.interpolate as si

# jitter block
def jitterturn(vort, div, sig, energy, dlon, grid, grid1): #, back=False):
    '''
    turns all the functions on the grid by an arbitrary angle in dlon
    '''
    vortfun =  si.interp2d(grid.lons, grid.lats, vort, kind='cubic')
    divfun =  si.interp2d(grid.lons, grid.lats, div, kind='cubic')
    sigfun =  si.interp2d(grid.lons, grid.lats, sig, kind='cubic')
    energyfun =  si.interp2d(grid.lons, grid.lats, energy, kind='cubic')
    vort1 = vortfun(grid1.lons, grid1.lats)
    div1 = divfun(grid1.lons, grid1.lats)
    sig1 = sigfun(grid1.lons, grid1.lats)
    energy1 = energyfun(grid1.lons, grid1.lats)
    return vort1, div1, sig1, energy1

def jitternod(vort0, div0, sig0, energy0, incl, grid, grid1):
    '''
    turns all the functions on the grid by an arbitrary inclination angle incl
    '''
    #    print(np.shape(grid.lons))
    #    print(np.shape(vort[:,::-1]))
    vort_spline = si.RectBivariateSpline(grid.lats[::-1], grid.lons, vort0[:,::-1], kx=3, ky=3,
                                         bbox=[-np.pi/2., np.pi/2., 0., 2.*np.pi])
    div_spline = si.RectBivariateSpline(grid.lats[::-1], grid.lons, div0[:,::-1], kx=3, ky=3,
                                         bbox=[-np.pi/2., np.pi/2., 0., 2.*np.pi])
    sig_spline = si.RectBivariateSpline(grid.lats[::-1], grid.lons, sig0[:,::-1], kx=3, ky=3,
                                         bbox=[-np.pi/2., np.pi/2., 0., 2.*np.pi])
    energy_spline = si.RectBivariateSpline(grid.lats[::-1], grid.lons, energy0[:,::-1], kx=3, ky=3,
                                         bbox=[-np.pi/2., np.pi/2., 0., 2.*np.pi])

    xlons,xlats = np.meshgrid(grid1.lons, grid1.lats)
    lats1 = np.arcsin(np.sin(xlats)*np.cos(incl)-np.cos(xlats)*np.sin(incl)*np.sin(xlons))
    clons1 = np.cos(xlons) * np.cos(xlats)/np.cos(lats1)
    slons1 = (np.sin(xlats) - np.sin(lats1)*np.cos(incl)) / np.sin(incl) / np.cos(lats1)
    slons1 = np.maximum(np.minimum(slons1, 1.),-1.)
    lons1 = np.arcsin(slons1)
    lons1[clons1 < 0.] = np.pi - lons1[clons1 < 0.]
    lons1 = lons1 % (2.*np.pi)
    #    print(slons1.min())
    #    print(slons1.max())
    #    input("j")
    #    lons1 = np.abs(lons1)*np.sign(np.sin(xlons))
    #    lons1 = xlons # !!! 
    
    vort1 = vort_spline.ev(lats1, lons1)
    div1 = div_spline.ev(lats1, lons1)
    sig1 = sig_spline.ev(lats1, lons1)
    energy1 = energy_spline.ev(lats1, lons1)
    
    return vort1, div1, sig1, energy1
