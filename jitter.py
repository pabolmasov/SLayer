from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.interpolate as si

kpower = 1

# jitter block
def jitterturn(vort0, div0, sig0, energy0, accflag0, dlon, grid, grid1): #, back=False):
    '''
    turns all the functions on the grid by an arbitrary angle in dlon
    '''
    vort_spline = si.RectBivariateSpline(-grid.lats, grid.lons, vort0, kx=kpower, ky=kpower,
                                         bbox=[-np.pi/2., np.pi/2., 0., 2.*np.pi])
    div_spline = si.RectBivariateSpline(-grid.lats, grid.lons, div0, kx=kpower, ky=kpower,
                                         bbox=[-np.pi/2., np.pi/2., 0., 2.*np.pi])
    sig_spline = si.RectBivariateSpline(-grid.lats, grid.lons, sig0, kx=kpower, ky=kpower,
                                         bbox=[-np.pi/2., np.pi/2., 0., 2.*np.pi])
    energy_spline = si.RectBivariateSpline(-grid.lats, grid.lons, energy0, kx=kpower, ky=kpower,
                                         bbox=[-np.pi/2., np.pi/2., 0., 2.*np.pi])
    accflag_spline = si.RectBivariateSpline(-grid.lats, grid.lons, accflag0, kx=kpower, ky=kpower,
                                         bbox=[-np.pi/2., np.pi/2., 0., 2.*np.pi])
    #    si.interp2d(grid.lons, grid.lats, vort, kind='cubic')
    xlons,xlats = np.meshgrid(grid1.lons, grid1.lats)

    vort1 = vort_spline.ev(-xlats, (xlons+dlon) % (2.*np.pi))
    div1 = div_spline.ev(-xlats, (xlons+dlon) % (2.*np.pi))
    sig1 = sig_spline.ev(-xlats, (xlons+dlon) % (2.*np.pi))
    energy1 = energy_spline.ev(-xlats, (xlons+dlon) % (2.*np.pi))
    accflag1 = accflag_spline.ev(-xlats, (xlons+dlon) % (2.*np.pi))
    
    return vort1, div1, sig1, energy1, accflag1

def jitternod(vort0, div0, sig0, energy0, accflag0, incl, grid, grid1):
    '''
    turns all the functions on the grid by an arbitrary inclination angle incl
    '''
    #    print(np.shape(grid.lons))
    #    print(np.shape(vort[:,::-1]))
    vort_spline = si.RectBivariateSpline(-grid.lats, grid.lons, vort0, kx=kpower, ky=kpower,
                                         bbox=[-np.pi/2., np.pi/2., 0., 2.*np.pi])
    div_spline = si.RectBivariateSpline(-grid.lats, grid.lons, div0, kx=kpower, ky=kpower,
                                         bbox=[-np.pi/2., np.pi/2., 0., 2.*np.pi])
    sig_spline = si.RectBivariateSpline(-grid.lats, grid.lons, sig0, kx=kpower, ky=kpower,
                                         bbox=[-np.pi/2., np.pi/2., 0., 2.*np.pi])
    energy_spline = si.RectBivariateSpline(-grid.lats, grid.lons, energy0, kx=kpower, ky=kpower,
                                         bbox=[-np.pi/2., np.pi/2., 0., 2.*np.pi])
    accflag_spline = si.RectBivariateSpline(-grid.lats, grid.lons, accflag0, kx=kpower, ky=kpower,
                                         bbox=[-np.pi/2., np.pi/2., 0., 2.*np.pi])

    xlons,xlats = np.meshgrid(grid1.lons, grid1.lats)
    lats1 = np.arcsin(np.sin(xlats)*np.cos(incl)-np.cos(xlats)*np.sin(incl)*np.sin(xlons))
    clons1 = np.cos(xlons) * np.cos(xlats)/np.cos(lats1)
    slons1 = (np.sin(xlats) - np.sin(lats1)*np.cos(incl)) / np.sin(incl) / np.cos(lats1)
    slons1 = np.maximum(np.minimum(slons1, 1.), -1.)
    lons1 = np.arcsin(slons1)
    lons1[clons1 < 0.] = np.pi - lons1[clons1 < 0.]
    lons1 = lons1 % (2.*np.pi)
    #    print(slons1.min())
    #    print(slons1.max())
    #    input("j")
    #    lons1 = np.abs(lons1)*np.sign(np.sin(xlons))
    #    lons1 = xlons # !!! 
    
    vort1 = vort_spline.ev(-lats1, lons1)
    div1 = div_spline.ev(-lats1, lons1)
    sig1 = sig_spline.ev(-lats1, lons1)
    energy1 = energy_spline.ev(-lats1, lons1)
    accflag1 = accflag_spline.ev(-lats1, lons1)
    
    return vort1, div1, sig1, energy1, accflag1
