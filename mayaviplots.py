import numpy as np
import mayavi
from mayavi import mlab
import glob
import os

from conf import rsphere


def mayout(infile, iobs = 45., phi0=0.):
    # make a quasi-3D sphere image
    lines = np.loadtxt(infile, comments="#", delimiter=" ", unpack=False)
    lats = lines[:,0] ; lons = lines[:,1] ; sig = lines[:,2] ; vort = lines[:,4]
    ug = lines[:,5] ; vg = lines[:,6] ; energy = lines[:,7] ; qminus = lines[:,8]
    theta = np.pi/2. - lats  ; phi = lons

    # make theta, phi 2D (again)
    thun = np.unique(theta) ; phun = np.unique(phi)
    nth = np.size(thun) ; nph = np.size(phun)
    theta = np.reshape(theta, [nth, nph])
    phi = np.reshape(phi, [nth, nph])
    sig = np.reshape(sig, [nth, nph])
    ug = np.reshape(ug, [nth, nph])  ;   vg = np.reshape(vg, [nth, nph])
    vort = np.reshape(vort, [nth, nph])
    qminus = np.reshape(qminus, [nth, nph])
    energy = np.reshape(energy, [nth, nph])

    x =  np.sin(theta) * np.cos(phi*1.01) 
    y =  np.sin(theta) * np.sin(phi*1.01) 
    z =  np.cos(theta) 

    # [xmin, xmax, ymin, ymax, zmin, zmax] 
    extent = np.array([-1., 1., -1., 1., -1., 1.] )
    
    mlab.options.offscreen = True
    fig = mlab.figure(1, fgcolor=(1, 1, 1), bgcolor=(0, 0, 0), size = (200, 200))
    # surface density plot:
    mlab.clf()
    mlab.mesh(x, y, z, scalars=sig, colormap='hot', representation='surface', extent = extent,
              resolution=1, line_width = 10., scale_factor = 1., mode = 'point' )
    mlab.view(elevation = iobs, azimuth = phi0, distance = 6.)
    mlab.savefig(infile+'_maysig.png', size = (200, 200))
    #    mlab.show()
    # flux map:
    fig = mlab.figure(1, fgcolor=(1, 1, 1), bgcolor=(0, 0, 0), size = (200, 200))
    mlab.clf()
    mlab.mesh(x, y, z, scalars=qminus, colormap='hot', representation='surface', extent = extent,
              resolution=1, line_width = 10., scale_factor = 1., mode = 'point' )
    mlab.view(elevation = iobs, azimuth = phi0, distance = 6.)
    mlab.savefig(infile+'_mayqm.png', size = (200, 200))
    #    mlab.show()
    #    mlab.close(all=True)
    # vorticity map:
    fig = mlab.figure(1, fgcolor=(1, 1, 1), bgcolor=(0, 0, 0), size = (200, 200))
    mlab.clf()
    mlab.mesh(x, y, z, scalars=vort, colormap='hot', representation='surface', extent = extent,
              resolution=1, line_width = 10., scale_factor = 1., mode = 'point' )
    mlab.view(elevation = iobs, azimuth = phi0, distance = 6.)
    mlab.savefig(infile+'_mayv.png', size = (200, 200))

    cutaway = (phi >= 5./6. * np.pi) & (phi < 5./6. * np.pi )

    geff = 1./rsphere**2 - (ug**2+vg**2)/rsphere
    beta = 0. # temporary!
    press = energy / 3./ (1.-beta/2.)
    height = 5. * press / sig / geff
    sigmascale = 1e8 ; mass1 = 1.4
    teff = (qminus*sigmascale/mass1)**0.25*3.64
    tbottom = 339.6*sig**0.25 # *((1.-beta)*sig*(sigmascale/1e8)/mass1/rsphere**2)**0.25
    
    xx =  (rsphere + height) * np.sin(theta) * np.cos(phi) 
    yy =  (rsphere + height) * np.sin(theta) * np.sin(phi) 
    zz =  (rsphere + height) * np.cos(theta) 

    x *= rsphere * 0.99 ; y *= rsphere * 0.99 ; z *= rsphere * 0.99
    
    print("Teff = "+str(teff.min())+".."+str(teff.max()))
    
    mlab.clf()
    fig = mlab.figure(bgcolor = (1.,1.,1.))
    mlab.mesh(xx, yy, zz, colormap="hot", mask = cutaway, scalars = teff,
              representation='surface', opacity = 1., line_width = 1., scale_factor=0.1)
    mlab.mesh(x, y, z, colormap='GnBu', mask = cutaway, 
              representation='surface', opacity = 1., line_width = 1., scale_factor=0.1)
    mlab.view(elevation = iobs, azimuth = phi0, distance = 6.*rsphere, focalpoint = (0.,0.,0.))
    mlab.savefig(infile+'_hsurface.png', magnification=3)
    mlab.savefig('hsurface.png', magnification=3)
    #    mlab.savefig('hsurface.eps', magnification=3)
    #    mlab.show()
    mlab.close(all=True)

def multimaya(prefix, skip=0, step=1, iobs=45., phi0 = 90.):
    
    flist0 = np.sort(glob.glob(prefix+"[0-9].dat"))
    flist1 = np.sort(glob.glob(prefix+"[0-9][0-9].dat"))
    flist2 = np.sort(glob.glob(prefix+"[0-9][0-9][0-9].dat"))
    flist3 = np.sort(glob.glob(prefix+"[0-9][0-9][0-9][0-9].dat"))
    flist4 = np.sort(glob.glob(prefix+"[0-9][0-9][0-9][0-9][0-9].dat"))
    flist = np.concatenate((flist0, flist1, flist2, flist3, flist4))
    nlist = np.size(flist)
    print(flist)
    outdir=os.path.dirname(prefix)
    
    for k in np.arange(skip, nlist, step):
        mayout(flist[k], iobs = iobs, phi0=phi0)
        os.system("mv "+flist[k]+"_maysig.png"+" "+outdir+'/maysig{:05d}'.format(k)+".png")
        os.system("mv "+flist[k]+"_mayqm.png"+" "+outdir+'/mayqm{:05d}'.format(k)+".png")
        os.system("mv "+flist[k]+"_mayv.png"+" "+outdir+'/mayv{:05d}'.format(k)+".png")
        os.system("mv "+flist[k]+"_hsurface.png"+" "+outdir+'/mayH{:05d}'.format(k)+".png")
        
    mlab.close(all=True)

def mayabatch():
    ddir = 'titania/out_8LR/'
    prefix = ddir + 'run.hdf5_map'
    multimaya(prefix, iobs=0.)
    os.system("ffmpeg -f image2 -r 15 -pattern_type glob -i \'"+ddir+"mayv*.png\' -pix_fmt yuv420p -b 4096k "+ddir+"mayv_iobs0.mp4")
    os.system("ffmpeg -f image2 -r 15 -pattern_type glob -i \'"+ddir+"maysig*.png\' -pix_fmt yuv420p -b 4096k "+ddir+"maysig_iobs0.mp4")
    os.system("ffmpeg -f image2 -r 15 -pattern_type glob -i \'"+ddir+"mayqm*.png\' -pix_fmt yuv420p -b 4096k "+ddir+"mayqm_iobs0.mp4")
    multimaya(prefix, iobs=45.)
    os.system("ffmpeg -f image2 -r 15 -pattern_type glob -i \'"+ddir+"mayv*.png\' -pix_fmt yuv420p -b 4096k "+ddir+"mayv_iobs45.mp4")
    os.system("ffmpeg -f image2 -r 15 -pattern_type glob -i \'"+ddir+"maysig*.png\' -pix_fmt yuv420p -b 4096k "+ddir+"maysig_iobs45.mp4")
    os.system("ffmpeg -f image2 -r 15 -pattern_type glob -i \'"+ddir+"mayqm*.png\' -pix_fmt yuv420p -b 4096k "+ddir+"mayqm_iobs45.mp4")
    multimaya(prefix, iobs=90.)
    os.system("ffmpeg -f image2 -r 15 -pattern_type glob -i \'"+ddir+"mayv*.png\' -pix_fmt yuv420p -b 4096k "+ddir+"mayv_iobs90.mp4")
    os.system("ffmpeg -f image2 -r 15 -pattern_type glob -i \'"+ddir+"maysig*.png\' -pix_fmt yuv420p -b 4096k "+ddir+"maysig_iobs90.mp4")
    os.system("ffmpeg -f image2 -r 15 -pattern_type glob -i \'"+ddir+"mayqm*.png\' -pix_fmt yuv420p -b 4096k "+ddir+"mayqm_iobs90.mp4")

# mayout('titania/out_8LR4/runcombine.hdf5_map12006.dat', iobs=45., phi0 = 0.)
# mayout('titania/out_8LR4/runcombine.hdf5_map13003.dat', iobs=45., phi0 = 0.)
# mayout('titania/out_8LR4/runcombine.hdf5_map14000.dat', iobs=45., phi0 = 0.)
# mayout('titania/out_3HR/run.hdf5_map1224.dat', iobs=90., phi0 = 90.)

    
# ffmpeg -f image2 -r 35 -pattern_type glob -i 'titania/out_8LR4/mayv*.png' -pix_fmt yuv420p -b 4096k titania/out_8LR4/mayv.mp4
# ffmpeg -f image2 -r 35 -pattern_type glob -i 'titania/out_3HR/mayH*.png' -pix_fmt yuv420p -b 4096k titania/out_3HR/mayH.mp4
# ffmpeg -f image2 -r 35 -pattern_type glob -i 'titania/out_3LRinc/mayH*.png' -pix_fmt yuv420p -b 4096k titania/out_3LRinc/mayH.mp4
# ffmpeg -f image2 -r 35 -pattern_type glob -i 'titania/out_off/mayH*.png' -pix_fmt yuv420p -b 4096k titania/out_off/mayH.mp4
