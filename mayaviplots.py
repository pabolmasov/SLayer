import numpy as np
import mayavi
from mayavi import mlab
import glob
import os

def mayout(infile):
    # make a quasi-3D sphere image
    lines = np.loadtxt(infile, comments="#", delimiter=" ", unpack=False)
    lats = lines[:,0] ; lons = lines[:,1] ; sig = lines[:,2] ; vort = lines[:,4] ; qminus = lines[:,8]
    theta = np.pi/2. - lats  ; phi = lons

    # make theta, phi 2D (again)
    thun = np.unique(theta) ; phun = np.unique(phi)
    nth = np.size(thun) ; nph = np.size(phun)
    theta = np.reshape(theta, [nth, nph])
    phi = np.reshape(phi, [nth, nph])
    sig = np.reshape(sig, [nth, nph])
    vort = np.reshape(vort, [nth, nph])
    qminus = np.reshape(qminus, [nth, nph])

    x =  np.sin(theta) * np.cos(phi) 
    y =  np.sin(theta) * np.sin(phi) 
    z =  np.cos(theta) 

    mlab.options.offscreen = True
    fig = mlab.figure(1, fgcolor=(1, 1, 1), bgcolor=(0, 0, 0), size = (100, 100))
    # surface density plot:
    mlab.clf()
    mlab.mesh(x, y, z, scalars=sig, colormap='hot', representation='wireframe',
              resolution=1, line_width = 10., scale_factor = 1., mode = 'point' )
    mlab.view(50., 40., -5.0)
    # mlab.show()
    mlab.savefig(infile+'_maysig.png', size = (200, 200))
    #    mlab.show()
    mlab.close()
    fig = mlab.figure(1, fgcolor=(1, 1, 1), bgcolor=(0, 0, 0), size = (100, 100))
    mlab.clf()
    mlab.mesh(x, y, z, scalars=qminus, colormap='hot', representation='wireframe',
              resolution=1, line_width = 10., scale_factor = 1., mode = 'point' )
    mlab.view(50., 40., -5.0)
    # mlab.show()
    mlab.savefig(infile+'_mayqm.png', size = (200, 200))
    #    mlab.show()
    mlab.close(all=True)

def multimaya(prefix, skip=0, step=1):
    
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
        mayout(flist[k])
        os.system("mv "+flist[k]+"_maysig.png"+" "+outdir+'/maysig{:05d}'.format(k)+".png")
        os.system("mv "+flist[k]+"_mayqm.png"+" "+outdir+'/mayqm{:05d}'.format(k)+".png")
