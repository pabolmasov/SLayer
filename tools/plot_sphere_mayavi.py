import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from scipy.special import sph_harm

from tvtk.tools import visual


#--------------------------------------------------
# Create a sphere
r = 1.0
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j]

x = r * sin(phi) * cos(theta)
y = r * sin(phi) * sin(theta)
z = r * cos(phi)

fig = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(300, 300))
mlab.clf()

m = 10
n = 10
s = sph_harm(m, n, theta, phi).real

print("sph shape", s.shape)
#mlab.mesh(x, y, z, scalars=s, colormap='inferno')


#--------------------------------------------------

img = plt.imread("surf_D.png")

# define a grid matching the map size, subsample along with pixels
#theta = np.linspace(0, np.pi, img.shape[0])
#phi   = np.linspace(0, 2*np.pi, img.shape[1])

count = 101 # keep 180 points along theta and phi
theta_inds = np.linspace(0, img.shape[0] - 1, count).round().astype(int)
phi_inds = np.linspace(0, img.shape[1] - 1, count).round().astype(int)
#theta = theta[theta_inds]
#phi = phi[phi_inds]
img = img[np.ix_(theta_inds, phi_inds)]

print("img shape", img.shape)
img = np.sqrt( img[:,:,0]**2 + img[:,:,1]**2 + img[:,:,2]**2 )
mlab.mesh(x, y, z, scalars=img[:,:], colormap='inferno')


#mlab.view(90, 70, 6.2, (-1.3, -2.9, 0.25))
#mlab.show()

#--------------------------------------------------
#plot rotation arrow

def arrow_from_A_to_B(visual, x1, y1, z1, x2, y2, z2):
    ar1=visual.arrow(x=x1, y=y1, z=z1)
    ar1.length_cone=0.4 #bypass tvtk bug with this line

    arrow_length=np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    ar1.actor.scale=[arrow_length, arrow_length, arrow_length]
    ar1.pos = ar1.pos/arrow_length
    ar1.axis = [x2-x1, y2-y1, z2-z1]

    return ar1


visual.set_viewer(fig)
ar1 = arrow_from_A_to_B(visual, 
        0.0, 0.0, 1.0, 
        0.0, 0.0, 1.8)

ar1.length_cone=0.4
ar1.color = (0,0,0)


#--------------------------------------------------
# plot inflowing matter from disk with ballerina skirt of arrows

#sph coordinates to cartesian
def sph2xyz(r, phi, theta):
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    return x,y,z

thetav = np.pi/2.0
r_inner = 1.3
r_outer = 1.8

for phiv in np.linspace(0, 2.0*np.pi, 50):
    x1,y1,z1 = sph2xyz(r_outer, phiv, thetav)
    x2,y2,z2 = sph2xyz(r_inner, phiv, thetav)
    
    print("phi {}".format(phiv))
    print(" x {} to {} \n y {} to {} \n z {} to {})".format(x1,x2,y1,y2,z1,z2))

    ar = arrow_from_A_to_B(visual, 
            x1,y1,z1,
            x2,y2,z2)

    #ar.length_cone=0.2
    ar.trait_set(opacity=0.1, color=(1,0,0))



#--------------------------------------------------

mlab.view(azimuth=180-40, elevation=60, distance=7.0)

mlab.savefig("sphere_mayavi.png")
#mlab.savefig("sphere_mayavi.pdf")
