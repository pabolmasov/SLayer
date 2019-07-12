import numpy as np
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

print(s.shape)
mlab.mesh(x, y, z, scalars=s, colormap='inferno')

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
        0.0, 0.0, 1.7)

ar1.length_cone=0.4
ar1.color = (0,0,0)


#--------------------------------------------------








mlab.savefig("sphere_mayavi.png")
#mlab.savefig("sphere_mayavi.pdf")
