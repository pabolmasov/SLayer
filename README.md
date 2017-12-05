# SLayer: the shallow-water two-dimensional code simulating a spreading layer
on the surface of a spherical gravitating object. The code uses
spherical harmonics to compute
atmospheric circulations. Based on https://github.com/natj/swm, but the
variables are normalized to the natural relativistic units GM=c=1, for a
neutron star rotating at a millisecond-range period.
We are either using an isothermal equation of state &Pi; = cs^2 &Sigma;, or
solving an energy equation. Details of the physics and solution methods are
given in the [description](https://github.com/pabolmasov/swarm/blob/master/slayer_art/swslayer.pdf). 

## Basic equations:

Mass conservation equation (for surface density &Sigma;):

https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20%5CSigma%7D%7B%5Cpartial%20t%7D%20%3D%20-%20%5Cnabla%20%5Ccdot%20%28%5CSigma%20%5Cmathbf%7Bv%7D%29%20&plus;%20S%5E&plus;%20-%20S%5E-%2C

where the right-hand side also contains source and sink terms. 

Euler equations re-written in terms of divergence &delta; and vorticity
&omega; :

<img alt="\frac{\partial \delta}{\partial t} = \nabla \cdot [\mathbf{v} \times
\omega \mathbf{e}^r] - \nabla^2 \left(\frac{v^2}{2}\right) + \nabla \cdot
\left( \frac{1}{\Sigma} \nabla \Pi \right) + D\delta,"
src="https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20%5Cdelta%7D%7B%5Cpartial%20t%7D%20%3D%20%5Cnabla%20%5Ccdot%20%5B%5Cmathbf%7Bv%7D%20%5Ctimes%20%5Comega%20%5Cmathbf%7Be%7D%5Er%5D%20-%20%5Cnabla%5E2%20%5Cleft%28%5Cfrac%7Bv%5E2%7D%7B2%7D%5Cright%29%20&amp;plus;%20%5Cnabla%20%5Ccdot%20%5Cleft%28%20%5Cfrac%7B1%7D%7B%5CSigma%7D%20%5Cnabla%20%5CPi%20%5Cright%29%20&amp;plus;%20D%5Cdelta%2C">,

<img alt="\frac{\partial \omega}{\partial t} = -\nabla \cdot (\omega \mathbf{v}) + \frac{7}{8}\left[\nabla \Pi \times \nabla \Sigma \right]_r + \frac{S^+}{\Sigma} \omega_{\rm d} + D\omega," src="https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20%5Comega%7D%7B%5Cpartial%20t%7D%20%3D%20-%5Cnabla%20%5Ccdot%20%28%5Comega%20%5Cmathbf%7Bv%7D%29%20&amp;plus;%20%5Cfrac%7B7%7D%7B8%7D%5Cleft%5B%5Cnabla%20%5CPi%20%5Ctimes%20%5Cnabla%20%5CSigma%20%5Cright%5D_r%20&amp;plus;%20%5Cfrac%7BS%5E&amp;plus;%7D%7B%5CSigma%7D%20%5Comega_%7B%5Crm%20d%7D%20&amp;plus;%20D%5Comega%2C">

So far, there is no friction term, and the accreted matter retains the mean
angular momentum of the source.

Finally, energy equation re-formulated as an equation for vertically
integrated pressure &Pi; :

<img alt="\frac{\partial \Pi}{\partial t} + \nabla \cdot \left( \Pi
\mathbf{v}\right) = \frac{1}{3\left(1-\frac{\beta}{2}\right)} \delta \Pi + Q^+
- Q^- + Q_{\rm NS},"
src="https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20%5CPi%7D%7B%5Cpartial%20t%7D%20&amp;plus;%20%5Cnabla%20%5Ccdot%20%5Cleft%28%20%5CPi%20%5Cmathbf%7Bv%7D%5Cright%29%20%3D%20%5Cfrac%7B1%7D%7B3%5Cleft%281-%5Cfrac%7B%5Cbeta%7D%7B2%7D%5Cright%29%7D%20%5Cdelta%20%5CPi%20&amp;plus;%20Q%5E&amp;plus;%20-%20Q%5E-%20&amp;plus;%20Q_%7B%5Crm%20NS%7D%2C">.

Here, &beta; is the gas to total pressure ratio, Q-terms in the right-hand
side are sources and sinks of heat. 

## Installation

Spherical harmonic transformation library [shtns](https://bitbucket.org/nschaeff/shtns)

```
./configure --enable-python
make
make install
```

h5py package; see h5py [website](http://docs.h5py.org/en/latest/index.html)

Usually it is enough to use pip with
```
pip2 install h5py
```

All the parameters of the problem are listed in the configuration file
[conf.c](https://github.com/pabolmasov/swarm/blob/master/conf.py). The code
may be simply run from python as
```
%run swarm
```

To restart an interrupted simulation, change the ifrestart, nrest, and
restartfile parameters in
[conf.c](https://github.com/pabolmasov/swarm/blob/master/conf.py), and make
sure the restart file name corresponds to the hdf5 output you are going to
read from (probably, you will need to rename your 'out/run.hdf5' to 'out/runOLD.hdf5').

## References

*  "non-linear barotropically unstable shallow water test case"
  example provided by Jeffrey Whitaker
  https://gist.github.com/jswhit/3845307

*  Galewsky et al (2004, Tellus, 56A, 429-440)
  "An initial-value problem for testing numerical models of the global
  shallow-water equations" DOI: 10.1111/j.1600-0870.2004.00071.x
  http://www-vortex.mcs.st-and.ac.uk/~rks/reprints/galewsky_etal_tellus_2004.pdf
  
*  shtns/examples/shallow_water.py

*  Jakob-Chien et al. 1995:
  "Spectral Transform Solutions to the Shallow Water Test Set"

