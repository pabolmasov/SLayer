# SLayer: the shallow-water two-dimensional code simulating a spreading layer

Shallow-water hydrodynamic simulations on the surface of a spherical gravitating object. The code uses spherical harmonics to compute atmospheric circulations. Hydrodynamic solver is tweaked to deal with highly supersonic flows present in plasma flows on top of rapidly rotating neutron stars.

Based on https://github.com/natj/swm, but the variables are normalized to the natural relativistic units GM=c=1, for a neutron star rotating at a millisecond-range period. We are either using an isothermal equation of state &Pi; = cs^2 &Sigma;, or solving an energy equation. Details of the physics and solution methods are given in the [description](https://github.com/pabolmasov/swarm/blob/master/slayer_art/swslayer.pdf). 


## Basic equations:

See out paper (coming soon!)

## Installation

### shtns
First we need to install the spherical harmonic transformation library [shtns](https://bitbucket.org/nschaeff/shtns).

After downloading/cloning the repository run:
```
./configure --enable-python
make
make install
```

### h5py

Code uses hdf5 to store simulation and analysis data. For this we need `h5py` python package. See h5py [website](http://docs.h5py.org/en/latest/index.html)

Usually it is enough to use pip with
```
pip2 install h5py
```
(or pip3, if you use python3)

### future

To be used equally well with both python2 and python3, the code makes use of the rutines of python-future routines. Probably, you will need to install it with pip
```
pip2 install future
```

## Code

All the parameters of the problem are listed in the configuration file [conf.c](https://github.com/pabolmasov/swarm/blob/master/conf.py). The code may be simply run from python as
```
%run swarm
```
alternatively, you can use your own configuration file (where all the parameters should be set, there are no default values), and run the code as
```
%run swarm <your_conf>
```

To restart an interrupted simulation, change the `ifrestart`, `nrest`, and `restartfile` parameters in [conf.c](https://github.com/pabolmasov/swarm/blob/master/conf.py), and make sure the restart file name corresponds to the hdf5 output you are going to read from (probably, you will need to rename your `out/run.hdf5` to `out/runOLD.hdf5`).


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

