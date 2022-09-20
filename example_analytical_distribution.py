"""
This is the example of generating a synthetic population of the interstellar object
using the Statistical method (Marceta, D.: Orbital Model of the interstellar Objects in the Solar System, 2022.)

This code uses Maxwell-Boltzmann distribution of interstellar velocities of ISOs. This is just one example, the code
can be used with any other distribution of interstellar velocities.

Here are given two examples. First one generates only orbital elements of ISOs, while the second one generates also 
their sizes according to the broken power law.

input:
    v - interstellar velocities of the ISOs (m/s)
    pv - distribution of the interstellar velocities of ISOs given in the array "v" (Maxwell_Boltzmann distribution with sigma=26140 m/s (Eubanks et al. 2021))
    n0 - interstellar number-density of ISOs (objects per cubic au)
    rm - radius of the model sphere (au)
    dr - step for the numerical integration with respect to heliocentric distance (au)

output:
    q - perihelion distance (au)
    e - eccentricity
    f - true anomaly (degrees)
    inc - orbital inclination (degrees)
    Omega - longitude of ascending node (degrees)
    omega - argument of perihelion (degrees)  
"""
import numpy as np
from statistical_method import maxwell_boltzmann, synthetic_population

v=np.arange(1e3, 101e3, 1e3) # interstellar velocities of the ISOs (unperturbed by the Sun's gravity)
pv=maxwell_boltzmann(v, sigma=26140.) # distribution of the interstellar velocities

# generating synthetic population without generating the sizes of objects
q,e,f,inc,Omega,omega,D=synthetic_population(v,pv,n0=0.1, rm=50, d=[], alpha=[], dr=1e-3)

## generating synthetic population with defined size-frequency distribution
#q,e,f,inc,Omega,omega, D=synthetic_population(v,pv,n0=1e-1, rm=10, d=[50,500, 2000], alpha=[-1.5, -2.1], dr=1e-3)


