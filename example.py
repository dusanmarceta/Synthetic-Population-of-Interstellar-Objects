"""
This is the example of generating a synthetic population of interstellar objects using the Statistical method 
(Marceta, D.: Synthetic population of interstellar objects in the Solar System, 2022)

This code can use an analytically defined distribution of interstellar velocities as well as a distribution defined in discrete form.
The example below shows both of these cases. The three most commonly used interstellar velocity distributions are defined in the file 
"analytical_distributions.py". One distribution in discrete form is defined in the file "discrete_distribution.txt". 
These distributions are for demonstration purposes only, the code can be used with any other distribution.

In addition to orbits, the code can also generate the sizes of ISOs, according to a (broken) power law. 
Below are two examples, one where only orbits are generated, as well as the example that generates both orbits and ISO sizes.

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
    D - diameters (m) (optional)
"""

import numpy as np
from analytical_distributions import maxwell_boltzmann
from synthetic_population import synthetic_population

# Defining input distribution of interstellar velocities

# Case 1: defining the range of interstellar velocities (v) and their analytically defined distribution (pv)
#v=np.arange(1e3, 101e3, 1e3) # interstellar velocities of the ISOs (unperturbed by the Sun's gravity)
#pv=maxwell_boltzmann(v, sigma=26140.) # distribution of the interstellar velocities

# Case 2: reading the discrete distribution from the input file
v, pv=np.loadtxt('discrete_distribution.txt', delimiter=',', skiprows=1, unpack=True)


# Generating synthetic population

# Case 1: generating only orbits without generating the sizes of objects
q, e, f, inc, Omega, omega, D = synthetic_population(v, pv, rm=50, n0=0.1)

# Case 2: generating both orbits and sizes
# In this case, the code will generate objects with sizes ranging from 50 to 2000 m. The referent number density is 1e-1 per cubic au for 
# objects larger than 1000 m. Objects within the size range [100, 500) have SFD slope of -1.5, 
# while the object within the size range [500, 2000] have slope of -2.1.
#q, e, f, inc, Omega, omega, D = synthetic_population(v, pv, rm=10, n0=1e-1, d_ref=1000, d=[50, 500, 2000], alpha=[-1.5, -2.1])


