# Synthetic-population-of-Interstellar-Obejcts

    Author:      Dusan Marceta
    Institution: University of Belgrade
    Email:       dmarceta@matf.bg.ac.rs
    Date:        September 2022
    Version:     1.0

Required python libraries:

    numpy, scipy, random, tqdm

This code can be used for generating a synthetic population of interstellar objects (orbits and sizes) in arbitrary volume of space around the Sun. The only necessary assumption is that the population of ISOs in the interstellar space (far from any massive body) is homogenous and isotropic. The assumed distribution of interstellar velocities of ISOs has to be provided as an input. This distribution can be defined analytically, but also in a discrete form. The example file "example.py" which demonstrates generation of a synthetic population (for both of these cases) is provided.
-----------------------------------------------------------------------------------------------------------------------------------------------------------

GENERATING SIZES

Beside orbits, the code can optionally generate sizes (diameters) of ISOs,  if required via input parameters. The sizes are generated according to (broken) power law for the size-frequency distribution (SFD). In order to generate sizes, the input parameters must be defined appropriately. This is explained through the following example:

    d_ref=1000 - reference diameter for which the number density is defined (m)
    n0=1e-2 - number of objects per cubic au whose diameters are larger than d0
    d=[100, 500, 10000] - critical diameters of the population
    alpha=[-2, -3] - slopes of the broken power law


In this example, the code will generate objects with sizes ranging from 100 to 10000 m (d[0] to d[-1]). The referent number density (n0) is 1e-2 per cubic au for objects larger than 1000 m (d_ref).
Objects within the size range [100, 500) have SFD slope of -2, while the object within the size range [500, 10000] have slope of -3.

The code will first calculate the total number-density for all objects inside the defined size-range. After that, it will calculate their sizes acording to the defined SFD. 

If the parameters d and alpha are not defined when calling the function, it will only generate orbits while number-density n0 will be considered as the total number-density.

-----------------------------------------------------------------------------------------------------------------------------------------------------------

Input and output parameters for the function synthetic_population are (also defined in the fucntion's docstring):

input:

    v - interstellar velocities of the ISOs (m/s)
    pv - distribution of the interstellar velocities of ISOs given in the array "v"
    n0 - interstellar number-density of ISOs for objects larger than d0 (objects per au^3)
    d_ref - reference diemeter for SFD (m)
    d - list of characteristic diameters of the population (see example above) (m)
    alpha - list of SFD slopes for the size ranges defined by the list d (see example above)
    rm - radius of the model sphere where the synthetic population is to be generated (au)
    dr - step for the numerical integration with respect to heliocentric distance (au) (see section 5.2 in Marceta, D.: Synthetic Population of Interstellar Objects in the Solar System, 2022)

output:

    q - perihelion distances (au)
    e - eccentricities
    f - true anomalies (degrees)
    inc - orbital inclinations (degrees)
    Omega - longitudes of ascending nodes (degrees)
    omega - arguments of perihelions (degrees)  
    D - diameters of ISOs (m) (optional)

Remark:
A potential problem can arise if the range of velocities is very large 
(e.g. larger than several sigmas from the mode of the distribution of the interstellar velocities, pv). If this happens, scipy
functions interpolate.splrep and interpolate.splev have problem because interpolated function pv oscilates around zero. 
To avoid this problem, keep the range of the interstellar velocities in a reasonable range (several sigmas).


