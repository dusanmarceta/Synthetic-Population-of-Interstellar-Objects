# Synthetic-population-of-Interstellar-Obejcts

    Author:      Dusan Marceta
    Institution: University of Belgrade
    Email:       dmarceta@matf.bg.ac.rs
    Date:        September 2022
    Version:     1.0

Required python libraries:

    numpy, scipy, random, tqdm

This code can be used for generating a synthetic population of interstellar objects (orbits and sizes) in arbitrary volume of space around the Sun according to Marceta (2023, Astronomy and Computing, vol 42). The assumed distribution of interstellar velocities of ISOs has to be provided as an input. The example file "example.py" which demonstrates generation of a synthetic population is provided.

-----------------------------------------------------------------------------------------------------------------------------------------------------------

GENERATING SIZES

Beside orbits, the code can optionally generate sizes (diameters) of ISOs, if required via input parameters. The sizes are generated according to (broken) power law for the size-frequency distribution (SFD). In order to generate sizes, the input parameters must be defined appropriately. This is explained through the following example:


    d_ref=1000 - reference diameter for which the interstellar number-density is defined (m)
    n0=1e-2 - number of objects per cubic au whose diameters are larger than d_ref (interstellar number-density)
    d=[100, 500, 10000] - characteristic diameters of the population
    alpha=[-2, -3] - slopes of the broken power law


In this example, the code will generate objects with sizes ranging from 100 to 10000 m (d[0] to d[-1]). The referent number density (n0) is 1e-2 per cubic au for objects larger than 1000 m (d_ref). Objects within the size range [100, 500) have SFD slope of -2, while object within the size range [500, 10000] have slope of -3.

The code will first calculate the total number-density for all objects inside the defined size-range. After that, it will calculate their sizes acording to the defined SFD. 

If the parameters d and alpha are not defined when calling the function, it will only generate orbits. In this case, interstellar number-density n0 will be considered as the total number-density.

-----------------------------------------------------------------------------------------------------------------------------------------------------------

Input and output parameters for the function synthetic_population (also defined in the fucntion's docstring):

input:

    rm: radius of the model sphere (au)
    n0: number-density of the ISOs in the interstellar space (unperturbed by the Sun's gravity)
        for objects with diameter >d0 (au^-1)
    v_min: minimum allowed interstellar speed (m/s)
    v_max: maximu allowed interstellar speed (m/s) 
    u_Sun:  u-component of the Sun's velocity w.r.t. LSR (m/s) 
    v_Sun: v-component of the Sun's velocity w.r.t. LSR (m/s) 
    w_Sun: w-component of the Sun's velocity w.r.t. LSR (m/s) 
    sigma_vx: standard deviation of x-component of ISOs' velocities w.r.t. LSR (m/s)
    sigma_vy: standard deviation of y-component of ISOs' velocities w.r.t. LSR (m/s)
    sigma_vz: standard deviation of z-component of ISOs' velocities w.r.t. LSR (m/s)
    vd: vertex deviation (radians)
    va:  assymetric drift (m/s)
    R_reff:  refference radius of the Sun (m)
    speed_resolution:  resolution of magnitudes of interstellar velocities (for numerical integration and inverse interpolation)
    angle_resolution: resolution of galactic longitude (for numerical integration and inverse interpolation)
    dr: increament step for heliocentric distance used for numerical integration and inverse interpolation (au)
    d_ref:  reference ISO diameter for which n0 is defined (m)
    d: array of diemetars for where power law for size frequency distribution (SFD) changes slope. This array also includes
       minimum and maximum diameter od the population (m). If this array is empty (default) the function does not calculate sizes of the objects 
       and takes n0 as the total number-density 
    alpha: array of slopes of the SFD

output:

    q - perihelion distances (au)
    e - eccentricities
    f - true anomalies (degrees)
    inc - orbital inclinations (degrees)
    Omega - longitudes of ascending nodes (degrees)
    omega - arguments of perihelions (degrees)  
    D - diameters of ISOs (m) (optional)


