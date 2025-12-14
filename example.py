"""
This is the example of generating a synthetic population of interstellar objects using the Probabilistic method 
(Marceta, D.: Synthetic population of interstellar objects in the Solar System, Astronomy and Computing, vol. 42, 2023)

In addition to orbits, the code can also generate the sizes of ISOs, according to a (broken) power law. 
Below are two examples, one where only orbits are generated, as well as the example that generates both orbits and ISO sizes.

Input:
    Input:
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
        
    
    Output (synthetic samples of orbital elements):
    q_s - perihelion distance (au)
    e_s - eccentricity
    f_s - true anomaliy (radians)
    inc_s - orbital inclination (radians])
    node_s - longitude of ascending node (radians)
    argument_s - argument of perihelion (radians) 
    D_s (optional) - diameters of ISOs (m)
"""
import numpy as np
from synthetic_population import synthetic_population, synthetic_population_stationary

## Case 1: generating only orbits without generating the sizes of objects
#q, e, f, inc, Omega, omega= synthetic_population(rm=10, n0=0.1, v_min=1e3, v_max=2e5, 
#                                                                u_Sun=1e4, v_Sun=1.1e4, w_Sun=7e3, 
#                                                                sigma_vx=3.1e4, sigma_vy=2.3e4, sigma_vz=1.6e4, 
#                                                                vd=np.deg2rad(7), va=0, R_reff=696340000.,
#                                                                speed_resolution=100, angle_resolution=90, dr=0.1)

# Case 2: generating both orbits and sizes
"""
In this case, the code will generate objects with sizes ranging from 50 to 2000 m. The referent number density is 1e-1 per cubic au for 
objects larger than 1000 m. Objects within the size range [50, 500) have SFD slope of -1.5, 
while the object within the size range [500, 2000] have slope of -2.1.
"""

#q, e, f, inc, Omega, omega, D = synthetic_population(rm=1, n0=10, v_min=1e3, v_max=2e5, 
#                                                                u_Sun=1e4, v_Sun=1.1e4, w_Sun=7e3, 
#                                                                sigma_vx=3.1e4, sigma_vy=2.3e4, sigma_vz=1.6e4, 
#                                                                vd=np.deg2rad(7), va=0, R_reff=696340000.,
#                                                                speed_resolution=100, angle_resolution=90, dr=0.1, 
#                                                                d_ref=1000, d=[50, 500, 2000], alpha=[-1.5, -2.1])



# Case 3: generating stationary population (only orbits)

## Case 1: generating only orbits without generating the sizes of objects
q, e, f, inc, Omega, omega, v_sample, l_sample, b_sample, B_sample= synthetic_population_stationary(1., rm=1, n0=100, v_min=1e3, v_max=2e5, 
                                                                u_Sun=1e4, v_Sun=1.1e4, w_Sun=7e3, 
                                                                sigma_vx=3.1e4, sigma_vy=2.3e4, sigma_vz=1.6e4, 
                                                                vd=np.deg2rad(7), va=0, R_reff=696340000.,
                                                                speed_resolution=100, angle_resolution=90, dr=0.1)


