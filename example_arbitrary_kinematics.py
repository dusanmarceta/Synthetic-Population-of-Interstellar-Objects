import numpy as np
from synthetic_population_arbitary_kinematics import synthetic_population

# =============================================================================
# This code generates synthetic population of ISOs in arbitrary helicentric sphere for the defined input parameters
# =============================================================================

# INPUT PARAMETERS
rm=50 # (au) radius of the sphere where the population is generated
n0=0.1 # interstellar spatial number-density (at infinity)
R_eff=696340000. # effective radius of the Sun (the smallet heliocentric distance which an ISO can survive)

v_min=1000 # (m/s) minimum allowed interstellar speed
v_max=100000 # (m/s) maximum allowed interstellar speed

# Parameters that control the numerical integration 
dr=0.1 # integration step for helicentric distance
speed_resolution=50 # resolution for the speed. Range (v_min, v_max) is divided by this number
angle_resolution=90 # resolution for galactic longitude and latitude

# Solar motion with respect to the LSR (m/s)
u_Sun=10000
v_Sun=11000
w_Sun=7000

# Dispersions of U,V,W components (m/s)
sigma_vx=31000
sigma_vy=23000
sigma_vz=16000

vd=np.deg2rad(7) # vertex deviation
va=0 # asymmetrical drift


# =============================================================================
# # Output from the function synthetic_population:
# 
#     q - perihelion distance [m]
#     e - eccentricity
#     inc - orbital inclination [radians]
#     omega - argument of perihelion [radians]
#     Omega - longitude of ascending node [radians]   
#     f - true anomaliy [degrees]
# =============================================================================


q, e, inc, omega, Omega, f = synthetic_population(rm=rm, dr=dr, n0=n0, 
                                                        v_min=v_min, v_max=v_max, 
                                                        speed_resolution=speed_resolution, 
                                                        angle_resolution=angle_resolution, 
                                                        u_Sun=u_Sun, v_Sun=v_Sun, w_Sun=v_Sun, 
                                                        sigma_vx=sigma_vx, sigma_vy=sigma_vy, sigma_vz=sigma_vz, 
                                                        vd=vd, va=va, R_reff=R_eff)



