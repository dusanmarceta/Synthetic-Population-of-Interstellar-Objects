import numpy as np
from synthetic_population_arbitary_kinematics import synthetic_population


## constants
#au=1.495978707e11 # astronomical unit
#R=696340000. # radius of the Sun
#mu=1.32712440042e20  # standard gravitional parameter of the Sun

# ---------------------------------------------------------------------------------------------

qs, es, incs, omegas, Omegas, fs = synthetic_population(rm=50, dr=1, n0=0.1, v_min=1000, v_max=100000, speed_resolution=50, angle_resolution=90, u_Sun=10000, v_Sun=11000, w_Sun=7000, sigma_vx=31000, sigma_vy=23000, sigma_vz=16000, vd=np.deg2rad(7), va=0, R_reff=696340000.)



