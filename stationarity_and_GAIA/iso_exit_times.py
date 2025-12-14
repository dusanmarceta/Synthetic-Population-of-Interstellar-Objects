import numpy as np
import auxiliary_functions as aux
from synthetic_population import synthetic_population, synthetic_population1


STEP_YEARS = 1.0 / 365.0
MAX_YEARS = 100.0
SECONDS_PER_YEAR = 365.0 * 86400.0
OUTPUT_FILE = "iso_exit_times.txt"

R_MODEL = 5
N0 = 1e-1
V_MIN = 1e3
V_MAX = 2e5
U_SUN = 1e4
V_SUN = 1.1e4
W_SUN = 7e3
SIGMA_VX = 1.2e4
SIGMA_VY = 1.1e4
SIGMA_VZ = 0.9e4
VD = np.deg2rad(36)
VA = 0
R_REFF = 696340000.
SPEED_RES = 100
ANGLE_RES = 90
DR = 0.1

use_gaia_data = False
if use_gaia_data:
    q, e, f, inc, node, argument = synthetic_population1(
            rm=R_MODEL, n0=N0, p_min=100, v_mag_min=V_MIN, v_mag_max=V_MAX, R_reff=R_REFF,
            speed_resolution=SPEED_RES, angle_resolution=ANGLE_RES, dr=DR
    )

else:
    q, e, f, inc, node, argument = synthetic_population(
            rm=R_MODEL, n0=N0, v_min=V_MIN, v_max=V_MAX, 
            u_Sun=U_SUN, v_Sun=V_SUN, w_Sun=W_SUN, 
            sigma_vx=SIGMA_VX, sigma_vy=SIGMA_VY, sigma_vz=SIGMA_VZ, 
            vd=VD, va=VA, R_reff=R_REFF,
            speed_resolution=SPEED_RES, angle_resolution=ANGLE_RES, dr=DR
        )


a = q / (1 - e)
num_objects = len(a)
print(f"Generated {num_objects} objects")

with open(OUTPUT_FILE, 'w') as f_out:
    f_out.write("q e inc node argument f_init exit_time_years\n")

    for i in range(num_objects):        
        H0 = aux.true2ecc(f[i], e[i])
        M0 = e[i] * np.sinh(H0) - H0
        
        n = np.sqrt(aux.mu / np.abs(a[i] * aux.au)**3) # rad/s
        
        t_curr = 0.0 # years
        r_prev = 0.0
        exited = False
        
        max_steps = int(MAX_YEARS / STEP_YEARS)
        
        for step in range(max_steps):
            M = M0 + n * t_curr * SECONDS_PER_YEAR
            H = aux.kepler(e[i], M, 1e-6)
            r_curr = np.abs(a[i]) * (e[i] * np.cosh(H) - 1.0)

            if r_curr > R_MODEL and r_prev < R_MODEL:
                exit_time = t_curr
                exited = True
                break

            r_prev = r_curr
            t_curr += STEP_YEARS
        
        if exited:
            line = (f"{q[i]:.6e} {e[i]:.6f} {inc[i]:.6f} {node[i]:.6f} "
                    f"{argument[i]:.6f} {f[i]:.6f} {exit_time:.6f}\n")
            f_out.write(line)

print(f"Results saved to {OUTPUT_FILE}")

