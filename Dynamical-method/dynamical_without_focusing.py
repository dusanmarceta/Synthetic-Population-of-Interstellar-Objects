import numpy as np
from utils import cart2orb, mean_anomaly, initialization
from tqdm import tqdm

# constants
mu = 1.3271244e+20  # Sun gravity parameter
au = 149597870700.0  # astronomical unit

#--------------------------INPUT PARAMTERS-------------------------------------

r_model=30. # (au)radius of model sphere (where the population is generated)
number_density = 0.1 # interstellar number density (unaffected by the solar gravity)

v_min=1e3 # (m/s) minimum allowed interstellar velocity
v_max=2e5 # (m/s) maximum allowed interstellar velocity

# Solar motion with respect to LSR (m/s)
u_Sun=10000
v_Sun=11000
w_Sun=7000

# M class 
# dispersionas with respect to LSR (m/s)
sigma_vx=31000
sigma_vy=26000
sigma_vz=16000
# vertex deviation (degrees)
vd=np.deg2rad(7) 

array_size=1000000 # to avaoid memory problems the calculation is conducted part by part. Every time is calculated number of objects defined by this parameter

output_file_name='ISO_population.txt'

#--------------------------SIMULATION------------------------------------------
# paramters of the Schwarzschild ellipsoid
mean=[-u_Sun, -v_Sun, -w_Sun]
sigma_vx_vy=0.5*(sigma_vx**2-sigma_vy**2)*np.tan(2*vd)
cov = [[sigma_vx**2, sigma_vx_vy, 0], [sigma_vx_vy, sigma_vy**2, 0], [0, 0, sigma_vz**2]]  # diagonal covariance

# calculating initialization time and radius of initialization sphere
t_init, r_init=initialization(v_min, v_max, r_model*au)

total_initial_number = int(4 / 3 * (r_init/au) ** 3 * np.pi * number_density)  # total number of objects in initialization sphere

number_of_loops=int(np.ceil(total_initial_number/array_size)) # number of loops in order to calculate all objects

# resulting orbital elements and velocity components
q_out=[] # perihelion distance (au)
e_out=[] # eccentricity
M_out=[] # Mean anomaly (radians)
o_out=[] # agument of perihelion (radians)
O_out=[] # longitude of ascending node (radians)
inc_out=[] # inclination (radians)
vx_out=[] # velocity vector x-component (m/s)
vy_out=[] # velocity vector y-component (m/s)
vz_out=[] # velocity vector z-component (m/s)

for i in tqdm(range(number_of_loops)):
    
    if i==number_of_loops-1:
        size=total_initial_number-(number_of_loops-1)*array_size
    else: # in the last loop we take only those which left
        size=array_size  
    # securing that all initial velocities are in the allowed range defined by v_min and v_max
    vx=[]
    vy=[]
    vz=[]
    while len(vx)<size:
        vxx, vyy, vzz = np.random.multivariate_normal(mean, cov, 2*size).T
        v = np.sqrt(vxx**2+vyy**2+vzz**2)
        vx=vx + list(vxx[np.logical_and(v>v_min, v<v_max)])
        vy=vy + list(vyy[np.logical_and(v>v_min, v<v_max)])
        vz=vz + list(vzz[np.logical_and(v>v_min, v<v_max)])
    
    vx=np.array(vx[:size])
    vy=np.array(vy[:size])
    vz=np.array(vz[:size])
    
    # initial spherical coordinates of objects
    long = np.random.random(size) * 2 * np.pi  # longitude polozaja objekata
    lat = np.arccos((0.5 - np.random.random(size)) * 2) - np.pi / 2  # latitude polozaja objekata
    
    # initial heliocentric distances
    r = np.cbrt(np.random.random(size)) * r_init
    
    #  initial cartesian coordinates
    x = r * np.cos(long) * np.cos(lat)
    y = r * np.sin(long) * np.cos(lat)
    z = r * np.sin(lat)
     
    # caluclating initial orbital elements       
    e, a, o, O, inc, M = cart2orb(np.transpose([x, y, z]), np.transpose([vx, vy, vz]), mu)
    
    #  calculating current mean anomaly (after initialization time has passed)
    Mtr = mean_anomaly(M, 0, a, t_init/86400, mu)
    
    # critical eccentric anomaly (when object is on the edge of the model sphere)
    Ekr = np.arccosh(1 / e - r_model * au / e / a)
    
    # correspoding critical mean anomaly
    Mkr_max = e * np.sinh(Ekr) - Ekr
    
    # condition that an object is hyperbolic and inside the model sphere
    inside=np.logical_and(e>1, abs(Mtr) < Mkr_max)
    
    # resulting objects
    e_out += list(e[inside])
    q_out += list(a[inside]*(1-e[inside]))
    M_out += list(Mtr[inside])
    o_out += list(o[inside])
    O_out += list(O[inside])
    inc_out += list(inc[inside])
    vx_out += list(vx[inside])
    vy_out += list(vy[inside])
    vz_out += list(vz[inside])
    
output=np.column_stack((e_out, np.array(q_out)/au, M_out, o_out, O_out, inc_out, vx_out, vy_out, vz_out))

# writing output file
np.savetxt(output_file_name, output, fmt='%.5f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f')
