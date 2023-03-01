import numpy as np
from utils import cart2orb, mean_anomaly, initialization
from scipy import interpolate
from tqdm import tqdm

# constants
mu = 1.3271244e+20  # Sun gravity parameter
au = 149597870700.0  # astronomical unit

#--------------------------INPUT PARAMTERS-------------------------------------

r_model=10. # (au)radius of model sphere (where the population is generated)
number_density = 0.1 # interstellar number density (unaffected by the solar gravity)

v_min=1e3 # (m/s) minimum allowed interstellar velocity
v_max=1e5 # (m/s) maximum allowed interstellar velocity

dr=0.5 # (au) division of initialization sphere to calculate gravitational focusing in spherical shells
dv=5000 # (m/s) division of velocity range to calculate gravitational focusing and velocity increment

# Solar motion with respect to LSR
u_Sun=10000
v_Sun=11000
w_Sun=7000

# M class 
# dispersionas with respect to LSR
sigma_vx=31000
sigma_vy=23000
sigma_vz=16000
# vertex deviation
vd=np.deg2rad(7) 

array_size=1000000 # to avaoid memory problams the calculation is conducted part by part. Every time is calculated number of objects defined by this parameter

output_file_name='ISO_population.txt'

#--------------------------SIMULATION------------------------------------------

# paramters of the Schwarzschild ellipsoid
mean=[-u_Sun, -v_Sun, -w_Sun]
sigma_vx_vy=0.5*(sigma_vx**2-sigma_vy**2)*np.tan(2*vd)
cov = [[sigma_vx**2, sigma_vx_vy, 0], [sigma_vx_vy, sigma_vy**2, 0], [0, 0, sigma_vz**2]]  # diagonal covariance

# calculating initialization time and radius of initialization sphere
t_init, r_init=initialization(v_min, v_max, r_model*au)

# arrays of helicentric distances and velocities in which the gravitational focusing is calculated
r_array=np.arange(r_model, r_init/au+dr, dr)
v_array=np.arange(v_min, v_max+dv, dv)

# Initial helicentric distances
r_initial=[]

# Initial interstellar velocities
vx_initial_inf=[]
vy_initial_inf=[]
vz_initial_inf=[]

# Initial velocities at given helicentric distance increased by factor np.sqrt(v**2+v_escape**2)/v
vx_initial_r=[]
vy_initial_r=[]
vz_initial_r=[]

# distribution of interstellar velocities
vx_sample, vy_sample, vz_sample = np.random.multivariate_normal(mean, cov, array_size).T
v_sample=np.sqrt(vx_sample**2+vy_sample**2+vz_sample**2)
prob=np.histogram(v_sample, 100, density=True)

# probability distribution of interstellar speeds used for calculation of the gravitational focusing
v_interp=prob[1][:-1]+(prob[1][1]-prob[1][0])/2
p_interp=prob[0]

tck = interpolate.splrep(v_interp, p_interp, s=0)

br_r=[] # number of objects in a spherical shell
for i in tqdm(range(len(r_array)-1)):
    
    v_escape=np.sqrt(2*mu/(r_array[i]+dr/2)/au) # escape velocity in ith spherical shell

    brr=0
    for j in range(len(v_array)-1):
        
        augmentation_density=np.sqrt(1+2*mu/((r_array[i]+dr/2)*au)/(v_array[j]+dv/2)**2) # increment of number density at ith spherical shell and jth velocity interval

        p_v = interpolate.splev(v_array[j]+dv/2, tck, der=0)*dv # probability for jth velocity range
        
        br=int(np.round((r_array[i+1]**3-r_array[i]**3)*4/3*np.pi*number_density*augmentation_density*p_v)) # total number of objects in ith spherical shell and jth velocity interval
        brr=brr+br # tatal number for all valocity intervals
        
        # initial velocity components in jth velocity interval and ith spherical shell
        vx_v_r=[]
        vy_v_r=[]
        vz_v_r=[]
        
        # corresponding initial interstellar velocity components in jth velocity interval and ith spherical shell
        vx_v_inf=[]
        vy_v_inf=[]
        vz_v_inf=[]
        
        # sampling from given resolution and wait until given number (br) of elements is achieved 
        while len(vx_v_r)<br:

            vxx, vyy, vzz = np.random.multivariate_normal(mean, cov, 10000).T # 1000 is arbitrary chosen. From those, only "br" will be selected which fall in the given speed interval
            
            v=np.sqrt(vxx**2+vyy**2+vzz**2)
            velocity_increment=np.sqrt(v**2+v_escape**2)/v # velocity increment at ith spherical shell due to the Solar gravity
            
            # increased velocities
            vx_v_r+=list((vxx*velocity_increment)[np.logical_and(v>v_array[j], v<v_array[j+1])])
            vy_v_r+=list((vyy*velocity_increment)[np.logical_and(v>v_array[j], v<v_array[j+1])])
            vz_v_r+=list((vzz*velocity_increment)[np.logical_and(v>v_array[j], v<v_array[j+1])])
            
            # corresponding interstellar velocities (without increment)
            vx_v_inf+=list(vxx[np.logical_and(v>v_array[j], v<v_array[j+1])])
            vy_v_inf+=list(vyy[np.logical_and(v>v_array[j], v<v_array[j+1])])
            vz_v_inf+=list(vzz[np.logical_and(v>v_array[j], v<v_array[j+1])])
            
        # when the number is larger than "br" we take only first "br" elements and put in the arrays for initial conditions
        vx_initial_r+=vx_v_r[:br]
        vy_initial_r+=vy_v_r[:br]
        vz_initial_r+=vz_v_r[:br]
        
        vx_initial_inf+=vx_v_inf[:br]
        vy_initial_inf+=vy_v_inf[:br]
        vz_initial_inf+=vz_v_inf[:br]
        
        # sampling helicentric distances in ith spherical shell
        random_min=(r_array[i]/(r_init/au))**3
        random_max=(r_array[i+1]/(r_init/au))**3
        
        random=np.random.random(br)*(random_max-random_min)+random_min

        r_r= list(np.cbrt(random) * r_init)

        r_initial+=r_r # adding helicentric distances from the ith spherical shell
    br_r.append(int(np.round(brr))) # number of objects in sherical shells (for check)

number_of_loops=int(np.ceil(len(r_initial)/array_size)) # number of loops in order to calculate all objects

# resulting orbital elements
q_out=[]
e_out=[]
M_out=[]
o_out=[]
O_out=[]
inc_out=[]
vx_inf_out=[]
vy_inf_out=[]
vz_inf_out=[]
    
for i in tqdm(range(number_of_loops)):
    
    print('{} of {}'.format(i, number_of_loops))
    
    if i==number_of_loops-1:
        size=len(r_initial)-(number_of_loops-1)*array_size
        part=np.arange(i*array_size,i*array_size+size)
    else: # in the last loop we take only those which left
        size=array_size
        part=np.arange(i*array_size,(i+1)*array_size)
    
    # initial components of velocity
    vx = np.array(vx_initial_r)[part]
    vy = np.array(vy_initial_r)[part]
    vz = np.array(vz_initial_r)[part]
    
    # corresponding interstellar components of velocity (not used for caluclation, just for results)
    vx_inf = np.array(vx_initial_inf)[part]
    vy_inf = np.array(vx_initial_inf)[part]
    vz_inf = np.array(vx_initial_inf)[part]
    
    # koordinate objekata
    long = np.random.random(len(part)) * 2 * np.pi  # initial longitudes of objects
    lat = np.arccos((0.5 - np.random.random(len(part))) * 2) - np.pi / 2  # initial latitudes of objects
    
    #  initial cartesian coordinates of objects
    x = np.array(r_initial)[part] * np.cos(long) * np.cos(lat)
    y = np.array(r_initial)[part] * np.sin(long) * np.cos(lat)
    z = np.array(r_initial)[part] * np.sin(lat)
    
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
    
    vx_inf_out += list(vx_inf[inside])
    vy_inf_out += list(vy_inf[inside])
    vz_inf_out += list(vz_inf[inside])
    
    e_out += list(e[inside])
    q_out += list(a[inside]*(1-e[inside]))
    M_out += list(M[inside])
    o_out += list(o[inside])
    O_out += list(O[inside])
    inc_out += list(inc[inside])

output=np.column_stack((e_out, np.array(q_out)/au, M_out, o_out, O_out, inc_out, vx_inf_out, vy_inf_out, vz_inf_out))

# writing output file
np.savetxt(output_file_name, output, fmt='%.5f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f')