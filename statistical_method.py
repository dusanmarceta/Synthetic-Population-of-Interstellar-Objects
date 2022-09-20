import numpy as np
from scipy import interpolate
import random
from tqdm import tqdm

# constants
au=1.495978707e11 # astronomical unit
R=696340000. # radius of the Sun
mu=1.32712440042e20  # standard gravitional parameter of the Sun

# cahracteristic distributions of the interstellar velocities
def maxwell_boltzmann(v, sigma):
    # v and sigma should be given in m/s   
    return np.sqrt(2/np.pi)*v**2*np.exp(-v**2/2/sigma**2)/sigma**3

def log_normal(v, sigma, mu):
    # v, sigma and mu should be given in km/s 
    return 1/sigma/np.sqrt(2*np.pi)/v*np.exp(-(np.log(v)-mu)**2/2/sigma**2)/1000

def normal(v, sigma, mu):
    # v, sigma and mu should be given in m/s
    return 1/sigma/np.sqrt(2*np.pi)*np.exp(-0.5*((v-mu)**2/sigma**2))

# statistical method for the generation of the synthetic population of the interstellar objects    
def synthetic_population(v, pv, rm, n0, d_ref=1000, d=[], alpha=[],dr=1e-3):
    '''
    Input:
    v (m/s): array of velocities for which the distribution of the interstellar velocities is calculated. 
    It has to be equidistant vector defining minimum and maximum assumed interstellar velocity of the ISOs and the
    increament step. Example: v=np.arange(1e3, 101e3, 1e3)
    
    pv: defines the distribution of the interstellar velocities of the ISOs over the domain defined in
    velocity_array. It can be obtained by calling some of the functions from the file velocity_distributions.py or can be
    independantly defined.
    
    For detailed explanation of parameter which define SFD please README.txt
    
    n0 (objects per cubic au): number-density of the ISOs in the interstellar space (unperturbed by the Sun's gravity)
    for objects with diameter >d0
    
    d0 (m): referent diameter for which number-density is defined
    
    d (m): array of diemetars for where povwer law for size frequency distribution (SFD) changes slope. This array also includes
    minimum and maximum diameter od the population. If this array is empty (default) the function does not calculate sizes of the objects 
    and takes n0 as the total number-density 
    
    alpha: array of slopes of the SFD. This has 1 element less than the array d (because minimum and maximum diameter)
    
    rm [au]: radius of the heliocentric model sphere where the population of the ISOs is to be generated
    
    dr [au]: increament step for heliocentric distance used for numerical integration and inverse interpolation
    
    Output:
    q - perihelion distances [au]
    e - eccentricities
    f - true anomalies [degrees]
    inc - orbital inclinations [degrees]
    Omega - longitudes of ascending nodes [degrees]
    omega - arguments of perihelions [degrees]   
    '''
    
    # calculating total number density for all object with diameters between d[0] and d[-1]
    d=np.array(d)
    alpha=np.array(alpha)
    if len(d)>1:
        ind=np.argwhere(d<=d_ref).flatten()[-1] # largest d smaller than d0 
        if ind==len(alpha): # this only asures that for the last point (d[-1]) the parameters from the last interval are used 
            ind-=1
        
        n=[]
        for i in range(len(d)):
            nn=n0
            d0=d_ref
            
        
            if i<=ind: 
                for j, dd in enumerate(d[i:ind+1][::-1]):
                    nn*=(dd/d0)**alpha[ind-j]
                    d0=dd
                n.append(nn)
            else:
                for j, dd in enumerate(d[ind+1:i+1]):
                    nn*=(dd/d0)**alpha[i-len(d[ind+1:i+1])+j]
                    d0=dd
                n.append(nn)
    
        n_total=n[0]-n[-1] # total number density for objects inside the defined size range
        
    else:
        n_total=n0 # if there is no requirement for calculating sizes of ISOs n0 is cosidered as total number-density
      
    r_min=1.001*R  # Coefficient 1.001 is used to avoid singularity at the surface of the Sun
    r_max=rm*au
    dr=dr*au  #  increament step converted to SI 
    number_of_elements= int(np.ceil((r_max-r_min)/dr))+1 # numer of elemnts in the array of heliceontric distances with a step closest to dr
    r_lin=np.linspace(r_min, r_max, number_of_elements) # array or helicentric distances used for numerical integration and inverse interpolation
    # r_lin ranges from the surface of the Sun to the edge of the model sphere. 

    # initializing marginal probability density finction of r and v (Eqs. 18 and 25)
    p_rv=np.zeros([len(r_lin), len(v)]) 
    
    # Calculating p_rv for every heliocentroc distance from the r_lin array according to Eq. 25
    for j in range(len(r_lin)):
        p_rv[j]=pv*((1+2*mu/r_lin[j]/v**2)**(1/2)+(1+2*mu/r_lin[j]/v**2-(R/r_lin[j])**2*(1+2*mu/R/v**2))**(1/2))/2
        
    pr=np.zeros(len(r_lin)) # initializing marginal probability density finction of r

    dv=v[1]-v[0] # increament step for the array of the interstellar velocities
    
    # calculating marginal probability density finction of r according to Eq. 26
    for i in range(len(r_lin)):
        pr[i]=np.sum(p_rv[i])*dv
        
    # initializing total number of objects inside heliocentric sphere 
    Nr=np.zeros(len(r_lin))
    
    # Calculating total number of objects inside heliocentric sphere accoridng to Eq. 23
    for i in range(1,len(r_lin)):
        Nr[i]=Nr[i-1]+pr[i]*dr*4*r_lin[i]**2*np.pi/au**3
    
    Nr=Nr*n_total # Adjusting the total number of objects to the defined value of the interstellar number-density
    
    total_number=int(np.floor(np.max(Nr)))
   
    D=[]
    
    # calculating the sizes of ISOs
    if len(d)>1: 
        N_ref = [n[0]-n[i] for i in range(len(d))]
        
        x=np.linspace(N_ref[0], N_ref[-1], total_number) # uniform sample which is transformed using the Inverse Transform Sampling method
        D=np.zeros_like(x)
    
        for i in range(len(x)):
            ind=np.argwhere(x[i]>=N_ref).flatten()[-1] # najveci koji je manji od d_ref
            if ind==len(alpha):
                ind-=1       
            D[i]=d[ind]*((n[0]-x[i])/n[ind])**(1/alpha[ind]) 

    if total_number>0: # if there are ISOs in the model sphere
        
        inclination=np.arccos(1-2*np.random.random(total_number))  # according to Eq. 16
        longitude_of_node=np.random.random(total_number)*2*np.pi  # see section 5.2
        argument_of_perihelion=np.random.random(total_number)*2*np.pi # see section 5.2
        
        # random number from 0 to total number of ISOs
        ur=np.random.random(int(np.floor(np.max(Nr))))*np.max(Nr) 
        
        # Inverse Transform Sampling Method for r
        # Inverse interpolation to obtaine ISO's helicentric distance according to Eq. 27 (cubic B-spline interpolation is used)
        tck = interpolate.splrep(Nr, r_lin, s=0)
        rs = interpolate.splev(ur, tck, der=0) # the set of helicentric distances
        
        # Initializin the sets of the orbital elements
        q=np.zeros(total_number)
        e=np.zeros(total_number)
        true_anomaly=np.zeros(total_number)

        for k in tqdm(range(total_number)): # for every helicentric distance has to be determined interstellar velocity and impact parameter
            
            q[k]=2*rm*au 
            while q[k]>rm*au: # When interpolating pdf of the interstellar velocities, 
                # the function may get negative values at the tail of the distribution. 
                # This condition is to prevent this from generating unrealistic ISOs with B > rm.
            
                # finding the increase of the number-density at helicentric distance rs[k]
                tck = interpolate.splrep(r_lin, pr, s=0)
                pr_rs = interpolate.splev(rs[k], tck, der=0) # maximum value or the random number uv (se below and section 5.2)
                
                # Initializing marginal distribution of r and v (Eqs. 18 and 25)
                p_rv=np.zeros(len(v))
                
                # Calculating p_rv for specific helicentric distance rs[k] according to Eq. 25
                for i in range(len(v)):
                    p_rv[i]=pv[i]/2*(np.sqrt(1+2*mu/rs[k]/v[i]**2)+np.sqrt(1+2*mu/rs[k]/v[i]**2-R**2/rs[k]**2*(1+2*mu/v[i]**2/R)))  
    
                # Initializing marginal distribution of r and cumulative with resepct to v (Eqs. 19)
                p_rv_cdf=np.zeros(len(v))
                # Numerical integration with respect to v according to Eq. 19
                cum=0
                for i in range(1,len(v)):
                    cum=cum+(p_rv[i]+p_rv[i-1])/2*dv
                    p_rv_cdf[i]=cum           
    
                # Inverse Transform Sampling Method for v
                uv=np.random.rand()*pr_rs #  random number in the range [0,pr(rs[k])]
                # Inverse interpolation to obtaine ISO's interstellar velocity according to Eq. 28 (cubic B-spline interpolation is used)
                tck = interpolate.splrep(p_rv_cdf, v, s=0)
                vs = interpolate.splev(uv, tck, der=0) # interstellar velocity which corresponds to rs[k]
    
                # calculating pr(rs, vs) - see section 5.2 (Eqs. 29 and 30)
                tck = interpolate.splrep(v, p_rv, s=0)
                pr_rs_vs = interpolate.splev(vs, tck, der=0) # maximum value or the random number uB (se below and section 5.2)
                
                # calculating pv(vs) used in Eqs. 31. It is here obtained by interpolation, but it is better replace this
                # with the original function for calculating pv (the one which is used to calculate input parametar pv for the whole function)
                tck = interpolate.splrep(v, pv, s=0)
                pv_vs = interpolate.splev(vs, tck, der=0) # this is pv(vs) 
                
                # limiting value of the function p_rv which defines if the ISO is Sun impactor or not (first equation in Eq. 25)
                p_rv_lim=pv_vs/2*(np.sqrt(1+2*mu/rs[k]/vs**2)+np.sqrt(1+2*mu/rs[k]/vs**2-R**2/rs[k]**2*(1+2*mu/vs**2/R)))  
    
                uB=np.random.rand()*pr_rs_vs
                
                # Inverse Transform Sampling Method for v according to Eqs. 31
                f1=np.sqrt(1+2*mu/rs[k]/vs**2)
                f2=np.sqrt(1+2*mu/rs[k]/vs**2)+np.sqrt(1+2*mu/rs[k]/vs**2-R**2/rs[k]**2*(1+2*mu/vs**2/R))
    
                if uB<p_rv_lim: # Sun impactor
                    Bs=rs[k]*np.sqrt(f1**2-((f1*pv_vs-2*uB)/pv_vs)**2)
                else: # passer
                    Bs=rs[k]*np.sqrt(f2**2-(f2*pv_vs-2*uB)**2/4/pv_vs**2)
                    
                # Conversion of r,v,B to orbital elements
                a=-mu/vs**2
                q[k]=a+np.sqrt(a**2+Bs**2)
                e[k]=np.sqrt(1+Bs**2/a**2)
    
                # every rs corresponds to two true anomalies with a same absolute value, one before and one after the perihelion. 
                # This is used to randomly chose between them.
                sign=[-1,1][random.randrange(2)] 
                
                # calculating true anomaly according to Eq. 9
                true_anomaly[k]=sign*np.arccos((a*(1-e[k]**2)/rs[k]-1)/e[k])
                
    return(q/au, e, np.rad2deg(true_anomaly), 
                               np.rad2deg(inclination), np.rad2deg(longitude_of_node), 
                                                                  np.rad2deg(argument_of_perihelion), D)
    

    

