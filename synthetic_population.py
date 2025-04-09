import numpy as np
from scipy import interpolate
from scipy.integrate import simps, cumtrapz
from tqdm import tqdm
from scipy.stats import multivariate_normal

# constants
au=1.495978707e11 # astronomical unit
mu=1.32712440042e20  # standard gravitional parameter of the Sun

def p_vx_vy_vz(vx, sigma_vx, mu_x, vy, sigma_vy, mu_y, vz, sigma_vz, mu_z, vd, va):
    """Generates samples of rectangular components of interstellar velocity vector in galactic coordinate frame"""
        
    sigma_vx_vy=0.5*(sigma_vx**2-sigma_vy**2)*np.tan(2*vd)
    
    mean=[mu_x, mu_y-va, mu_z]
    
    cov = [[sigma_vx**2, sigma_vx_vy, 0], [sigma_vx_vy, sigma_vy**2, 0], [0, 0, sigma_vz**2]]  # diagonal covariance
   
    rv = multivariate_normal(mean, cov)
    
    return rv.pdf(np.transpose([vx, vy, vz]))

def p_v_l_b(v,l,b, sigma_vx, mu_x, sigma_vy, mu_y, sigma_vz, mu_z, vd, va):
    """Generates samples of interstellar velocity vector (magnitude, longitude/latitude of the vector) in galactic coordinate frame"""
    vx=-v*np.cos(b)*np.cos(l)
    vy=-v*np.cos(b)*np.sin(l)
    vz=-v*np.sin(b)
    
    return np.transpose(p_vx_vy_vz(vx, sigma_vx, mu_x, vy, sigma_vy, mu_y, vz, sigma_vz, mu_z, vd, va))*(vx**2+vy**2+vz**2)*np.cos(b)


def synthetic_population(rm, n0, v_min, v_max, 
                         u_Sun, v_Sun, w_Sun, 
                         sigma_vx, sigma_vy, sigma_vz, 
                         vd, va, R_reff,
                         speed_resolution=100, angle_resolution=90, dr=0.1, 
                         d_ref=1000, d=[], alpha=[]):

    '''
    This function generates synthetic orbits and/or sizes of interstellar objects (ISO) in the solar system according to Marceta (2023, Astronomy and Computing, vol 42).
    The input parameters define kinematics and number-density of ISOs in the interstellar space (unperturbed by the solar gravity). Beside this, the function also generates
    sizes of ISOs according to the (broken) power law according to the input parameters.
    
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
    '''
    
    # setting maximum size of arrays in order to avoid memory problems
    maximum_array_size=int(1e7) 
    # calculating total number density for all object with diameters between d[0] and d[-1]
    d=np.array(d)
    alpha=np.array(alpha)
    if len(d)>1:
        ind=np.argwhere(d<=d_ref).flatten()[-1] # largest d smaller of equal than d_ref 
        if ind==len(alpha): # this only asures that for the last point (d[-1]), the parameters from the last interval are used 
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
        n_total=n0 # if there is no requirement for calculating sizes of ISOs, n0 is cosidered as total number-density
     
    r_min=1.001*R_reff  # Coefficient 1.001 is used to avoid singularity at the surface of the Sun
    
    # conversion of units
    r_max=rm*au
    dr=dr*au  #  increament step converted to SI 
    
    # setting the grid
    r_resolution= int(np.ceil((r_max-r_min)/dr))+1 # number of elements in the array of heliceontric distances with a step closest to dr
    
    r_arr=np.linspace(r_min, r_max, r_resolution)
    v_arr=np.linspace(v_min, v_max,speed_resolution)
    l_arr=np.linspace(0,2*np.pi, angle_resolution)
    b_arr=np.linspace(-np.pi/2,np.pi/2,int(angle_resolution/2))
    
    if speed_resolution*angle_resolution**2/2>maximum_array_size:
        raise Exception("Maximum number of points on v-l-b grid has been exceeded. Resolution for v and/or l and/or b must be reduced.") 
    
    # mesh for longitude and latitude of IS velocity vector
    l_mesh, b_mesh = np.meshgrid(l_arr, b_arr)
    
    # making 3D arrays for v, l, b
    ind=np.mgrid[0:len(v_arr), 0:len(l_arr), 0:len(b_arr)]
    v=v_arr[ind[0]]
    l=l_arr[ind[1]]
    b=b_arr[ind[2]]
    
    # probability density distribution w.r.t. magnitude and direction of intrstellar velocities (at infinity), Eqs. 14 and 15
    p_vlb=p_v_l_b(v, l, b, sigma_vx, -u_Sun, sigma_vy, -v_Sun, sigma_vz, -w_Sun, vd, va) 
    
    # marginal distribution of magnitudes interstellar velocities
    p_v=simps(simps(p_vlb, b_arr), l_arr)
    
    # If necessary, to avoid problems with memory the job is divided so that the larges array is smaller than the predefined value
    size=len(r_arr) * np.shape(v)[0] * np.shape(v)[1] * np.shape(v)[2]
    
    
    # no division
    indices=[0, len(r_arr)]
    new_sizes=[len(r_arr)]
    new_size = len(r_arr)
    
    if size>maximum_array_size:  # division 
        div=int(np.ceil(size/maximum_array_size)) # number of arrays to which the large array is divided 
        new_size=int(np.floor(len(r_arr)/div)) # new size of the arrays
        num, remainder=divmod(len(r_arr), new_size) # remainder is the size of the last array which is general different from the others
        if remainder!=0:
            new_sizes=num*[new_size]+[remainder]  # new sizes of the arrays
        else:
            new_sizes=num*[new_size]    
        indices=[0] # indices where the large array is divided into smaller ones
        for i in range(len(new_sizes)):
            indices.append(sum(new_sizes[:i+1]))       
#    else: # no division
#        indices=[0, size]
#        new_sizes=[size]
        
    p_r=np.zeros(len(r_arr))   
    for i in tqdm(range(len(new_sizes)), desc="1/5: Heliocentric distance"):
    
        # p6 marginal with respect to B and phi
        p_rvlb=np.zeros([new_sizes[i], np.shape(v)[0], np.shape(v)[1], np.shape(v)[2]])
        
        for j in range(new_sizes[i]):
            
            p_rvlb[j]=p_vlb*((1+2*mu/r_arr[i*new_size+j]/v**2)**(1/2)+
                  (1+2*mu/r_arr[i*new_size+j]/v**2-
                   (R_reff/r_arr[i*new_size+j])**2*(1+2*mu/R_reff/v**2))**(1/2))/2
        
        # marginal with repsect to all except r (Eq.18) 
        p_r[indices[i]:indices[i+1]]=simps(simps(simps(p_rvlb, b_arr), l_arr),v_arr)
    
    # total number of object inside heliocentric sphere
    N_r=np.zeros(len(r_arr)) 
    N_r[1:]=cumtrapz(p_r*r_arr**2,r_arr)
    N_r=N_r*4*np.pi/au**3*n_total # Adjusting the total number of objects to the defined value of the interstellar number-density (Eq. 20)
    
    total_number=int(np.floor(np.max(N_r))) # total number of objects in the population
    
    if total_number > maximum_array_size:
        raise Exception("The number of ISOs is greater than the defined maximum array size. Try decreasing number-density and/or radius of the model sphere.") 
    
# =============================================================================
# Determining diameters of ISOs
# ============================================================================= 
        
    D_s=[] # diameters
        
    # calculating the sizes of ISOs
    if len(d)>1: 
        N_ref = [n[0]-n[i] for i in range(len(d))]
        x=np.linspace(N_ref[0], N_ref[-1], total_number) # uniform sample which is transformed using the Inverse Transform Sampling method
        D_s=np.zeros_like(x)
    
        for i in range(len(x)):
            ind=np.argwhere(x[i]>=N_ref).flatten()[-1] # najveci koji je manji od d_ref
            if ind==len(alpha):
                ind-=1       
            D_s[i]=d[ind]*((n[0]-x[i])/n[ind])**(1/alpha[ind]) 

# =============================================================================
# Determining orbits of ISOs
# ============================================================================= 
    if total_number>0: # if there are ISOs in the model sphere
        
        #  Initialization of synthetic samples
        vs=np.zeros(total_number) # interstellar velocity
        Bs=np.zeros(total_number) # impact parameter
        ls=np.zeros(total_number) # longitude of intestellar velocity vector
        bs=np.zeros(total_number) # latitude of intestellar velocity vector
        q_s=np.zeros(total_number) # perihelion distance
        inc_s=np.zeros(total_number) # inclination
        node_s=np.zeros(total_number) # longitude of ascending node
        argument_s=np.zeros(total_number) # argument of perihelion
        
# =============================================================================
# Determining heliocentric distances
# ============================================================================= 
        ur=np.random.random(total_number)*np.max(N_r) # random number for inverting helicentric distance
            
        # Inverse interpolation to obtaine ISO's helicentric distance according to Eq. 21 (cubic B-spline interpolation is used)
        tck = interpolate.splrep(N_r, r_arr, s=0)
        rs = interpolate.splev(ur, tck, der=0) # the set of helicentric distances (Eq. 21)
    
# =============================================================================
# Determining intestalar velocity
# ============================================================================= 
        
        # division of arrays if necessary      
        size=total_number*speed_resolution # total size of array
    
        if size>maximum_array_size: # division
            div=int(np.ceil(size/maximum_array_size))  
            new_size=int(np.floor(total_number/div))
            num, remainder=divmod(total_number, new_size)
            if remainder!=0:
                new_sizes=num*[new_size]+[remainder]  
            else:
                new_sizes=num*[new_size]
            indices=[0]
            for i in range(len(new_sizes)):
                indices.append(sum(new_sizes[:i+1]))
        else: # no division
            indices=[0, total_number]
            new_sizes=[total_number]
            new_size=total_number
      
        p__rs__vs=np.zeros(total_number) # marginal w.r.t l,b,B, conditional w.r.t. rs, vs
        for i in tqdm(range(len(new_sizes)), desc="2/5: Interstellar velocity"):
            
            p_v_cdf=np.zeros([new_sizes[i], speed_resolution]) 
            v_vec=np.ones([new_sizes[i], speed_resolution]) 
            v_vec=np.multiply(v_arr, v_vec)
            
            r_vec=np.ones([new_sizes[i], speed_resolution]) 
            r_vec=np.transpose(np.multiply(rs[indices[i]:indices[i+1]], np.transpose(r_vec)))
            
            # probability density function of v, marginal w.r.t l,b,B, conditional w.r.t. rs
            p_v__rs=p_v/2*(np.sqrt(1+2*mu/r_vec/v_vec**2)+np.sqrt(1+2*mu/r_vec/v_vec**2-R_reff**2/r_vec**2*(1+2*mu/v_vec**2/R_reff)))
            
            # Numerical integration w.r.t v to obtain cumulative probability 
            p_v_cdf=cumtrapz(p_v__rs, v_vec)
            p_v_cdf=np.insert(p_v_cdf, 0, 0,axis=1)
            p_v_cdf=np.transpose(np.multiply(1/np.max(p_v_cdf, axis=1), np.transpose(p_v_cdf))) # normalization
            
            uv=np.random.rand(new_sizes[i]) # random number for inverting interstellar velocity
           
            for j in range(new_sizes[i]):
                v_try=0
                while v_try<v_min or v_try>v_max: # ensuring that the resulting value falls in allowed range of interstellar velocities
                    # Inverse Transform Sampling Method for v
                    # Inverse interpolation to obtaine ISO's interstellar velocity (cubic B-spline interpolation is used)       
                    spl = interpolate.InterpolatedUnivariateSpline(v_arr, p_v_cdf[j]-uv[j])
                    v_try=spl.roots()
                    
                vs[i*new_size+j]=v_try
                spl = interpolate.InterpolatedUnivariateSpline(v_arr, p_v__rs[j])
                p__rs__vs[i*new_size+j]=spl(v_try)  # probability density function of v, marginal w.r.t l,b,B, conditional w.r.t. rs
                
# =============================================================================
# Determining impact parameter for every rs, vs
# =============================================================================       
    
        spl=interpolate.InterpolatedUnivariateSpline(v_arr, p_v)
        p_v__vs=spl(vs)
        p_v__vs_norm=p_v__vs/p__rs__vs
    
        # limiting value of the function p_rv which defines if the ISO is Sun impactor or not (first equation in Eq. 18)
        p_rv_lim=p_v__vs_norm/2*(np.sqrt(1+2*mu/rs/vs**2)-np.sqrt(1+2*mu/rs/vs**2-R_reff**2/rs**2*(1+2*mu/vs**2/R_reff)))

        uB=np.random.rand(total_number) # random number for inverting interstellar impact parameter
    
        # Inverse Transform Sampling Method for B according to Eqs. 24
        f1=np.sqrt(1+2*mu/rs/vs**2)
        f2=np.sqrt(1+2*mu/rs/vs**2)+np.sqrt(1+2*mu/rs/vs**2-R_reff**2/rs**2*(1+2*mu/vs**2/R_reff))
     
        Bs[uB<p_rv_lim]=rs[uB<p_rv_lim]*np.sqrt(f1[uB<p_rv_lim]**2-((f1[uB<p_rv_lim]*p_v__vs_norm[uB<p_rv_lim]-2*uB[uB<p_rv_lim])/p_v__vs_norm[uB<p_rv_lim])**2)
        Bs[uB>p_rv_lim]=rs[uB>p_rv_lim]*np.sqrt(f1[uB>p_rv_lim]**2-(f2[uB>p_rv_lim]*p_v__vs_norm[uB>p_rv_lim]-2*uB[uB>p_rv_lim])**2/4/p_v__vs_norm[uB>p_rv_lim]**2)
    
# =============================================================================
# Determining longitude of incoming velocity vector for every rs, vs, Bs
# =============================================================================
        # division of arrays if necessary 
        size=total_number * int(angle_resolution/2) * angle_resolution
        
        if size>maximum_array_size: # division
            div=int(np.ceil(size/maximum_array_size))   
            new_size=int(np.floor(total_number/div))
            num, remainder=divmod(total_number, new_size)
            if remainder!=0:
                new_sizes=num*[new_size]+[remainder]  
            else:
                new_sizes=num*[new_size]
            indices=[0]
            for i in range(len(new_sizes)):
                indices.append(sum(new_sizes[:i+1]))      
        else: # no division
            indices=[0, total_number]
            new_sizes=[total_number]
            new_size=total_number
        
        for i in tqdm(range(len(new_sizes)), desc="3/5: Longitude of interstellar velocity vector"):
            
            vec=np.ones([new_sizes[i],  int(angle_resolution/2), angle_resolution]) 
            vs_vec=np.transpose(np.multiply(vs[indices[i]:indices[i+1]], np.transpose(vec))) 
            Bs_vec=np.transpose(np.multiply(Bs[indices[i]:indices[i+1]], np.transpose(vec)))
            rs_vec=np.transpose(np.multiply(rs[indices[i]:indices[i+1]], np.transpose(vec)))
            l_vec=np.broadcast_to(l_mesh, np.shape(vec))
            b_vec=np.broadcast_to(b_mesh, np.shape(vec))
    
            p_lb__vs=p_v_l_b(vs_vec, l_vec, b_vec, sigma_vx, -u_Sun, sigma_vy, -v_Sun, sigma_vz, -w_Sun, vd, va)
            
            # probability density function of l, marginal w.r.t b,phi and conditional w.r.t. rs, vs, Bs
      
            p_l__rs__vs__Bs=np.transpose(np.transpose(cumtrapz(np.swapaxes(Bs_vec*p_lb__vs*vs_vec/(2*rs_vec*np.sqrt(vs_vec*rs_vec*2*mu*rs_vec-Bs_vec*vs_vec**2)),1,2), np.swapaxes(b_vec,1,2)))[-1])
         
            ul=np.random.rand(new_sizes[i]) # random number for inverting longitude of IS velocity vector
            for j in range(new_sizes[i]):
                p_l_cdf=np.zeros(len(l_arr)) # Initialization of cumulative distribution function
                
                # numerical integration to obtain cumulative probability function
                p_l_cdf[1:]=cumtrapz(p_l__rs__vs__Bs[j], l_arr)
                p_l_cdf=p_l_cdf/max(p_l_cdf) # normalization
                
                # inverse transform sampling
                spl = interpolate.InterpolatedUnivariateSpline(l_arr, p_l_cdf-ul[j])
                ls[i*new_size+j]=spl.roots()
                
# =============================================================================
# Determining latitude of incoming velocity vector for every rs, vs, Bs, ls
# =============================================================================
        # division of arrays if necessary 
        size=total_number * int(angle_resolution/2)
    
        if size>maximum_array_size: # division
            div=int(np.ceil(size/maximum_array_size))
            new_size=int(np.floor(total_number/div))
            num, remainder=divmod(total_number, new_size)
            if remainder!=0:
                new_sizes=num*[new_size]+[remainder]  
            else:
                new_sizes=num*[new_size]
            indices=[0]
            for i in range(len(new_sizes)):
                indices.append(sum(new_sizes[:i+1]))           
        else: # no division
            indices=[0, total_number]
            new_sizes=[total_number]
            new_size=total_number
               
        for i in tqdm(range(len(new_sizes)), desc="4/5: Latitude of interstellar velocity vector"):
        
            ls_vec=np.transpose(np.broadcast_to(ls[indices[i]:indices[i+1]], (int(angle_resolution/2), new_sizes[i])))
            vs_vec=np.transpose(np.broadcast_to(vs[indices[i]:indices[i+1]], (int(angle_resolution/2), new_sizes[i])))
            Bs_vec=np.transpose(np.broadcast_to(Bs[indices[i]:indices[i+1]], (int(angle_resolution/2), new_sizes[i])))
            rs_vec=np.transpose(np.broadcast_to(rs[indices[i]:indices[i+1]], (int(angle_resolution/2), new_sizes[i])))
            b_vec=np.broadcast_to(b_arr, (new_sizes[i], int(angle_resolution/2)))
        
            p_b__vs__ls=p_v_l_b(vs_vec, ls_vec, b_vec, sigma_vx, -u_Sun, sigma_vy, -v_Sun, sigma_vz, -w_Sun, vd, va)
        
            # probability density function of b, marginal w.r.t phi and conditional w.r.t. rs, vs, Bs, ls
            p_b__rs__vs__Bs__ls=Bs_vec*p_b__vs__ls*vs_vec/(2*rs_vec*np.sqrt(vs_vec*rs_vec*2*mu*rs_vec-Bs_vec*vs_vec**2))
            
            ub=np.random.rand(new_sizes[i])
            for j in range(new_sizes[i]):
                p_b_cdf=np.zeros(len(b_arr)) # Initialization of cumulative distribution function
                
                # numerical integration to obtain cumulative probability function
                p_b_cdf[1:]=cumtrapz(p_b__rs__vs__Bs__ls[j], b_arr)
                p_b_cdf=p_b_cdf/max(p_b_cdf) # normalization
                
                # inverse transform sampling
                spl = interpolate.InterpolatedUnivariateSpline(b_arr, p_b_cdf-ub[j])
                bs[i*new_size+j]=spl.roots()
            
# =============================================================================
# transforming determined parameters (rs, vs, Bs, ls, bs) inro orbital elements
# =============================================================================
    
        # orthogonal base (uu, vv) in a plane normal to interstellar velocity vector
        for i in tqdm(range(total_number), desc='5/5: Conversion to orbital elements'):
            xx=np.cos(bs[i])*np.cos(ls[i])
            yy=np.cos(bs[i])*np.sin(ls[i])
            zz=np.sin(bs[i])
        
            uu=np.array([0, -zz, yy])
            uu=uu/np.linalg.norm(uu)
            vv=np.array([yy**2+zz**2,-xx*yy, -xx*zz])
            vv=vv/np.linalg.norm(vv)
            
            # angle phi (See Figure 2)
            phi=np.random.rand()*np.pi*2 
            
            # unit normal vector to orbital plane
            orbital_plane_normal=uu*np.cos(phi)+vv*np.sin(phi);
            
            # inclination
            inc_s[i]=np.arccos(np.dot(orbital_plane_normal, np.array([0,0,1])))
        
            # unit vector toward ascending node
            node = np.cross(np.array([0, 0, 1]), orbital_plane_normal)
            
            # longitude of ascending node
            node_s[i] = np.arctan2(node[1], node[0])
            
            # angular momentum vector
            h= orbital_plane_normal*Bs[i]*vs[i]
            
            # auxiliary vector
            nn = np.cross(np.array([0, 0, 1]), h)
            
            # initial unit position vector
            r0=np.array([np.cos(bs[i])*np.cos(ls[i]), np.cos(bs[i])*np.sin(ls[i]), np.sin(bs[i])])
            
            # eccentricity vector
            e_vector = np.cross(-vs[i]*r0, h) / mu - r0
        
            # argument of perihelion
            argument_s[i] = np.arccos(np.linalg.linalg.dot(nn, e_vector) / np.linalg.norm(nn) / np.linalg.norm(e_vector))
            
            if e_vector[2] < 0:
                argument_s[i] = 2 * np.pi - argument_s[i]
        
        # semi-major axis       
        semi_major_axis_s=-mu/vs**2
        # perihelion distance
        q_s=semi_major_axis_s+np.sqrt(semi_major_axis_s**2+Bs**2)
        # eccentricity
        e_s=np.sqrt(1+Bs**2/semi_major_axis_s**2)
    
        node_s=np.mod(node_s,2*np.pi)
    
        # random choice of inbound or outbound branch of the orbit
        sign=np.random.random(len(rs))
        sign[sign<0.5]=-1
        sign[sign>0.5]=1
        
        # true anomaly
        f_s=sign*np.arccos((semi_major_axis_s*(1-e_s**2)/rs-1)/e_s)
      
    if len(d)>1:
        return(q_s/au, e_s, f_s, inc_s, node_s, argument_s, D_s)
    else:
        return(q_s/au, e_s, f_s, inc_s, node_s, argument_s)
            
