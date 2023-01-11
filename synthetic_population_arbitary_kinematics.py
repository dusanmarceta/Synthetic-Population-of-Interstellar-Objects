import numpy as np
from scipy import interpolate
import random
from tqdm import tqdm
from scipy.stats import multivariate_normal

# constants
au=1.495978707e11 # astronomical unit
mu=1.32712440042e20  # standard gravitional parameter of the Sun
def p_vx_vy_vz(vx, sigma_vx, mu_x, vy, sigma_vy, mu_y, vz, sigma_vz, mu_z, vd, va):
        
    sigma_vx_vy=0.5*(sigma_vx**2-sigma_vy**2)*np.tan(2*vd)
    
    mean=[mu_x, mu_y-va, mu_z]
    
    cov = [[sigma_vx**2, sigma_vx_vy, 0], [sigma_vx_vy, sigma_vy**2, 0], [0, 0, sigma_vz**2]]  # diagonal covariance
   
    rv = multivariate_normal(mean, cov)
    
    return rv.pdf(np.transpose([vx, vy, vz]))

def p_v_l_b(v,l,b, sigma_vx, mu_x, sigma_vy, mu_y, sigma_vz, mu_z, vd, va):
    
    vx=-v*np.cos(b)*np.cos(l)
    vy=-v*np.cos(b)*np.sin(l)
    vz=-v*np.sin(b)
    
    return np.transpose(p_vx_vy_vz(vx, sigma_vx, mu_x, vy, sigma_vy, mu_y, vz, sigma_vz, mu_z, vd, va))*(vx**2+vy**2+vz**2)*np.cos(b)
    

# statistical method for the generation of the synthetic population of the interstellar objects    
def synthetic_population(rm=50, dr=1, n0=0.1, v_min=1000, v_max=100000, speed_resolution=50, angle_resolution=90, u_Sun=10000, v_Sun=11000, w_Sun=7000, sigma_vx=31000, sigma_vy=23000, sigma_vz=16000, vd=np.deg2rad(7), va=0, R_reff=696340000.):
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
    
    d_ref (m): referent diameter for which number-density is defined
    
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
    
    r_min=1.001*R_reff  # Coefficient 1.001 is used to avoid singularity at the surface of the Sun
    r_max=rm*au
    dr=dr*au  #  increament step converted to SI 
    
    r_resolution= int(np.ceil((r_max-r_min)/dr))+1 # numer of elemnts in the array of heliceontric distances with a step closest to dr
    
    r_arr=np.linspace(r_min, r_max, r_resolution)
    v_arr=np.linspace(v_min, v_max,speed_resolution)
    l_arr=np.linspace(0,2*np.pi, angle_resolution, endpoint=False)
    b_arr=np.linspace(-np.pi/2,np.pi/2,int(angle_resolution/2), endpoint=False)
    
    l_mesh, b_mesh = np.meshgrid(l_arr, b_arr)
    
    dv=v_arr[1]-v_arr[0]
    dl=l_arr[1]-l_arr[0]
    db=b_arr[1]-b_arr[0]
    
    l_arr=l_arr+dl/2
    b_arr=b_arr+db/2
    
    ind=np.mgrid[0:len(v_arr), 0:len(l_arr), 0:len(b_arr)]
    
    v=v_arr[ind[0]]
    l=l_arr[ind[1]]
    b=b_arr[ind[2]]
    
    # gustina raspodele za v,l,b u beskonacnosti
    p_vlb=p_v_l_b(v, l, b, sigma_vx, -u_Sun, sigma_vy, -v_Sun, sigma_vz, -w_Sun, vd, va)
    
    # marginal distribution of interstellar velocities
    p_v=np.zeros_like(v_arr)
    for i in range(len(p_v)):
        p_v[i]=np.sum(p_vlb[i][:])*db*dl
    
    # p6 marginal with respect to B
    p_rvlb=np.zeros([len(r_arr), np.shape(v)[0], np.shape(v)[1], np.shape(v)[2]])
    for i in range(len(r_arr)):
        p_rvlb[i]=p_vlb*((1+2*mu/r_arr[i]/v**2)**(1/2)+(1+2*mu/r_arr[i]/v**2-(R_reff/r_arr[i])**2*(1+2*mu/R_reff/v**2))**(1/2))/2
      
        
        
    # marginal with repsect to except r
    p_r=np.zeros(len(r_arr))
    for i in range(len(p_r)):
        p_r[i]=np.sum(p_rvlb[i])*dv*db*dl
        
    
    # total number of object inside heliocentric sphere
    N_r=np.zeros(len(r_arr))
    for i in range(1,len(r_arr)):
        N_r[i]=N_r[i-1]+p_r[i]*dr*4*r_arr[i]**2*np.pi/au**3
        
    N_r=N_r*n0 # Adjusting the total number of objects to the defined value of the interstellar number-density
    
    total_number=int(np.floor(np.max(N_r))) # total number of objects in the population
    
    if total_number>0: # if there are ISOs in the model sphere
        
        # ODREDJIVANJE r_arr
        ur=np.random.random(int(np.floor(np.max(N_r))))*np.max(N_r) 
            
        # Inverse Transform Sampling Method for r_arr
        # Inverse interpolation to obtaine ISO's helicentric distance according to Eq. 27 (cubic B-spline interpolation is used)
        tck = interpolate.splrep(N_r, r_arr, s=0)
        rs = interpolate.splev(ur, tck, der=0) # the set of helicentric distances
        
        vs=np.zeros_like(rs)
        Bs=np.zeros_like(rs)
        ls=np.zeros_like(rs)
        bs=np.zeros_like(rs)
        
        qs=np.zeros_like(rs)
    
        es1=np.zeros_like(rs)
        incs=np.zeros_like(rs)
        Omegas=np.zeros_like(rs)
        omegas=np.zeros_like(rs)
        
        
        p_rv_cdf=np.zeros(len(v))
        p_rl_cdf=np.zeros(len(l_arr))
        p_rb_cdf=np.zeros(len(b_arr))
        for i in range(total_number):
            
            # ODREDJIVANJE v
            if np.mod(i,1000)==0:
                print('{} od {}'.format(i, total_number))
                
            tck = interpolate.splrep(r_arr, p_r, s=0)
            pr_rs = interpolate.splev(rs[i], tck, der=0) # maximum value or the random number uv (se below and section 5.2)
     
            # Racunamo v za dato rs. Treba nam raspodela koja je kumulativna po v, uslovna po r_arr, i marginalna po B, l, b
        
            # uslovna za v po r_arr, a marginalna po l,b,B
            
            p_rv=p_v/2*(np.sqrt(1+2*mu/rs[i]/v_arr**2)+np.sqrt(1+2*mu/rs[i]/v_arr**2-R_reff**2/rs[i]**2*(1+2*mu/v_arr**2/R_reff)))
                
            # Numerical integration with respect to v according to Eq. 19
            cum=0
            for j in range(1,len(v)):
                cum=cum+(p_rv[j]+p_rv[j-1])/2*dv
                p_rv_cdf[j]=cum           
            
            v_try=0
            while v_try<v_min or v_try>v_max:
                # Inverse Transform Sampling Method for v
                uv=np.random.rand()*pr_rs #  random number in the range [0,pr(rs[k])]
                # Inverse interpolation to obtaine ISO's interstellar velocity according to Eq. 28 (cubic B-spline interpolation is used)
                tck = interpolate.splrep(p_rv_cdf, v_arr, s=0)
                v_try = interpolate.splev(uv, tck, der=0) # interstellar velocity which corresponds to rs[k]
            vs[i]=v_try
            
            # ODREDJIVANJE B
            
            # calculating pr(rs, vs) - see section 5.2 (Eqs. 29 and 30)
            tck = interpolate.splrep(v_arr, p_rv, s=0)
            pr_rs_vs = interpolate.splev(vs[i], tck, der=0) # maximum value or the random number uB (se below and section 5.2)
            
            # calculating pv(vs) used in Eqs. 31. It is here obtained by interpolation, but it is better replace this
            # with the original function for calculating pv (the one which is used to calculate input parametar pv for the whole function)
            tck = interpolate.splrep(v_arr, p_v, s=0)
            pv_vs = interpolate.splev(vs[i], tck, der=0) # this is pv(vs) 
            
            # limiting value of the function p_rv which defines if the ISO is Sun impactor or not (first equation in Eq. 25)
            p_rv_lim=pv_vs/2*(np.sqrt(1+2*mu/rs[i]/vs[i]**2)+np.sqrt(1+2*mu/rs[i]/vs[i]**2-R_reff**2/rs[i]**2*(1+2*mu/vs[i]**2/R_reff)))  
    
            uB=np.random.rand()*pr_rs_vs
            
            # Inverse Transform Sampling Method for v according to Eqs. 31
            f1=np.sqrt(1+2*mu/rs[i]/vs[i]**2)
            f2=np.sqrt(1+2*mu/rs[i]/vs[i]**2)+np.sqrt(1+2*mu/rs[i]/vs[i]**2-R_reff**2/rs[i]**2*(1+2*mu/vs[i]**2/R_reff))
    
            if uB<p_rv_lim: # Sun impactor
                Bs[i]=rs[i]*np.sqrt(f1**2-((f1*pv_vs-2*uB)/pv_vs)**2)
            else: # passer
                Bs[i]=rs[i]*np.sqrt(f1**2-(f2*pv_vs-2*uB)**2/4/pv_vs**2)
                
            # Odredjivanje l
            
             # uslovna po r_arr,v,B, a marginalna po B
             
            p_vs_l_b=p_v_l_b(vs[i], l_mesh, b_mesh, sigma_vx, -u_Sun, sigma_vy, -v_Sun, sigma_vz, -w_Sun, vd, va)
             
            pr_rs_vs_Bs=Bs[i]*p_vs_l_b*vs[i]/(2*rs[i]*np.sqrt(vs[i]*rs[i]*2*mu*rs[i]-Bs[i]*vs[i]**2))#*np.cos(b_mesh)
            
            pr_rs_vs_Bs=sum(pr_rs_vs_Bs)*db
            
            cum=0
            for j in range(1,len(l_arr)):
                cum=cum+(pr_rs_vs_Bs[j]+pr_rs_vs_Bs[j-1])/2*dl
                p_rl_cdf[j]=cum 
             
            ul=np.random.rand()*max(p_rl_cdf)
            
            tck = interpolate.splrep(p_rl_cdf, l_arr, s=0)
            ls[i] = interpolate.splev(ul, tck, der=0) # interstellar velocity which corresponds to rs[k]
    
    
            # Odredjivanje b
            
             # uslovna po r_arr,v,B,l 
             
            p_vsl_b=p_v_l_b(vs[i], ls[i], b_arr, sigma_vx, -u_Sun, sigma_vy, -v_Sun, sigma_vz, -w_Sun, vd, va)
             
            pr_rs_vs_Bs_ls=Bs[i]*p_vsl_b*vs[i]/(2*rs[i]*np.sqrt(vs[i]*rs[i]*2*mu*rs[i]-Bs[i]*vs[i]**2))#*np.cos(b_arr)
            
            cum=0
            for j in range(1,len(b_arr)):
                cum=cum+(pr_rs_vs_Bs_ls[j]+pr_rs_vs_Bs_ls[j-1])/2*db
                p_rb_cdf[j]=cum 
             
            ub=np.random.rand()*max(p_rb_cdf)
            
            tck = interpolate.splrep(p_rb_cdf, b_arr, s=0)
            bs[i] = interpolate.splev(ub, tck, der=0) # interstellar velocity which corresponds to rs[k]
    
    
            #normala na orbitu
            
            xx=np.cos(bs[i])*np.cos(ls[i]);
            yy=np.cos(bs[i])*np.sin(ls[i]);
            zz=np.sin(bs[i]);
            
            rr=np.array([xx,yy,zz]);
            
            
            uu=np.array([0, -zz, yy]);
            uu=uu/np.linalg.norm(uu);
            
            vv=np.array([yy**2+zz**2,-xx*yy, -xx*zz]);
            
            vv=vv/np.linalg.norm(vv);
            
            
            # inklinacija
            
            angle=np.random.rand()*np.pi*2
            
            
            orbital_plane_normal=uu*np.cos(angle)+vv*np.sin(angle);
            
            incs[i]=np.arccos(np.dot(orbital_plane_normal, np.array([0,0,1])))
    
    
            
            # longituda cvora
            
            node = np.cross(np.array([0, 0, 1]), orbital_plane_normal)
            
    #        Omegas[i] = np.arccos(node[0] / np.linalg.norm(node))
            
            Omegas[i] = np.arctan2(node[1], node[0])
            
            
            # argument perihela
            h= orbital_plane_normal*Bs[i]*vs[i]
            
            nn = np.cross(np.array([0, 0, 1]), h)
            
            
            r0=np.array([np.cos(bs[i])*np.cos(ls[i]), np.cos(bs[i])*np.sin(ls[i]), np.sin(bs[i])])
            
              
            e_vector = np.cross(-vs[i]*r0, h) / mu - r0
    
    
            omegas[i] = np.arccos(np.linalg.linalg.dot(nn, e_vector) / np.linalg.norm(nn) / np.linalg.norm(e_vector))
            
            
            if e_vector[2] < 0:
                omegas[i] = 2 * np.pi - omegas[i]
                
        semi_major_axis=-mu/vs**2
        qs=semi_major_axis+np.sqrt(semi_major_axis**2+Bs**2)
        es=np.sqrt(1+Bs**2/semi_major_axis**2)
    
        Omegas=np.mod(Omegas,2*np.pi)
    
        sign=np.random.random(len(rs))
        sign[sign<0.5]=-1
        sign[sign>0.5]=1
        
        # calculating true anomaly according to Eq. 9
        fs=sign*np.arccos((semi_major_axis*(1-es**2)/rs-1)/es)
    
    return (qs, es, incs, omegas, Omegas, fs)