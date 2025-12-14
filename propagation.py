import numpy as np
from auxiliary_functions import true2ecc, kepler, ecc2true
from synthetic_population import synthetic_population_stationary
import matplotlib.pyplot as plt
y2s = 31558196.01538755
au=1.495978707e11 # astronomical unit
mu=1.32712440042e20  # standard gravitional parameter of the Sun


rm = 5.2
n0 = 5
T_stat = 3.3
t10 = 1.1
t20 = 3.2

q, e, f, inc, Omega, omega,q_add, e_add, f_add, inc_add, node_add, argument_add, t_add, r_add = synthetic_population_stationary(T_stat, rm=rm, n0=n0, v_min=1e3, v_max=2e5, 
                                                                u_Sun=1e4, v_Sun=1.1e4, w_Sun=7e3, 
                                                                sigma_vx=3.1e4, sigma_vy=2.3e4, sigma_vz=1.6e4, 
                                                                vd=np.deg2rad(30), va=0, R_reff=696340000.,
                                                                speed_resolution=50, angle_resolution=180, B_resolution = 50, dr=0.1)



#q, e, f, inc, Omega, omega,q_add, e_add, f_add, inc_add, node_add, argument_add, t_add, r_add = synthetic_population_stationary(T_stat, rm=rm, n0=n0, v_min=1e3, v_max=2e5, 
#                                                                u_Sun=0, v_Sun=0, w_Sun=0, 
#                                                                sigma_vx=2e4, sigma_vy=2e4, sigma_vz=2e4, 
#                                                                vd=np.deg2rad(0), va=0, R_reff=696340000.,
#                                                                speed_resolution=20, angle_resolution=180, B_resolution = 50, dr=0.1)



#q_in = q[:total_number]
#q_out = q[total_number:]
#
#
#f_in = f[:total_number]
#f_out = f[total_number:]
#
#
#q = q[total_number:]
#e = e[total_number:]
#f = f[total_number:]
#inc = inc[total_number:]
#Omega = Omega[total_number:]
#omega = omega[total_number:]
#
#
#a = q / (1-e)
#
#r = a * (1-e**2) / (1 + e * np.cos(f))



#q = q[:total_number]
#e = e[:total_number]
#f = f[:total_number]
#inc = inc[:total_number]
#Omega = Omega[:total_number]
#omega = omega[:total_number]




#e1, e2 = synthetic_population_stationary(1., rm=1, n0=10, v_min=1e3, v_max=2e5, 
#                                                                u_Sun=1e4, v_Sun=1.1e4, w_Sun=7e3, 
#                                                                sigma_vx=3.1e4, sigma_vy=2.3e4, sigma_vz=1.6e4, 
#                                                                vd=np.deg2rad(7), va=0, R_reff=696340000.,
#                                                                speed_resolution=100, angle_resolution=90, dr=0.1)




#q = q_in
# constants



a = q / (1-e) * au
mm = np.sqrt(-mu/a**3)

E0 = np.zeros_like(f)

for i in range(len(f)):
    E0[i] = true2ecc(f[i], e[i])
    
M0 = e * np.sinh(E0) - E0


coshE = 1/e * (1 - rm/(a/au))
coshE = np.maximum(coshE, 1)

E_cr = np.arccosh(coshE)

M_cr = e * np.sinh(E_cr) - E_cr


T_ulaz = (M_cr - M0) / mm









T = np.linspace(0, T_stat, 300) * y2s * 1.2



r = a * (1 - e * np.cosh(E0))

print('rmax', np.nanmax(r) / au)
N_in = np.zeros_like(T)





ft = np.zeros_like(q)
Et = np.zeros_like(q)
t1 = t10
t2 = t20
for i, t in enumerate(T):
    
    if np.mod(i, 10)==0:
        print (f'{i} od {len(T)}')
    
    M  = M0 + mm * t
    
    for j in range(len(M)):
        
        Et[j] = kepler(e[j], M[j])
        ft[j] = ecc2true(Et[j], e[j])
        
        
    r = a * (1 - e * np.cosh(Et))
    
    N_in[i] = np.sum(r < rm * au)
    
    
    if t > t1* y2s:
        q1 = q[r/au<rm]
        e1 = e[r/au<rm]
        f1 = ft[r/au<rm]
        inc1 = inc[r/au<rm]
        Omega1 = Omega[r/au<rm]
        omega1 = omega[r/au<rm]
        
        M1 = M[r/au<rm]
        E1 = Et[r/au<rm]
        
        t1 = 2 * T_stat* y2s
        
        
        
    if t > t2* y2s:
        q2 = q[r/au<rm]
        e2 = e[r/au<rm]
        f2 = ft[r/au<rm]
        inc2 = inc[r/au<rm]
        Omega2 = Omega[r/au<rm]
        omega2 = omega[r/au<rm]
        
        M2 = M[r/au<rm]
        E2 = Et[r/au<rm]
        
        t2 = 2 * T_stat* y2s
        
    

plt.plot(T/y2s, N_in, 'k', linewidth = 5)
plt.plot([T_stat, T_stat], [0, 1.2 * N_in[0]], '--r', linewidth = 3)
plt.xlabel('vreme (god)', fontsize = 32)
plt.ylabel('broj objekata u sferi', fontsize = 32)
plt.xticks(fontsize = 24)
plt.yticks(fontsize = 24)
plt.grid()
        

plt.figure()
plt.subplot(231)
plt.hist(q[:len(q) - len(q_add)], np.linspace(0, rm, 20), density = True, alpha = 0.5, label = 'inicijalno')
plt.hist(q1, np.linspace(0, rm, 20), density = True, alpha = 0.5, label = f't = {t10} god')
plt.hist(q2, np.linspace(0, rm, 20), density = True, alpha = 0.5, label = f't = {t20} god')
plt.title('perihel', fontsize = 12)
plt.legend(fontsize = 12)

plt.subplot(232)
plt.hist(e[:len(q) - len(q_add)], np.linspace(1, 20, 20), density = True, alpha = 0.5, label = 'inicijalno')
plt.hist(e1, np.linspace(1, 20, 20), density = True, alpha = 0.5, label = f't = {t10} god')
plt.hist(e2, np.linspace(1, 20, 20), density = True, alpha = 0.5, label = f't = {t20} god')
plt.title('ekscentricnost', fontsize = 12)
plt.legend(fontsize = 12)

plt.subplot(233)
plt.hist(f[:len(q) - len(q_add)], np.linspace(-np.pi, np.pi, 20), density = True, alpha = 0.5, label = 'inicijalno')
plt.hist(f1, np.linspace(-np.pi, np.pi, 20), density = True, alpha = 0.5, label = f't = {t10} god')
plt.hist(f2, np.linspace(-np.pi, np.pi, 20), density = True, alpha = 0.5, label = f't = {t20} god')
plt.title('anomalija', fontsize = 12)
plt.legend(fontsize = 12)

plt.subplot(234)
plt.hist(inc[:len(q) - len(q_add)], np.linspace(0, np.pi, 20), density = True, alpha = 0.5, label = 'inicijalno')
plt.hist(inc1, np.linspace(0, np.pi, 20), density = True, alpha = 0.5, label = f't = {t10} god')
plt.hist(inc2, np.linspace(0, np.pi, 20), density = True, alpha = 0.5, label = f't = {t20} god')
plt.title('inklinacija', fontsize = 12)
plt.legend(fontsize = 12)

plt.subplot(235)
plt.hist(Omega[:len(q) - len(q_add)], np.linspace(0, 2*np.pi, 20), density = True, alpha = 0.5, label = 'inicijalno')
plt.hist(Omega1, np.linspace(0, 2*np.pi, 20), density = True, alpha = 0.5, label = f't = {t10} god')
plt.hist(Omega2, np.linspace(0, 2*np.pi, 20), density = True, alpha = 0.5, label = f't = {t20} god')
plt.title('Cvor', fontsize = 12)
plt.legend(fontsize = 12)

plt.subplot(236)
plt.hist(omega[:len(q) - len(q_add)], np.linspace(0, 2*np.pi, 20), density = True, alpha = 0.5, label = 'inicijalno')
plt.hist(omega1, np.linspace(0, 2*np.pi, 20), density = True, alpha = 0.5, label = f't = {t10} god')
plt.hist(omega2, np.linspace(0, 2*np.pi, 20), density = True, alpha = 0.5, label = f't = {t20} god')
plt.title('argument', fontsize = 12)
plt.legend(fontsize = 12)


