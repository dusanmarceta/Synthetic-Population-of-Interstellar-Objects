import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#            initialization sphere and initialization time
# =============================================================================

def initialization(v_min, v_max, r0):
    
    Rs=696340.
    G=1.32712440042e20
    au=1.495978707e11

    q=np.linspace(Rs,r0,100000)
    T=np.zeros(len(q))
    a=-G/v_min**2
    sk = np.sqrt(G / np.abs(a ** 3))  # mean motion [rad/s]
    for i in range(len(q)):
        H0 = np.arccosh((a-r0)/(a-q[i])) # limiting eccentric anomaly when object is at q_max from the Sun    
        # limiting mean anomaly when object is at q_max from the Sun
        T[i] = 2*(((a-q[i])/a * np.sinh(H0) - H0)/sk)
      
    index = (np.argwhere(T == np.max(np.max(T)))).flatten()[0]
    
    t_init=T[index]
    
    rm=r0
    r_init = 1.1*rm
    dr_init = 0.1*au
    t = 0.
    dt = 864000.
    
    while t <  t_init:
        t = 0
        r_init=r_init+dr_init
    
        x = r_init
        v = v_max
        while x > rm:
            ax = G / x ** 2
            v += ax * dt
            x -= v * dt
            t += dt
            
    return t_init, r_init




def cart2orb(x, y, z, vx, vy, vz, G):
    # =============================================================================
    # calculates keplerian orbital elements from cartesian
    # position and velocity
    # Input:
    # x, y, z - [m] cartesian coordinates
    # vx, vy, vz - [m/s] cartesian velocities
    # G - central body gravity parameter (m**3/s**2)
    #
    # Output
    # o - argument of perihelion [degrees]
    # O - longitude of ascending node [degrees]
    # inc - inclination [degrees]
    # a - semimajor axis [meters]
    # e - eccentricity
    # M - mean anomaly [radians]
    # =============================================================================
    r = np.array([x, y, z])
    v = np.array([vx, vy, vz])

    # momenat kolicine kretanja
    h = np.cross(r, v)

    # vektor ekscentriciteta
    e = np.cross(v, h) / G - r / np.linalg.norm(r)

    # vektor u pravcu uzlaznog cvora
    n = np.cross(np.array([0, 0, 1]), h)

    # prava anomalija
    nu = np.arccos(np.linalg.linalg.dot(e, r) / np.linalg.norm(e) / np.linalg.norm(r))

    if np.linalg.linalg.dot(r, v) < 0 and np.linalg.norm(e)<1.:
        nu = 2 * np.pi - nu
    elif np.linalg.linalg.dot(r, v) < 0 and np.linalg.norm(e)>1.:
        nu=-nu
        

    # nagib
    ink = np.arccos(h[2] / np.linalg.norm(h))
    
            
    #        Omegas[i] = np.arccos(node[0] / np.linalg.norm(node))
            
    O = np.arctan2(n[1], n[0])
            

#    # longituda cvora
#    O = np.arccos(n[0] / np.linalg.norm(n))
#    if n[1] < 0:
#        O = 2 * np.pi - O

    # argument pericentra
    o = np.arccos(np.linalg.linalg.dot(n, e) / np.linalg.norm(n) / np.linalg.norm(e))
    if e[2] < 0:
        o = 2 * np.pi - o

    # ekscentricitet
    e = np.linalg.norm(e)

    # ekscentricna anomalija
    E = true2ecc(nu, e)
    print('EEE',E)
    
    # srednja anomalija
    if e < 1:
        M = E - e * np.sin(E)
    else:
        M = e * np.sinh(E) - E

    # poluosa
    a = 1 / (2 / np.linalg.norm(r) - np.linalg.norm(v) ** 2 / G)

    return np.rad2deg(o), np.rad2deg(O), np.rad2deg(ink), e, a, M, nu


def cart2orb(r, v, G):
    # =============================================================================
    # calculates keplerian orbital elements from cartesian
    # position and velocity
    # Input:
    # x, y, z - [m] cartesian coordinates
    # vx, vy, vz - [m/s] cartesian velocities
    # G - central body gravity parameter (m**3/s**2)
    #
    # Output
    # o - argument of perihelion [degrees]
    # O - longitude of ascending node [degrees]
    # inc - inclination [degrees]
    # a - semimajor axis [meters]
    # e - eccentricity
    # M - mean anomaly [radians]
    # =============================================================================

    # momenat kolicine kretanja
    h = np.cross(r, v, axis=1)



    # vektor ekscentriciteta
    e = np.cross(v, h, axis=1) / G - np.transpose(np.transpose(r) / np.linalg.norm(r, axis=1))
    

    n = np.cross(np.array([0, 0, 1]), h)
    
    # prava anomalija
    nu = np.arccos(np.transpose(np.transpose((e*r).sum(axis=1)) / np.linalg.norm(e, axis=1) / np.linalg.norm(r, axis=1)))

    nu[np.logical_and((r*v).sum(axis=1)<0, np.linalg.norm(e, axis=1)<1)]=2*np.pi-nu[np.logical_and((r*v).sum(axis=1)<0, np.linalg.norm(e, axis=1)<1)]
    nu[np.logical_and((r*v).sum(axis=1)<0, np.linalg.norm(e, axis=1)>1)]=-nu[np.logical_and((r*v).sum(axis=1)<0, np.linalg.norm(e, axis=1)>1)]
    
    # nagib
    inc = np.arccos(np.transpose(h)[2] / np.linalg.norm(h, axis=1))

    # longituda cvora
    O = np.arccos(np.transpose(n)[0] / np.linalg.norm(n, axis=1))
    
    # argument pericentra
    o = np.arccos((n*e).sum(axis=1) / np.linalg.norm(n, axis=1) / np.linalg.norm(e, axis=1))
    
    
    o[np.transpose(e)[2]<0] = 2 * np.pi - o[np.transpose(e)[2]<0] 
    O[np.transpose(n)[1]<0] = 2 * np.pi - O[np.transpose(n)[1]<0]

    # ekscentricitet
    e = np.linalg.norm(e, axis=1)

    # ekscentricna anomalija
    E = true2ecc(nu, e)

    # srednja anomalija
    M=np.zeros(len(e))
    M[e<1]=E[e<1] - e[e<1] * np.sin(E[e<1])
    M[e>1]= e[e>1] * np.sinh(E[e>1]) - E[e>1]

    a = 1 / (2 / np.linalg.norm(r, axis=1) - np.linalg.norm(v, axis=1) ** 2 / G)

    return e, a, o, O, inc, M


def true2ecc(f, e):
    # =============================================================================
    # converts true anomaly to eccentric (or hyperbolic) anomaly
    # Input:
    # f [radians] - true anomaly
    # Output:
    # eccentric (or hyperbolic anomaly) [radians]
    # =============================================================================
    ecc=np.zeros(len(f))
    ecc[e>1]=2 * np.arctanh(np.sqrt((e[e>1] - 1) / (e[e>1] + 1)) * np.tan(f[e>1] / 2))
    ecc[e<1]= np.arctan2(np.sqrt(1 - e[e<1] ** 2) * np.sin(f[e<1]), e[e<1] + np.cos(f[e<1]))
    
    return ecc


def mean_anomaly(M0, epoch0, a, epoch, G):
    # =============================================================================
    # Calculates mean anomaly for a given epoch
    # Input:
    # epoch0 [modified julian date]- epoch of mean anomaly M0
    # M0 [radians] - mean anomaly for epoch0
    # a [m] - semi major axis
    # epoch [modified julian date] - epoch for which the mean anomaly is calculated
    # G [m ** 3 / s ** 2] - central body gravity parameter
    # Output:
    # mean anomaly for epoch [radians]
    # =============================================================================
    n = np.sqrt(G / np.abs(a) ** 3)  # mean motion [rad/s]

    return (epoch - epoch0) * 86400.0 * n + M0
