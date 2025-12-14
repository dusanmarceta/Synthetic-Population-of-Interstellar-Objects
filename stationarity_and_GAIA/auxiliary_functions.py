import numpy as np
from scipy.interpolate import CubicSpline

# =============================================================================
# constants
# =============================================================================
mu = 1.3271244e+20
au = 149597870700.0


def orb2cart(o, O, inc, e, a, E, G):
    # =============================================================================
    # calculates cartesian coordinates (in meters) from orbital elements for
    # elliptic and hyperbolic orbit depanding on eccentricity
    #
    # input:
    # o - argument of perihelion (rad)
    # O - longitude of ascending node (rad)
    # inc - inclination (rad)
    # a - semimajor axis (meters) (periapsis distance if parabolic orbit)
    # e - eccentricity
    # E - eccentric anomaly (radians) (true anomaly if parabolic orbit)
    # Output:
    # x,y,z [meters] - cartesian coordinates
    # =============================================================================
    
    if e > 1:  # hyperbolic orbit
        r = a * (1 - e * np.cosh(E))  # heliocentric distance
        f = np.mod(ecc2true(E, e), 2 * np.pi)  # true anomaly
        xt = -np.sqrt(-a * G) / r * np.sinh(E)  # minus sign to chose appropriate branch of hyperbola
        yt = np.sqrt(-a * G * (e ** 2 - 1)) / r * np.cosh(E)

    elif e < 1:  # elliptic orbit
        r = a * (1 - e * np.cos(E))  # helicentric distance
        f = np.mod(ecc2true(E, e), 2 * np.pi)  # true anomaly
        xt = -np.sqrt(G * a) / r * np.sin(E)
        yt = np.sqrt(G * a * (1 - e ** 2)) / r * np.cos(E)



    # cartesian coordinates
    x = r * (np.cos(O) * np.cos(o + f) - np.sin(O) * np.cos(inc) * np.sin(o + f))
    y = r * (np.sin(O) * np.cos(o + f) + np.cos(O) * np.cos(inc) * np.sin(o + f))
    z = r * (np.sin(inc) * np.sin(o + f))

    # cartesian components (ecliptical coordinate system)
    vx = xt * (np.cos(o) * np.cos(O) - np.sin(o) * np.cos(inc) * np.sin(O)) \
         - yt * (np.sin(o) * np.cos(O) + np.cos(o) * np.cos(inc) * np.sin(O))

    vy = xt * (np.cos(o) * np.sin(O) + np.sin(o) * np.cos(inc) * np.cos(O)) \
         - yt * (np.sin(o) * np.sin(O) - np.cos(o) * np.cos(inc) * np.cos(O))

    vz = xt * np.sin(o) * np.sin(inc) + yt * np.cos(o) * np.sin(inc)


    return x, y, z, vx, vy, vz


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
    # o - argument of perihelion [rad]
    # O - longitude of ascending node [rad]
    # inc - inclination [rad]
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

    if np.linalg.linalg.dot(r, v) < 0 and np.linalg.norm(e) < 1.:
        nu = 2 * np.pi - nu
    elif np.linalg.linalg.dot(r, v) < 0 and np.linalg.norm(e) > 1.:
        nu = -nu

    # nagib
    ink = np.arccos(h[2] / np.linalg.norm(h))

    # longituda cvora
    O = np.arccos(n[0] / np.linalg.norm(n))
    if n[1] < 0:
        O = 2 * np.pi - O

    # argument pericentra
    o = np.arccos(np.linalg.linalg.dot(n, e) / np.linalg.norm(n) / np.linalg.norm(e))
    if e[2] < 0:
        o = 2 * np.pi - o

    # ekscentricitet
    e = np.linalg.norm(e)

    # ekscentricna anomalija
    E = true2ecc(nu, e)

    # srednja anomalija
    #    if e < 1:
    #        M = E - e * np.sin(E)
    #    else:
    #        M = e * np.sinh(E) - E

    # poluosa
    a = 1 / (2 / np.linalg.norm(r) - np.linalg.norm(v) ** 2 / G)

    return o, O, ink, e, a, E, nu


# =============================================================================
#                                     EARTH
# =============================================================================
def earth(MJD):
    """ 
    Calculates cartesian heliocentric ecliptic coordinates, 
    velocity and acceleration components of Earth for specified
    Modified Julian Date (MJD).
    
    Input:
    MJD - Modified Julian Date
    
    Output:
    x,y,z (m) - cartesian ecliptic coordinates
    vx,vy,vz (m/s) - cartesian ecliptic velocity components
    accx,accy,accz (m/s**2) - cartesian ecliptic velocity components
    """

    # Earth orbital elements
    o = 1.99330267;
    O = -0.1965350;
    inc = 0.
    e = 0.01671022;
    a = 149597870700.0

    # Gravitational parameter of the Sun
    G = 1.3271244e+20

    # Earth mean motion
    n = 1.9909836745e-07

    # mean anomaly
    M = (MJD - 51545) * 0.01720209895 - 0.239869  # MJD=51545, M=-0.239869 za J2000.0

    # eccentric anomaly
    E = kepler(e, M, 1e-6)
    E_dot = n / (1 - e * np.cos(E))

    # helicentric distance
    r = a * (1 - e * np.cos(E))
    r_dot = a * e * n * np.sin(E) / (1 - e * np.cos(E))

    # true anomaly
    f = np.mod(ecc2true(E, e), 2 * np.pi)  # true anomaly

    # velocity components in orbital coordinate system
    xt = -np.sqrt(G * a) / r * np.sin(E)
    yt = np.sqrt(G * a * (1 - e ** 2)) / r * np.cos(E)

    # acceleration components in orbital coordinate system
    xt_dot = np.sqrt(G * a) * (r_dot * np.sin(E) - r * E_dot * np.cos(E)) / r ** 2
    yt_dot = -np.sqrt(G * a * (1 - e ** 2)) * (r * E_dot * np.sin(E) - r_dot * np.cos(E)) / r ** 2

    # heliocentric coordinates
    x = r * (np.cos(O) * np.cos(o + f) - np.sin(O) * np.cos(inc) * np.sin(o + f))
    y = r * (np.sin(O) * np.cos(o + f) + np.cos(O) * np.cos(inc) * np.sin(o + f))
    z = r * (np.sin(inc) * np.sin(o + f))

    # cartesian velocity components (ecliptical coordinate system)
    vx = xt * (np.cos(o) * np.cos(O) - np.sin(o) * np.cos(inc) * np.sin(O)) \
         - yt * (np.sin(o) * np.cos(O) + np.cos(o) * np.cos(inc) * np.sin(O))

    vy = xt * (np.cos(o) * np.sin(O) + np.sin(o) * np.cos(inc) * np.cos(O)) \
         - yt * (np.sin(o) * np.sin(O) - np.cos(o) * np.cos(inc) * np.cos(O))

    vz = xt * np.sin(o) * np.sin(inc) + yt * np.cos(o) * np.sin(inc)

    accx = xt_dot * (np.cos(o) * np.cos(O) - np.sin(o) * np.cos(inc) * np.sin(O)) \
           - yt_dot * (np.sin(o) * np.cos(O) + np.cos(o) * np.cos(inc) * np.sin(O))

    accy = xt_dot * (np.cos(o) * np.sin(O) + np.sin(o) * np.cos(inc) * np.cos(O)) \
           - yt_dot * (np.sin(o) * np.sin(O) - np.cos(o) * np.cos(inc) * np.cos(O))

    accz = xt_dot * np.sin(o) * np.sin(inc) + yt_dot * np.cos(o) * np.sin(inc)

    return x, y, z, vx, vy, vz, accx, accy, accz


# =============================================================================
#                                   GEOCENTRIC
# =============================================================================
def geocentric_coor(o, O, inc, e, a, E, MJD):
    # =============================================================================
    # calculates cartesian geocentric ecliptic coordinates
    # Input
    # o - argument of perihelion(degrees)
    # O - longitude of ascending node (degrees)
    # inc - inclination (degrees)
    # a - semi-major axis (meters)
    # e - eccentricity
    # E - eccentric anomaly (radians)
    # MJD - Modified Julian Date
    # accuracy for Kepler equation for earth (i.e. 1e-6)
    # Output:
    # x,y,z [meters] - geocentric cartesian coordinates
    # =============================================================================
    # heliocentric coordinates of an object
    x_o, y_o, z_o, vx_o, vy_o, vz_o, accx_o, accy_o, accz_o = orb2cart(o, O, inc, e, a, E)

    # heliocentric coordinates of Earth
    x_e, y_e, z_e, vx_e, vy_e, vz_e, accx_e, accy_e, accz_e = earth(MJD)

    # geocentric vectors of the object
    x = x_o - x_e
    y = y_o - y_e
    z = z_o - z_e
    vx = vx_o - vx_e
    vy = vy_o - vy_e
    vz = vz_o - vz_e
    accx = accx_o - accx_e
    accy = accy_o - accy_e
    accz = accz_o - accz_e

    return x, y, z, vx, vy, vz, accx, accy, accz


# =============================================================================
#                       SPHERICAL COORDINATE SYSTEM
# =============================================================================
def spherical_coor(x, y, z):
    # =============================================================================
    # calculates spherical coordinates
    # Input
    # x,y,z - cartesian coordinates
    # Output:
    # long, lat [degrees] - spherical coordinates
    # =============================================================================
    # geocentric spherical coordinates
    long = np.mod(np.arctan2(y, x), 2 * np.pi)
    lat = np.arctan(z / np.sqrt(x ** 2 + y ** 2))

    # converts to degrees
    return long, lat


def spherical_vel(long, lat, r, vx, vy, vz):
    # =============================================================================
    # Calculates spherical velocities
    # Input:
    # long, lat [degrees] - spherical coordinates
    # r [meters] - distance
    # vx,vy,vz [m/s] - cartesian velocities
    # Output:
    # long_dot, lat_dot [degrees/day] - spherical velocities
    # =============================================================================

    # projections of velocity vector
    v_long = -np.sin(long) * vx + np.cos(long) * vy
    v_lat = -np.sin(lat) * np.cos(long) * vx - np.sin(lat) * np.sin(long) * vy + np.cos(lat) * vz

    # time derivatives
    long_dot = v_long / np.cos(lat) / r
    lat_dot = v_lat / r

    # converts to degrees per day
    return long_dot, lat_dot


def spherical_acc(long, lat, r, long_dot, lat_dot, vx, vy, vz, ax, ay, az):
    # =============================================================================
    # Calculates spherical accelerations
    # Input
    # long, lat [degrees] - spherical coordinates
    # r [meters] - distance
    # long_dot, lat_dot [degrees/day] - spherical velocities
    # vx,vy,vz [m/s] - cartesian velocities
    # ax,ay,az [m/s**2] - cartesian accelerations
    #
    # Output:
    # long_d_dot, lat_d_dot [degrees/day**2] - spherical accelerations
    # =============================================================================

    # projections of velocity vector
    v_long = -np.sin(long) * vx + np.cos(long) * vy
    v_lat = -np.sin(lat) * np.cos(long) * vx - np.sin(lat) * np.sin(long) * vy + np.cos(lat) * vz
    v_r = np.cos(lat) * np.cos(long) * vx + np.cos(lat) * np.sin(long) * vy + np.sin(lat) * vz

    # projections of acceleration vector
    a_long = -np.sin(long) * (ax + long_dot * vy) + np.cos(long) * (ay - long_dot * vx)

    a_lat = -lat_dot * (np.cos(lat) * np.cos(long) * vx + np.cos(lat) * np.sin(long) * vy + np.sin(lat) * vz) \
            + long_dot * (np.sin(lat) * np.sin(long) * vx - np.sin(lat) * np.cos(long) * vy) \
            - np.sin(lat) * np.cos(long) * ax - np.sin(lat) * np.sin(long) * ay + np.cos(lat) * az

    # second order time derivatives

    long_d_dot = (a_long * r * np.cos(lat) - v_long * (
            v_r * np.cos(lat) - r * lat_dot * np.sin(lat))) / r ** 2 / np.cos(lat) ** 2
    lat_d_dot = (a_lat * r - v_r * v_lat) / r ** 2

    # converts to degrees per day**2
    return long_d_dot, lat_d_dot


# =============================================================================
#                       OBSERVATIONAL PARAMETERS
# =============================================================================
def elongation(R, rhc, rgc):
    # =============================================================================
    # Calculates elongation of the object
    #
    # Input:
    # R - Earth distance to the Sun
    # rgc - geocentric distance of the object
    # rhc - heliocentric distance of the object
    #
    # Output:
    # elongation [degrees]
    # =============================================================================
    # elongation
    return np.rad2deg(np.arccos((rgc ** 2 + R ** 2 - rhc ** 2) / 2 / R / rgc))


def phase_angle(R, rhc, rgc):
    # =============================================================================
    # Calculates phase angle of the object
    #
    # Input:
    # R - Earth distance to the Sun
    # rgc - geocentric distance of the object
    # rhc - heliocentric distance of the object
    #
    # Output:
    # phase angle [degrees]
    # =============================================================================
    # phase angle
    return np.rad2deg(np.arccos((rgc ** 2 + rhc ** 2 - R ** 2) / 2 / rgc / rhc))


def absolute_magnitude_asteroid(D, albedo):
    # =============================================================================
    # Calculates absolute magnitude according to Harris, Alan W.; Harris, Alan W.(1997)
    # input:
    # D [m]- diameter
    # albedo - geometrical albedo
    # Output:
    # absolute magnitude
    # =============================================================================
    return 15.618 - 2.5 * np.log10(albedo) - 5 * np.log10(D / 1000)


def apparent_magnitude_asteroid(D, albedo, G, r_gc, r_hc, phase):
    # =============================================================================
    # Calculates apparent magnitude (Jewit et al. (2017), Bowell et al(1989))
    # Input:
    # D (m) - diameter
    # albedo - geometrical albedo
    # r_gc [au] - geocentric distance
    # r_hc [au]- heliocentric distance
    # phase [radians]- phase angle
    # G - slope parameter
    # Output:
    # Apparent visual magnitude of an asteroid
    # =============================================================================
    A = np.zeros(2)
    B = np.zeros(2)
    A[0] = 3.33
    A[1] = 1.87
    B[0] = 0.63
    B[1] = 1.22

    phi = np.exp(-A * np.power(np.tan(0.5 * phase), B))
    phase_function = 2.5 * np.log10((1 - G) * phi[0] + G * phi[1])
    


    H = absolute_magnitude_asteroid(D, albedo)
    return H + 5 * np.log10(r_gc) + 5 * np.log10(r_hc) - phase_function


def absolute_magnitude_comet(D, b1, b2):
    # =============================================================================
    # Calculates absolute magnitude of a comet according to Cook et al, 2016 (Eq.4)
    # input:
    # D (m))- diameter
    # b1, b2 - empirical parameters
    # Output:
    # absolute magnitude
    # =============================================================================
    return (np.log10(D/1000)-b2)/b1


def apparent_magnitude_comet(D, b1, b2, n, G, r_gc, r_hc, phase):
    # =============================================================================
    # Calculates apparent magnitude of a comet according to of a comet according to Cook et al, 2016 (Eq.5)
    # Input:
    # D (m) - diameter
    # b1, b2 - empirical parameters
    # n - brightening factor due to activity
    # G - slope parameter
    # r_gc (au) - geocentric distance
    # r_hc (au) - heliocentric distance
    # phase (radians) - phase angle
    # Output:
    # Apparent visual magnitude of a comet
    # =============================================================================
    A = np.zeros(2)
    B = np.zeros(2)
    A[0] = 3.33
    A[1] = 1.87
    B[0] = 0.63
    B[1] = 1.22

    phi = np.exp(-A * np.power(np.tan(0.5 * phase), B))
    phase_function = 2.5 * np.log10((1 - G) * phi[0] + G * phi[1])

    # reference asteroid from which comet cannot be fainter
    V_ast = apparent_magnitude_asteroid(D, 0.06, G, r_gc, r_hc, phase)

    # comet
    Hc = absolute_magnitude_comet(D, b1, b2)

    V = Hc + 2.5 * (n / 2 * np.log10(r_hc ** 2) + np.log10(r_gc ** 2)) - phase_function

    return np.minimum(V_ast, V)

#    return V






def max_hc_distance_asteroid(D, albedo, V_cut):
    # =============================================================================
    # Calculates maximum heliocentric distance where the object can be observed
    # Input:
    # m_cut:minimum apparent magnitude
    # D: diameter (m)
    # albedo
    # output:
    # max heliocentric distance where the object can be observed
    # =============================================================================
    """
    Acording to Jewit et al. (2017), Bowell et al(1989)
    V = H + 5 * np.log10(r_gc) + 5 * np.log10(r_hc) - phase_function
    for ideal situation we can set:
    phase_function = 0 (object in oposition)
    r_gc= r_hc -1 (object in oposition)

    Now we have
    V = H + 5*log10(r_hc(r_hc-1))

    for some critical V = V_cut we have
    r_hc^2 - r_hc - C = 0, where C = 10 ** ((V_cut - H) / 5)
    solvin quadratic equation wrt r_hc gives maximum heliocentric distance where the object can be observed
    """

    H = absolute_magnitude_asteroid(D, albedo)
    C = 10 ** ((V_cut - H) / 5)
    return (1 + np.sqrt(1 + 4 * C)) / 2


def max_hc_distance_comet(D, b1, b2, n, analog_asteroid_albedo, V_cut):
    # =============================================================================
    # Calculates maximum heliocentric distance where the object can be observed
    # Input:
    # m_cut:minimum apparent magnitude
    # D: diameter (m)
    # albedo
    # output:
    # max heliocentric distance where the object can be observed
    # =============================================================================
#    Hc = absolute_magnitude_comet(D, b1, b2)
#
#    def func(x, *data):
#        nn, CC = data
#        return x ** (nn + 2) - 2 * x ** (n + 1) + x ** n - CC # this follows from eq.5 from Cook, 2016
#
#    C = 10 ** (2 * (V_cut - Hc) / 5)


    r_min=max_hc_distance_asteroid(D, analog_asteroid_albedo, V_cut) # initial try to solve the above equation
    
    r_hc=np.arange(r_min, r_min + 10, 1)
#    print(r_hc)
    

    V = apparent_magnitude_comet(D, b1, b2, n, 0.15, r_hc-1, r_hc, 0)    
    cs = CubicSpline(V, r_hc)
    
    r_hc_max = cs(V_cut)
    
    return np.max([r_min, r_hc_max]) # greater distance for a comet and analog asteroid



def max_hc_distance_comet_array(D, b1, b2, n, analog_asteroid_albedo, V_cut):
    # =============================================================================
    # Calculates maximum heliocentric distance where the object can be observed
    # Input:
    # m_cut:minimum apparent magnitude
    # D: diameter (m)
    # albedo
    # output:
    # max heliocentric distance where the object can be observed
    # =============================================================================
#    Hc = absolute_magnitude_comet(D, b1, b2)
#
#    def func(x, *data):
#        nn, CC = data
#        return x ** (nn + 2) - 2 * x ** (n + 1) + x ** n - CC # this follows from eq.5 from Cook, 2016
#
#    C = 10 ** (2 * (V_cut - Hc) / 5)


    r_min=max_hc_distance_asteroid(D, analog_asteroid_albedo, V_cut) # initial try to solve the above equation
    
    
    
    r_hc = np.arange(1.1,100, 0.5)


    V = apparent_magnitude_comet(D[:, np.newaxis], b1, b2, n, 0.15, r_hc-1, r_hc, 0)
 
    ofset = 100

    B = np.arange(0, len(V)*ofset, ofset)
    
    # Add B to each row of A
    result = V + B[:, np.newaxis]
    
    result = result.ravel()
    
    xx=np.tile(r_hc, len(V))
    
    V_int = B + V_cut
    
    r_int = np.interp(V_int, result, xx)    
    return np.maximum(r_min, r_int)


# =============================================================================
#                               CONVERSIONS
# =============================================================================
def kepler(e, M, accuracy):
    # =============================================================================
    # solves Kepler equation using Newton-Raphson method
    # for elliptic and hyperbolic orbit depanding on eccentricity

    # Input:
    # e - eccentricity
    # M - mean anomaly (radians)
    # accuracy - accuracy for Newton-Raphson method (for example 1e-6)
    #
    # Output:
    # E [radians] - eccentric (hyperbolic) anomaly
    # =============================================================================
    if e > 1:  # hyperbolic orbit (GOODIN & ODELL, 1988)

        L = M / e
        g = 1 / e

        q = 2 * (1 - g)
        r = 3 * L
        s = (np.sqrt(r ** 2 + q ** 3) + r) ** (1 / 3)

        H00 = 2 * r / (s ** 2 + q + (q / s) ** 2)

        #    if np.abs(np.abs(M)-1)<0.01:
        if np.mod(np.abs(M), 0.5) < 0.01 or np.mod(np.abs(M), 0.5) > 0.49:  # numerical problem about this value
            E = (M * np.arcsinh(L) + H00) / (M + 1 + 0.03)  # initial estimate
        else:
            E = (M * np.arcsinh(L) + H00) / (M + 1)  # initial estimate

        delta = 1.0
        while abs(delta) > accuracy:
            f = M - e * np.sinh(E) + E
            f1 = -e * np.cosh(E) + 1
            delta = f / f1
            E = E - delta

    elif e < 1:  # elliptic orbit
        delta = 1.0
        E = M

        while abs(delta) > accuracy:
            f = E - e * np.sin(E) - M
            f1 = 1 - e * np.cos(E)
            delta = f / f1
            E = E - delta

    return E


# =============================================================================
#                           CONVERSIONS
# =============================================================================

# ecliptic to equatorial
def ecl2eq_spherical(long, lat):
    # =============================================================================
    # converts ecliptic to equatorial coordinates
    # Input:
    # long,lat [degrees] - ecliptic longitude, ecliptic latitude
    # Output:
    # alpha, delta [degrees] - right ascension, declination
    # =============================================================================

    eps = 0.409093  # ecliptic obliquity

    delta = np.arcsin(np.sin(eps) * np.sin(long) * np.cos(lat) + np.cos(eps) * np.sin(lat))
    sinus = (np.cos(eps) * np.sin(long) * np.cos(lat) - np.sin(eps) * np.sin(lat)) / np.cos(delta)
    kosinus = np.cos(long) * np.cos(lat) / np.cos(delta)
    alpha = np.arctan2(sinus, kosinus)

    return alpha, delta


def ecl2eq_cart(x, y, z):
    # =============================================================================
    # converts ecliptic to equatorial coordinates
    # Input:
    # long,lat [degrees] - ecliptic longitude, ecliptic latitude
    # Output:
    # alpha, delta [degrees] - right ascension, declination
    # =============================================================================

    eps = 0.409093  # ecliptic obliquity

    return (x, y * np.cos(eps) - z * np.sin(eps), y * np.sin(eps) + z * np.cos(eps))


# equatorial to ecliptic

def eq2ecl_spherical(alpha, delta):
    eps = 0.409093  # ecliptic obliquity

    lat = np.arcsin(np.sin(delta) * np.cos(eps) - np.cos(delta) * np.sin(eps) * np.sin(alpha))

    sinus = (np.sin(delta) * np.sin(eps) + np.cos(delta) * np.cos(eps) * np.sin(alpha)) / np.cos(lat)

    cosinus = np.cos(delta) * np.cos(alpha) / np.cos(lat)

    long = np.arctan2(sinus, cosinus)

    return (long, lat)


def eq2ecl_cart(x, y, z):
    eps = 0.409093  # ecliptic obliquity

    return (x, y * np.cos(eps) + z * np.sin(eps), -y * np.sin(eps) + z * np.cos(eps))


# ecliptic to galactic
def ecl2gal_spherical(long, lat):
    # =============================================================================
    # converts ecliptic to galactic coordinates
    # Input:
    # long,lat [degrees] - ecliptic longitude, ecliptic latitude
    # Output:
    # l, b [degrees] - galactic longitude, galactic latitude
    # =============================================================================
    lg = 3.14177
    bg = 0.52011
    bk = 1.68302

    b = np.arcsin(np.sin(bg) * np.sin(lat) + np.cos(bg) * np.cos(lat) * np.cos(long - lg))
    sinus = np.cos(lat) * np.sin(long - lg) / np.cos(b)
    kosinus = (np.cos(bg) * np.sin(lat) - np.sin(bg) * np.cos(lat) * np.cos(long - lg)) / np.cos(b)
    l = bk - np.arctan2(sinus, kosinus)

    return l, b


def ecl2gal_cart(x, y, z):
    # =============================================================================
    # converts ecliptic to galactic coordinates
    # Input:
    # long,lat [degrees] - ecliptic longitude, ecliptic latitude
    # Output:
    # l, b [degrees] - galactic longitude, galactic latitude
    # =============================================================================

    r = (x ** 2 + y ** 2 + z ** 2) ** (1 / 2)
    long = np.arctan2(y, x)
    lat = np.arctan(z / (x ** 2 + y ** 2) ** (1 / 2))

    lg = 3.14177
    bg = 0.52011
    bk = 1.68302

    b = np.arcsin(np.sin(bg) * np.sin(lat) + np.cos(bg) * np.cos(lat) * np.cos(long - lg))
    sinus = np.cos(lat) * np.sin(long - lg) / np.cos(b)
    kosinus = (np.cos(bg) * np.sin(lat) - np.sin(bg) * np.cos(lat) * np.cos(long - lg)) / np.cos(b)
    l = bk - np.arctan2(sinus, kosinus)

    return r * np.cos(l) * np.cos(b), r * np.sin(l) * np.cos(b), r * np.sin(b)


# galactic to ecliptic
def gal2ecl_spherical(l, b):
    # =============================================================================
    # converts ecliptic to galactic coordinates
    # Input:
    # l, b (radians) - galactic longitude, galactic latitude
    # Output:
    # long,lat (radians) - ecliptic longitude, ecliptic latitude
    # =============================================================================
    lg = 3.14177
    bg = 0.52011
    bk = 1.68302

    lat = np.arcsin(np.sin(bg) * np.sin(b) + np.cos(bg) * np.cos(b) * np.cos(bk - l))
    sinus = np.cos(b) * np.sin(bk - l) / np.cos(lat)
    kosinus = (np.cos(bg) * np.sin(b) - np.sin(bg) * np.cos(b) * np.cos(bk - l)) / np.cos(lat)
    long = lg + np.arctan2(sinus, kosinus)

    return long, lat


def gal2ecl_cart(x, y, z):
    # =============================================================================
    # converts  galactic coordinates
    # Input:
    # x,y,z - galactic
    # Output:
    # x,y,z - ecliptic
    # =============================================================================

    r = (x ** 2 + y ** 2 + z ** 2) ** (1 / 2)
    l = np.arctan2(y, x)
    b = np.arctan(z / (x ** 2 + y ** 2) ** (1 / 2))
    
    long, lat = gal2ecl_spherical(l, b)
    
    return r * np.cos(long) * np.cos(lat), r * np.sin(long) * np.cos(lat), r * np.sin(lat)


# galactic to equatorial
def gal2eq_spherical(l, b):
    long, lat = gal2ecl_spherical(l, b)
    return (ecl2eq_spherical(long, lat))


def gal2eq_cart(x, y, z):
    xe, ye, ze = gal2ecl_cart(x, y, z)
    return (ecl2eq_cart(xe, ye, ze))


# equatorial to galactic
def eq2gal_spherical(alpha, delta):
    long, lat = eq2ecl_spherical(alpha, delta)
    return (ecl2gal_spherical(long, lat))


def eq2gal_cart(x, y, z):
    xe, ye, ze = eq2ecl_cart(x, y, z)
    return (ecl2gal_cart(xe, ye, ze))


def ecc2true(E, e):
    # =============================================================================
    # converts eccentric (or hyperbolic) anomaly to true anomaly
    # Input:
    # E [radians] - eccentric (or hyperbolic anomaly)
    # Output:
    # True anomaly [radians]
    # =============================================================================
    if e > 1:
        return 2 * np.arctan(np.sqrt((e + 1) / (e - 1)) * np.tanh(E / 2))
    else:
        return np.arctan2(np.sqrt(1 - e ** 2) * np.sin(E), np.cos(E) - e)


def true2ecc(f, e):
    # =============================================================================
    # converts true anomaly to eccentric (or hyperbolic) anomaly
    # Input:
    # f [radians] - true anomaly
    # Output:
    # eccentric (or hyperbolic anomaly) [radians]
    # =============================================================================
    if e > 1:
        return 2 * np.arctanh(np.sqrt((e - 1) / (e + 1)) * np.tan(f / 2))

    else:
        return np.arctan2(np.sqrt(1 - e ** 2) * np.sin(f), e + np.cos(f))


def ecc2mean(E, e):
    # =============================================================================
    # converts eccentric anomaly to mean anomaly
    # Input:
    # E [radians] - eccentric anomaly
    # Output:
    # Mean anomaly [radians]
    # =============================================================================
    if e > 1:
        return e * np.sinh(E) - E
    else:
        return E - e * np.sin(E)


def mean2tp(M, a, epoch):
    # =============================================================================
    # converts mean anomaly to perihelion passage
    # Input:
    # M - mean anomaly (rad)
    # a - semi-major axis (au)
    # epoch  - epoch for M (MJD)
    # Output:
    # perihelion passage (MJD)
    # =============================================================================
    mean_motion = np.sqrt(mu / np.abs(a * au) ** 3)
    return epoch - (M / mean_motion) / 86400.


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


#                                   PLOT
# =============================================================================
def orbit_plot(o, O, i, e, a, rmax, plane, nodes, apse, color, ax):
    # =============================================================================
    # plots orbite in 3D given the orbital elements
    # plane=1 (plots orbital plane)
    # nodes=1 (plots line of nodes)
    # apse=1 (plots line of apsides)
    # rmax - maximum distance from the central body
    # color - color of the orbit, e.g. 'r' for red
    #
    # ax - projection which should be defined as
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # =============================================================================
    o = np.deg2rad(o)
    O = np.deg2rad(O)
    i = np.deg2rad(i)

    if e > 1:
        teta0 = np.arccos(a * (1 - e ** 2) / e / rmax - 1 / e)
        teta1 = np.linspace(-teta0, teta0, 100)

    else:
        teta1 = np.linspace(0, 2 * np.pi, 100)

    r = a * (1 - e ** 2) / (1 + e * np.cos(teta1))

    x = r * (np.cos(O) * np.cos(o + teta1) - np.sin(O) * np.cos(i) * np.sin(o + teta1))
    y = r * (np.sin(O) * np.cos(o + teta1) + np.cos(O) * np.cos(i) * np.sin(o + teta1))
    z = r * (np.sin(i) * np.sin(o + teta1))

    # plotting the orbit
    ax.plot(x, y, z, color, linewidth=2)

    # rotation matrix
    RO = np.transpose(np.array([[np.cos(O), np.sin(O), 0], [-np.sin(O), np.cos(O), 0], [0, 0, 1]]))  # rotacija za O
    Ri = np.transpose(np.array([[1, 0, 0], [0, np.cos(i), np.sin(i)], [0, -np.sin(i), np.cos(i)]]))  # rotacija za i
    Ro = np.transpose(np.array([[np.cos(o), np.sin(o), 0], [-np.sin(o), np.cos(o), 0], [0, 0, 1]]))  # rotacija za o
    R = RO.dot(Ri).dot(Ro)

    # plotting the orbital plane
    if plane == 1:

        duzina = min(np.abs(rmax / np.cos(O)), np.abs(rmax / np.sin(O)))
        duzina = max(np.abs(duzina * np.cos(o)), np.abs(duzina * np.sin(o)))
        xx, yy = np.meshgrid([-duzina, duzina], [-duzina, duzina])

        xp = np.zeros_like(xx)
        yp = np.zeros_like(xx)
        zp = np.zeros_like(xx)

        for i1 in range(0, len(xx)):
            for i2 in range(0, len(xx)):
                [xp[i1, i2], yp[i1, i2], zp[i1, i2]] = np.dot(R, np.transpose(np.array([xx[i1, i2], yy[i1, i2], 0])))

        ax.plot_surface(xp, yp, zp, color='r', linewidth=0, alpha=0.2)

    # plotting the line of apsides
    if apse == 1:
        [xa1, ya1, za1] = np.dot(R, np.transpose(np.array([rmax, 0, 0])))
        [xa2, ya2, za2] = np.dot(R, np.transpose(np.array([-rmax, 0, 0])))

        ax.plot([xa1, xa2], [ya1, ya2], [za1, za2], '--k')

    # plotting the line of nodes
    if nodes == 1:
        duzina = min(np.abs(rmax / np.cos(O)), np.abs(rmax / np.sin(O)))
        ax.plot([duzina * np.cos(O), -duzina * np.cos(O)], [duzina * np.sin(O), -duzina * np.sin(O)], [0, 0], 'k')

def year2sec(x):
    return x*31557600.0
# =============================================================================
#                                   OTHER
# =============================================================================
def moid(o1, O1, i1, e1, a1, o2, O2, i2, e2, a2, limit, r_max=50):
    # =============================================================================
    # Calculates minimum orbit intersection distance between elliptic and hyperbolic orbits
    # Input:
    # o1,O1,i1,e1,a1 [rad]- ellipse
    # o2,O2,i2,e2,a2 [rad]- hyperbola
    # limit - limiting step for division of orbit in order to find approximate solution of moid (rad)
    # r_max - maximum helicentric distance for ISO
    # Output:
    # moid [same as a1 and a2] - minimum orbit intersection dostance
    # E1_konacno, E2_konacno [radians] - eccentric (hyperbolic) corresponding to moid
    # =============================================================================
    
    try:
        korak = np.deg2rad(30)  # pocetni korak za E1 i E2
    
        # pretvaranje u radijane
        E1_min = 0
        E1_max = 2 * np.pi - korak
    
        E0 = np.arccosh(1 / e2 - r_max / e2 / a2)  # granicna anomalija kada je r=radijus
        E2_min = -E0
        E2_max = E0
    
        while korak > limit:
    
            E1 = np.linspace(E1_min, E1_max, int((E1_max - E1_min) / korak) + 1)  # Planet
            T1 = np.mod(ecc2true(E1, e1), 2 * np.pi)
    
            E2 = np.linspace(E2_min, E2_max, int((E2_max - E2_min) / korak) + 1)  # ISO
            T2 = np.mod(ecc2true(E2, e2), 2 * np.pi)
    
            r1 = a1 * (1 - e1 * np.cos(E1))
            r2 = a2 * (1 - e2 * np.cosh(E2))
    
            x1 = r1 * (np.cos(O1) * np.cos(o1 + T1) - np.sin(O1) * np.cos(i1) * np.sin(o1 + T1))
            y1 = r1 * (np.sin(O1) * np.cos(o1 + T1) + np.cos(O1) * np.cos(i1) * np.sin(o1 + T1))
            z1 = r1 * np.sin(o1 + T1) * np.sin(i1)
    
            x2 = r2 * (np.cos(O2) * np.cos(o2 + T2) - np.sin(O2) * np.cos(i2) * np.sin(o2 + T2))
            y2 = r2 * (np.sin(O2) * np.cos(o2 + T2) + np.cos(O2) * np.cos(i2) * np.sin(o2 + T2))
            z2 = r2 * np.sin(o2 + T2) * np.sin(i2)
    
            r = np.zeros([len(E1), len(E2)])
    
            for i in range(0, len(E1)):
    
                for j in range(0, len(E2)):
                    r[i, j] = np.sqrt((x1[i] - x2[j]) ** 2 + (y1[i] - y2[j]) ** 2 + (z1[i] - z2[j]) ** 2)
    
            ind1, ind2 = np.argwhere(r == np.min(r))[0]
    
            # priblizna resenja
            E1_konacno = E1[ind1]
            E2_konacno = E2[ind2]
    
            E1_min = E1_konacno - korak
            E1_max = E1_konacno + korak
    
            E2_min = E2_konacno - korak
            E2_max = E2_konacno + korak
    
            korak = korak / 2
    
        MOID = np.min(r)
    
    except:
        
        MOID=0
        E1_konacno=0
        E2_konacno=0

    return MOID


def mean_distance(b, n):
    # =============================================================================
    # Estimates mean minimum apparent distance among objects in number of latitudinal belts,
    # assuming uniform distribution in longitude
    # Input:
    # b [degrees]- ecliptic latitudes of objects
    # n - number of latitudinal belts to divide the sphere
    # Output:
    # mean minimum apparent distance among objects [degrees]
    # =============================================================================
    LAT = np.linspace(90 / n, 90, n)
    korak = 90 / n

    b = b[np.argwhere(b > 0)]  # taking only one hemisphere (assuming symetrical situation)

    br = np.array([])  # number of object with b les then some value (boundary of belt)
    for lat in LAT:
        br = np.append(br, len(np.argwhere(b < lat)))

    broj_pojasevi = np.array(br[0])
    for i in range(1, len(br)):
        broj_pojasevi = np.append(broj_pojasevi, br[i] - br[i - 1])

    # Area of belt

    S = np.array(2 * np.pi * np.sin(np.deg2rad(korak)))  # area of th first belt

    for i in range(1, len(LAT)):
        S = np.append(S, 2 * np.pi * (np.sin(np.deg2rad(LAT[i])) - np.sin(np.deg2rad(LAT[i - 1]))))

    S = S * 32400 / np.pi ** 2  # coverts to squared degrees

    # mean minimum distance
    return np.sqrt(S / broj_pojasevi) / 2

def imitate_sample(X, n_bins, N):
    '''
    Takes arbitrary array X and generates sample of N elements that imitates the 
    distribution of X uses inverse transform sampling method and spline interpoalation. 
    This can be used to generate sample of arbitrary size of, say, albedos that 
    imitates some known sample of albedos, e.g. from WISE.
    
    input:
        X - array whose distribution is imitated
        nbins - number of bins for array X to make discrete distribution
        N - number of elements in the output sample
        
    output:
        sample - sample of N elements with the distribution similar to X 
    '''

    # binning data
    y, xx= np.histogram(X, bins=n_bins)
    x = np.zeros(len(y))
    
    for i in range(len(x)):
        x[i]=(xx[i+1]+xx[i])/2
    
 
    # natural spline interpolation
    a=y[:-1]
    b=np.zeros(len(a), dtype='float')
    d=np.zeros(len(a), dtype='float')
    h=np.zeros(len(x)-1, dtype='float')
    for i in range(0,len(x)-1):
        h[i]=x[i+1]-x[i]
        
    A=np.zeros([len(x), len(x)], dtype='float')
    v=np.zeros(len(x))
    
    for i in range(2,len(A)):
    
        A[i-1,i-1]=2*(h[i-2]+h[i-1])
        A[i,i-1]=h[i-1]
        A[i-2,i-1]=h[i-2]
        
        v[i-1]=3*((y[i]-y[i-1])/h[i-1]-(y[i-1]-y[i-2])/h[i-2])
    
    A[0,0]=1; A[-1][-1]=1; A[-1,-2]=0; A[0,1]=0; A[1,0]=h[0]; A[-2,-1]=h[-1] 
                  
    c = np.linalg.solve(A,v)
    
    for i in range(len(a)):
        b[i]=(y[i+1]-y[i])/h[i]-h[i]/3*(2*c[i]+c[i+1])
        d[i]=(c[i+1]-c[i])/3/h[i]
        
    c=c[:-1]
    Y0=np.zeros(len(x)) # integral in nodes
    
    for i in range(1,len(x)):
        Y0[i]=np.polyval([d[i-1]/4,c[i-1]/3,b[i-1]/2,a[i-1],Y0[i-1]],x[i]-x[i-1]) # integral in nodes
    
    # Generating sample using Inverse Transform Sampling
    r=np.random.random(N)*np.max(Y0) # uniform sample
    sample=np.zeros_like(r) # output sample
    
    for i in range(len(r)):
        ind=np.argwhere(Y0<=r[i])[-1]
        if ind==len(Y0):
            ind=ind-1
        
        t=np.roots((np.array([d[ind]/4,c[ind]/3,b[ind]/2,a[ind],Y0[ind]-r[i]])).flatten()) 
        t=t[(np.argwhere(np.imag(t)==0)).flatten()] 
        t=t[(np.argwhere(t.real>0)).flatten()]
        t=t[(np.argwhere(t.real<x[ind+1]-x[ind])).flatten()]
        t=t+x[ind]
        sample[i]=np.real(t)    
    
    return sample