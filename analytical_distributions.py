import numpy as np

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