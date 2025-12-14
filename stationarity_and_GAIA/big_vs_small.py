import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import GM_sun, au

q_big, e_big, f_big, inc_big, node_big, argument_big = np.loadtxt('big_metallicity.txt', unpack=True)
q_small, e_small, f_small, inc_small, node_small, argument_small = np.loadtxt('small_metallicity.txt', unpack=True)

plt.figure()
plt.hist(q_big, np.linspace(0, 5, 100), alpha = 0.5, label = 'high metallicity')
plt.hist(q_small, np.linspace(0, 5, 100), alpha = 0.5, label = 'low metallicity')

a_big = q_big / (1- e_big)

v_big = np.sqrt(-GM_sun.value / (a_big * au.value))


plt.figure()
plt.hist(inc_big, np.linspace(0, np.pi, 100), alpha = 0.5, label = 'high metallicity')
plt.hist(inc_small, np.linspace(0, np.pi, 100), alpha = 0.5, label = 'low metallicity')

a_small = q_small / (1- e_small)

v_small = np.sqrt(-GM_sun.value / (a_small * au.value))


plt.figure()
plt.hist(v_big/1000, np.linspace(0, 200, 100), alpha=0.5, label='[M/H] > 0', color='red', density=True)
plt.hist(v_small/1000, np.linspace(0, 200, 100), alpha=0.5, label='[M/H] < -1', color='blue', density=True)

plt.xlabel("Interstellar velocity (km/s)", fontsize=36)
plt.ylabel("pdf", fontsize=36)
plt.legend(fontsize=30)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)



plt.figure()
plt.hist(e_big, np.linspace(1, 20, 20), alpha=0.5, label='[M/H] > 0', color='red', density=True)
plt.hist(e_small, np.linspace(1, 20, 20), alpha=0.5, label='[M/H] < -1', color='blue', density=True)

plt.xlabel("Interstellar velocity (km/s)", fontsize=36)
plt.ylabel("pdf", fontsize=36)
plt.legend(fontsize=30)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)



plt.figure()
plt.hist(np.rad2deg(node_big), np.linspace(0, 360, 36), alpha=0.5, label='[M/H] > 0', color='red', density=True)
plt.hist(np.rad2deg(node_small), np.linspace(0, 360, 36), alpha=0.5, label='[M/H] < -1', color='blue', density=True)

plt.xlabel("Longitude of ascending node (deg)", fontsize=36)
plt.xticks(np.arange(0, 420, 60))
plt.ylabel("pdf", fontsize=36)
plt.legend(fontsize=30)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)



plt.figure()
plt.hist(np.rad2deg(argument_big), np.linspace(0, 360, 36), alpha=0.5, label='[M/H] > 0', color='red', density=True)
plt.hist(np.rad2deg(argument_small), np.linspace(0, 360, 36), alpha=0.5, label='[M/H] < -1', color='blue', density=True)

plt.xlabel("Interstellar velocity (km/s)", fontsize=36)
plt.ylabel("pdf", fontsize=36)
plt.legend(fontsize=30)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

