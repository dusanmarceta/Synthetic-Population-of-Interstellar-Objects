import numpy as np
import matplotlib.pyplot as plt

q, e, inc, node, argument, f_init, time = np.loadtxt("iso_exit_times.txt", skiprows=1, unpack=True)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].hist(time, bins=30)
axs[0, 0].set_title("Exit time distribution")
axs[0, 0].set_xlabel("Years")

axs[0, 1].scatter(e, time, s=2)
axs[0, 1].set_title("Eccentricity vs Time")
axs[0, 1].set_xlabel("e")

axs[1, 0].scatter(q, time, s=2)
axs[1, 0].set_title("Perihelion distance vs Time")
axs[1, 0].set_xlabel("q [AU]")

axs[1, 1].scatter(np.degrees(inc), time, s=2)
axs[1, 1].set_title("Inclination vs Time")
axs[1, 1].set_xlabel("i [deg]")

plt.tight_layout()
plt.show()