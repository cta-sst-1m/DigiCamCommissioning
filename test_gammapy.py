from gammapy.spectrum import diffuse_gamma_ray_flux
import numpy as np
import matplotlib.pyplot as plt

E = np.logspace(-2, 2, 100)

plt.figure()
plt.loglog(E, diffuse_gamma_ray_flux(E))
plt.xlabel('')
plt.show()

