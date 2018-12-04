
import numpy as np
import matplotli.pyplot as plt

from inputFile import *

kd = 1. / Ld

Uy =  diff(U0,2,0,dy)
Uyy = diff(Uy,2,0,dy)

cmin = np.zeros((N,N))
for j in range(0,N):
	for i in range(0,N):
		cmin[j,i] = U0[j] + (beta - Uyy[j] + kd**2 * U[j]) / (K[i]**2 + kd[j]**2)

plt.contourf(cmin)
plt.colorbar()
plt.show()


