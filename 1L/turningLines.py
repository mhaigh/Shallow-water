# turningLines.py

#=================================================================

import numpy as np
import matplotlib.pyplot as plt

from inputFile import *

#=================================================================

# Take U0 to be Gaussian, first find its second derivative.
Uy = diff(U0,2,0,dy)
Uyy = diff(Uy,2,0,dy)

# Pick a zonal wavenumber and define its zonal phase speed
K = 2*np.pi * K_nd / L
omega = 2*np.pi*omega

c = omega / K

Qy = beta - Uyy

QQ = U * chi / Ly
#n2 = (beta - Uyy) / (U - c) - K**2

cmin = np.zeros((N,N))
ceff = np.zeros((N,N))
for i in range(1,N):
	for j in range(0,N):
		cmin[j,i] = U0[j] - (Qy[j]) / K[i]**2
		ceff[j,i] = c[i] - cmin[j,i]

cmin = np.fft.fftshift(cmin,axes=1)
K_nd = np.fft.fftshift(K_nd)

plt.plot(y_nd,Qy); plt.grid(); plt.show()

v = np.linspace(-.1, 2.0, 15, endpoint=True)

ma = 1.e-2
mi = 0.

cma = 1.
cmi = -1.

ceff = np.where(ceff>ma,ma,ceff)
ceff = np.where(ceff<mi,mi,ceff)

cmin = np.where(cmin>cma,cma,cmin)
cmin = np.where(cmin<cmi,cmi,cmin)
#plt.plot(y/1000.,Qy); plt.grid(); plt.show()

plt.subplot(121)
plt.contourf(K_nd,y_nd,cmin); plt.colorbar(); plt.grid();
plt.subplot(122)
plt.contourf(K_nd,y_nd,ceff); plt.colorbar(); plt.grid(); plt.show()


plt.contourf(cmin); plt.colorbar(); plt.grid(); plt.show()
