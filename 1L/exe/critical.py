
import numpy as np
import matplotlib.pyplot as plt

from inputFile import *

kd = 1. / Ld


Ns = 11

Uy =  diff(U0,2,0,dy)
Uyy = diff(Uy,2,0,dy)

sigma_set = np.linspace(0.015,0.045,Ns) * 3840000.0
ds = sigma_set[1] - sigma_set[0]

umag_set = np.linspace(0.4,1.2,Ns) 
du = umag_set[1] - umag_set[0]

lim = 10.
cmin = np.zeros((N,N,Ns))
for ui in range(0,Ns):

	sigma = sigma_set[ui]
	#Umag = umag_set[ui]
	U0, H0 = BG_state.BG_Gaussian(Umag,sigma,JET_POS,Hflat,f0,beta,g,y,L,N)
	Uy =  diff(U0,2,0,dy); Uyy = diff(Uy,2,0,dy)

	for j in range(0,N):
		for i in range(1,N-1):
			cmin[j,i,ui] = U0[j] - (beta - Uyy[j] + kd[i]**2 * U0[j]) / (K[i]**2 + kd[j]**2)
			if cmin[j,i,ui] > lim:
				cmin[j,i,ui] = lim
			elif cmin[j,i,ui] < -lim:
				cmin[j,i,ui] = -lim

cmin = np.fft.fftshift(cmin,axes=1)

plt.contour(cmin[:,N//2+3,:],1,colors='k')
plt.show()


plt.contourf(np.fft.fftshift(K)*L,y/L,cmin)
plt.colorbar()
plt.grid()
plt.show()

plt.plot(y_nd,cmin[:,3])
plt.plot(y_nd,cmin[:,N//2+5])
plt.show()
