# PVforcing

#=======================================================================

import numpy as np
import matplotlib.pyplot as plt

import forcing
import BG_state
from diagnostics import diff

from inputFile import *

#=======================================================================

nn = 81

Uset = np.linspace(-.3,.5,nn)

FPV = np.zeros((N,N))
fpv = np.zeros(nn)

Q = np.zeros((N,nn))
HH = np.zeros((nn))

for ui in range(0,nn):
	
	U0, H0 = BG_state.BG_uniform(Uset[ui],Hflat,f0,beta,g,y,N)

	U0 = U0 / U; H0 = H0 / chi

	U0y = diff(U0,2,0,dy_nd)
	Q[:,ui] = (f_nd / Ro - U0y) / H0
	
	HH[ui] = np.mean(H0) - Hflat

	F2x = diff(F2_nd,1,1,dx_nd)
	F1y = diff(F1_nd,0,0,dy_nd)

	for j in range(0,N):
		for i in range(0,N):
			FPV[j,i] = (F2x[j,i] - F1y[j,i]) / H0[j]# + Q[j] * F3_nd[j,i] / H0[j]

	fpv[ui] = np.sum(FPV)

plt.plot(1./HH)
plt.show()
plt.plot(Uset,fpv)
plt.show()

	
	#plt.contourf(FPV)
	#plt.colorbar()
	#plt.show()

	
