# BGPV

# Plot BG for a range of uniform BG flows

#==================================================

import numpy as np
import matplotlib.pyplot as plt

from diagnostics import diff

#==================================================

N = 513
nn = 81

L = 3840000.

U0 = np.linspace(-0.1,0.1,nn)
dU = U0[1] - U0[0]

y = np.linspace(-L/2,L/2,N)
dy = y[1] - y[0]

U0g, yg = np.mgrid[slice(-0.1,0.1+dU,dU),slice(-L/2,L/2+dy,dy)]

g = 9.81
f0 = 0.83e-4
beta = 2.0e-11
f = f0 + beta * y

Hflat = 4000.

H0 = np.zeros((N,nn))
for ui in range(0,nn):
	for j in range(0,N):
		H0[j,ui] = - (U0[ui] / g) * (f0 * y[j] + beta * y[j]**2 / 2) + Hflat

# Now PV
Q = np.zeros((nn,N))
for ui in range(0,nn):
	for j in range(0,N):
		Q[ui,j] = f[j] / H0[j,ui]

# PV gradient
Qy = np.zeros((nn,N))
Qyy = np.zeros((nn,N))
for ui in range(0,nn):
	Qy[ui,:] = diff(Q[ui,:],2,0,dy)
	Qyy[ui,:] = diff(Qy[ui,:],2,0,dy)

print(np.shape(Qy))

#plt.plot(Qy[:,500],label='north')
#plt.plot(Qy[:,300],label='center')
#plt.plot(Qy[:,10],label='south')
#plt.legend()
#plt.show()

plt.plot(Qy[31,:],label=str(U0[31]))
plt.plot(Qy[50,:],label=str(U0[50]))
plt.plot(Qy[65,:],label=str(U0[65]))
plt.legend()
plt.show()

plt.pcolormesh(np.transpose(U0g),np.transpose(yg),np.transpose(Qy))
plt.colorbar()
plt.grid()
plt.xlim([U0.min(),U0.max()])
plt.ylim([y.min(),y.max()])
plt.show()


