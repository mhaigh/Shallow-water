# RossbyWave.py

import numpy as np
import matplotlib.pyplot as plt

I = np.complex(0,1)

N = 256
Nt = 300

L = 3840000.

A = 3. + 2 * I
B = 2 - 3 * I

x = np.linspace(0,L,N)
y = np.linspace(0,L,N)

beta = 2.0e-11

U = -0.08
k1 = -4
l1 = 6

k2 = -5
l2 = -7

k1 = k1 / L
l1 = l1 / L
k2 = k2 / L
l2 = l2 / L

K1 = k1**2 + l1**2
om1 = U * k1 - beta * k1 / K1

K2 = k2**2 + l2**2
om2 = U * k2 - beta * k2 / K2

T1 = 1./ om1
T2 = 1. / om2
print(T1)
t = np.linspace(0,2*T1,Nt)

psi = np.zeros((N,N,Nt))
for ti in range(0,Nt):
	for i in range(0,N):
		psi[:,i,ti] = np.real(A * np.exp(2 * np.pi * I * (k1 * x[i] + l1 * y[:] - om1 * t[ti])) + B * np.exp(2 * np.pi * I * (k2 * x[i] + l2 * y[:] - om2 * t[ti])))

np.save('psi',psi)

plt.contourf(psi[:,:,10]); 
plt.colorbar(); plt.show()
