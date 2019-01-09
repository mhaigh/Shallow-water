# ANALYSE_MODES.py

import numpy as np
import matplotlib.pyplot as plt

from core import solver, diagnostics

from inputFile import *

#=======================================================

I = np.complex(0.,1.)

path = "eig/"
phi1 = np.load(path+'mode1.npy')
phi2 = np.load(path+'mode2.npy')
phi3 = np.load(path+'mode3.npy')
phi4 = np.load(path+'mode4.npy')

theta = np.load(path+'theta0123.npy')

phi1 = theta[0]*phi1
phi2 = theta[1]*phi2
phi3 = theta[2]*phi3
phi4 = theta[3]*phi4

k = 4
cos = np.cos(2*np.pi*k*x_nd)
sin = np.sin(2*np.pi*k*x_nd)

dim = len(phi1)

phi = phi1 + phi2 + phi3 + phi4
solution = np.zeros((dim,N),dtype=complex)
solution[:,4] = phi

utilde, vtilde, etatilde = solver.extractSols(solution,N,N2,BC)
u, v, h = solver.SPEC_TO_PHYS(utilde,vtilde,etatilde,T_nd,dx_nd,omega_nd,N)

u = np.real(u); v = np.real(v); h = np.real(h)

uv = u*v

plt.contourf(uv[:,:,ts]); plt.show()

print(np.shape(uv))

uv = diagnostics.timeAverage(uv,T_nd,Nt)

print(np.shape(uv))

plt.contourf(uv); plt.show()

uvy = diagnostics.diff(uv,0,0,dy)

plt.contourf(uvy); plt.show()

quit()

uvy = diagnostics.extend(uvy)

uvy_xav = np.trapz(uvy,x_nd,dx_nd,axis=1);


vy = diff(v,2,0,dy)
uv = u * vy
P = diff(uv,2,0,dy)


plt.plot(P)
plt.show()


m12 = m1 * m2
m13 = m1 * m3
m23 = m2 * m3

l2 = 54

