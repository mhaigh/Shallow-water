# ANALYSE_MODES.py

import numpy as np
import matplotlib.pyplot as plt

from inputFile import *
from core.diagnostics import diff

#=======================================================

path = "eig/"
m1 = np.load(path+'mode1.npy')
m2 = np.load(path+'mode2.npy')
m3 = np.load(path+'mode3.npy')
m4 = np.load(path+'mode4.npy')

theta = np.load(path+'theta0123.npy')

m1 = np.real(theta[0]*m1)
m2 = np.real(theta[1]*m2)
m3 = np.real(theta[2]*m3)
m4 = np.real(theta[3]*m4)

m = m1 + m2 + m3 + m4

plt.plot(m); plt.show()

u = m[0:N]
v = np.zeros(N)
v[1:N-1] = m[N:2*N-2]
h = m[2*N-2:3*N-2]

vy = diff(v,2,0,dy)
uv = u * vy
P = diff(uv,2,0,dy)


plt.plot(P)
plt.show()


m12 = m1 * m2
m13 = m1 * m3
m23 = m2 * m3

l2 = 54

