# testSovler.py
# A test of the solver for a problem with a known solution
#==================================================================

import numpy as np
import matplotlib.pyplot as plt

#==================================================================

######
#
# d^2 u / dy^2 - u = F 
#
#####
I = np.complex(0,1);

N = 256;
L = 3000 * 1000.;

x = np.linspace(-L/2,L/2,N+1);
y = np.linspace(-L/2,L/2,N);

dx = x[1] - x[0];
dy = y[1] - y[0];

K = np.fft.fftfreq(N,dx);

# The exact solution
v = np.zeros((N,N));
for i in range(0,N):
	for j in range(0,N):
		v[j,i] = - L**3 * np.sin(2*np.pi*x[i]/L) * np.sin(2*np.pi*y[j]/L) / (2 * np.pi * (L**2 + 4 * np.pi**2)) / 60000.;

# The forcing
F = np.zeros((N,N+1));
for i in range(0,N+1):
	for j in range(0,N):
		F[j,i] = np.cos(2*np.pi*x[i]/L) * np.sin(2*np.pi*y[j]/L) / 60000.;
Ftilde = np.fft.hfft(F,N,axis=1);

# The solver
a = 2. * np.pi * I * K;
b = 1. / (dy**2);

A = np.zeros((N,N),dtype=complex);
utilde = np.zeros((N,N),dtype=complex);

# Periodic BCs
A[0,0] = 1 + b;
A[0,1] = - 2 * b;
A[0,2] = b;

A[N-1,N-1] = 1 + b;
A[N-1,N-2] = - 2 * b;
A[N-1,N-3] = b;
	
for j in range(1,N-1):
	A[j,j] = - 1 - 2 * b;
	A[j,j-1] = b;
	A[j,j+1] = b;
	
A = A * a[1];
utilde[:,1] = np.linalg.solve(A,Ftilde[:,1]);

A = A * a[N-1] / a[1];
utilde[:,N-1] = np.linalg.solve(A,Ftilde[:,N-1]);
	
u = np.fft.ifft(utilde,axis=1);

# Calculate the error

# Plots

plt.figure(1);
plt.subplot(121);
plt.contourf(u);
plt.colorbar();
plt.subplot(122);
plt.contourf(v);
plt.colorbar();
plt.show();
	
	
