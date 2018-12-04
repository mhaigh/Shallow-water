# testSovler.py
# A test of the solver for a problem with a known solution
#=================================================================

from diagnostics import diff
import numpy as np
import matplotlib.pyplot as plt

#==================================================================

######
#
# d^2 u / dy^2 - y * u = F 
#
#####

BC = 'NO-SLIP';

I = np.complex(0,1);

N = 2**10;
L = 3000 * 1000.;

y = np.linspace(-L/2,L/2,N);

dy = y[1] - y[0];

# The forcing
F = np.zeros(N);
a = 90 * 1000;
for j in range(0,N):
	if abs(y[j]) < a:
		F[j] = (1 + np.cos(np.pi * y[j] / (1 * a)));
plt.plot(F);
plt.show();


# The solver
d = 1. / (dy**2);

u = np.zeros(N);

if BC == 'NO-SLIP':
	A = np.zeros((N-2,N-2));
	A[0,0] = - y[1] - 2 * d;
	A[0,1] = d;
	A[N-3,N-3] = - y[N-2] - 2 * d;
	A[N-3,N-4] = d;
	for j in range(1,N-3):
		A[j,j] = - y[j+1] - 2 * d;
		A[j,j-1] = d;
		A[j,j+1] = d;	
	u[1:N-1] = np.linalg.solve(A,F[1:N-1]);
	
if BC == 'FREE-SLIP':
	A = np.zeros((N,N),dtype=complex);
	A[0,0] = - y[1] - 2 * d;
	A[0,1] = d;
	A[N-3,N-3] = - y[N-2] - 2 * d;
	A[N-3,N-4] = d;
	for j in range(1,N-3):
		A[j,j] = - y[j+1] - 2 * d;
		A[j,j-1] = d;
		A[j,j+1] = d;	
	u[1:N-1] = np.linalg.solve(A,F[1:N-1]);

# Useful vectors
u_y = diff(u,2,0,dy);
u_yy = diff(u_y,2,0,dy);
uy = - u * y;

# Error
error = u_yy + uy - F;
plt.plot(error);
plt.show();
error = np.sqrt((error**2).mean());
print(error);

plt.plot(u);
plt.show();

plt.subplot(221);
plt.plot(u_yy);
plt.subplot(222);
plt.plot(uy);
plt.subplot(223);
plt.plot(u_yy + uy);
plt.subplot(224);
plt.plot(F);
plt.show();

# Calculate the error




	
	
