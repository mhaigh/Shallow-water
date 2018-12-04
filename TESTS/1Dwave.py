# 1Dwave
#=============================================================

# 1-D wave equation plunger test

#=============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse.linalg import eigsh
import sys
#=============================================================

#=============================================================
#=============================================================

# diff
def diff(f):
	
	N = len(f);
	df = np.zeros(N);

	for i in range(0,N-1):
		df[i] = f[i+1] - f[i];
	df[N-1] = f[0] - f[N-1];	

	return df

# diff_center
def diff_center(f):
	
	N = len(f);
	df = np.zeros(N);

	for i in range(1,N-1):
		df[i] = f[i+1] - f[i-1];
	df[0] = f[1] - f[N-1];
	df[N-1] = f[0] - f[N-2];	

	return df

#=============================================================
#=============================================================

I = np.complex(0.0,1.0);

N = 256+1;

g = 9.81;
H = 4000.0;
# Wave speed
c = np.sqrt(g*H);

# Space parameters
L = 3840.0 * 1000.0
x = np.linspace(-L/2,L/2,N+1);
x0 = 1. * 90.0 * 1000.0
K = np.fft.fftshift(np.fft.fftfreq(N,L/N)) * L;	# K is non-dimensional.
dx = x[1] - x[0];

# Time parameters
kk = 4.
period_days = 1.0e-2;
period = 1.*(L/(kk*c));# * 24. * period_days;		# Periodicity of plunger (s)
omega = 1. / (period);          		# Frequency of plunger, once every 50 days (e-6) (s-1)
Nt = 200;								# Number of time samples
T = np.linspace(0,kk*period,Nt+1);			# Array of time samples across one forcing period (s)
dt = T[1] - T[0];						# Size of the timestep (s)
ts = 0; 								# index at which the time-snapshot is taken
t = T[ts];
	
print('Dist = ' + str(c*period/L));

time_dep = np.cos(2. * np.pi * omega * T);
# Plunger
A = 1.0e-7
F_dcts = np.zeros(N);
F_cts = np.zeros(N);
for i in range(0,N):
	if abs(x[i]) < x0:
		F_dcts[i] = A * np.cos((np.pi / 2) * x[i] / x0);
		F_cts[i] = 0.5 * A * (1 + np.cos(np.pi * x[i] / x0));

F_wave = 1. * np.sin(2 * np.pi * 2.0 * x[0:N] / L);
F_delta = np.zeros(N);
F_delta[int(N/2)]=1.

F = F_cts;

#plt.plot(F);
#plt.show();

# Solution via finite differences
A = np.zeros((N,N));
a = - 4 * np.pi**2 * omega**2;	b = - c**2 / dx**2;
A[0,0] = a - 2.0 * b; A[0,1] = b; A[0,N-1] = b;	
A[N-1,N-1] = a - 2.0 * b; A[N-1,N-2] = b; A[N-1,0] = b;
for i in range(1,N-1):
	A[i,i] = a - 2.0 * b;
	A[i,i+1] = b;
	A[i,i-1] = b;
u = np.linalg.solve(A,F);

# Eigenmodes and frequencies
vec = np.zeros((N,N),dtype=complex);	# N eigenmodes (first index), each N long (second index).
val = np.zeros(N,dtype=float);			# N eigenvalues/frequencies. omega = c * k		
for i in range(0,N):
	vec[i,:] = np.exp(2.0 * np.pi * I * K[i] * x[0:N] / L);
	val[i] = c * K[i] / L;
	#print(1./val[i]);
val_period_secs = 1. / val;
val_period_days = 1. / (val * 24.0 * 3600.0);

# Decompose solution snapshot into eigenmodes
theta = np.linalg.solve(vec,u);
theta_abs = np.abs(theta);
dom_index = np.argsort(-theta_abs);	

# Create a projection of Nm most dominant modes
Nm = 8;
proj = theta[dom_index[0]] * vec[dom_index[0],:];
for mi in range(1,4):
	di = dom_index[mi];
	proj = proj + theta[di] * vec[di,:];
print('Forcing period = ' + str(period));
#print(val_period_secs[dom_index[0:Nm]]);
##print(val_period_days[dom_index[0:Nm]]);

# Now plot
u = np.real(u);
proj = np.real(proj);
plt.figure(1);
plt.plot(u,label='sol');	
plt.plot(proj,label='PROJ');
plt.legend();
plt.show();

#=============================================

# Project forcing onto eigenmodes at all times
theta_F = np.zeros((N,Nt),dtype=complex);
theta_u = np.zeros((N,Nt),dtype=complex);
vec_time = np.zeros((N,N,Nt),dtype=complex);	# (vector, x, time)
for wi in range(0,N):
	for ti in range(0,Nt):
		vec_time[wi,:,ti] = vec[wi,:]*np.exp(-2*np.pi*I*val[wi]*T[ti]);

for ti in range(0,Nt):
	F_time = F * np.exp(-2*np.pi*I*omega*T[ti]);
	u_time = u * np.exp(-2*np.pi*I*omega*T[ti]);
	theta_F[:,ti] = np.linalg.solve(vec_time[:,:,ti],F_time);
	theta_u[:,ti] = np.linalg.solve(vec_time[:,:,ti],u_time);

theta_F_norm = np.max(abs(theta_F));
theta_u_norm = np.max(abs(theta_u));

theta_F_av = np.zeros(N);
theta_u_av = np.zeros(N);
for wi in range(0,N):
	theta_F_av[wi] = np.sum(theta_F[wi,:]);
	theta_F_av[wi] = theta_F_av[wi] + theta_F[wi,0];
	theta_u_av[wi] = np.sum(theta_u[wi,:]);
	theta_u_av[wi] = theta_u_av[wi] + theta_u[wi,0]
theta_F_av = theta_F_av / (Nt+1);
theta_u_av = theta_u_av / (Nt+1);

F_proj2 = theta_F[0,0] * vec_time[0,:,0];
for ki in range(1,N):
	F_proj2[:] = F_proj2[:] + theta_F[ki,0] * vec_time[ki,:,0];

dom_index = np.argsort(-theta_F_av);
print(dom_index);
	
tf_av = abs(theta_F_av) / np.max(abs(theta_F_av));
tu_av = abs(theta_u_av) / np.max(abs(theta_u_av));

# Average wavelength
#=============================================

k_av_F = np.zeros(Nt);
k_av_u = np.zeros(Nt);
for ti in range(0,Nt):
	for wi in range(0,N):	
		k_av_F[ti] = k_av_F[ti] + abs(K[wi]) * abs(theta_F[wi,ti]);
		k_av_u[ti] = k_av_u[ti] + abs(K[wi]) * abs(theta_u[wi,ti]);
	k_av_F[ti] = k_av_F[ti] / np.sum(abs(theta_F[:,ti]));
	k_av_u[ti] = k_av_u[ti] / np.sum(abs(theta_u[:,ti]));

plt.plot(k_av_F,label='forcing');
plt.plot(k_av_u,label='sol');
plt.title('Average wavenumber vs time')
plt.xlabel('time')
plt.ylabel('Average wavenumber')
plt.legend();
plt.show();

print(sum(k_av_F)/Nt);
print(sum(k_av_u)/Nt);

# Plots
#=============================================

# Plot of average over time.
plt.plot(K,tf_av,label='Forcing');
plt.plot(K,tu_av,label='Solution');
plt.title('Decomposition of forcing/solution',fontsize=22);
plt.xlabel('wavenumber',fontsize=22);
plt.ylabel('absoltute average weight',fontsize=22);
plt.legend();
plt.show();

# Plot of weights in space and zonal wavenumber
tf = np.abs(theta_F);
tu = np.abs(theta_u);
plt.subplot(121);
plt.contourf(T[0:Nt],K,tf/theta_F_norm);
plt.colorbar();
plt.xlabel('time');
plt.ylabel('zonal wavenumber')
plt.title('forcing');
plt.subplot(122);
plt.contourf(T[0:Nt],K,tu/theta_u_norm);
plt.colorbar();
plt.xlabel('time');
plt.title('solution');
plt.show();


for ti in range(0,0):
	theta_u_max = np.max(abs(theta_u[:,ti]));
	theta_F_max = np.max(abs(theta_F[:,ti]));
	plt.plot(K,theta_u[:,ti]/theta_u_max,label='Sol');
	plt.plot(K,theta_F[:,ti]/theta_F_max,label='Force');
	plt.legend();
	plt.title('Time = ' + str(ti));
	plt.show();


# Error of the solution
u_tt = - 4 * np.pi**2 * omega**2 * u;
u_xx = diff(diff(u)) / dx**2
e = u_tt - c**2 * u_xx - F_cts; 

# Plot error
if False:
	plt.subplot(221);
	plt.plot(u_tt);
	plt.subplot(222);
	plt.plot(c**2*u_xx);
	plt.subplot(223);
	plt.plot(F_cts*time_dep[ts]);
	plt.subplot(224);
	plt.plot(e);
	plt.show();


