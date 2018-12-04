# energySpectral
#=========================================================

# Code for calculation spectral kinetic energy for some solution (u,v,eta), found by RSW... code.
# When running this code, the variables in the input file must match those that are originally used to calculate (u,v,h),
# which are pre-solved and loaded below.
#=========================================================

import numpy as np
import matplotlib.pyplot as plt

from inputFile_1L import *
from diagnostics import extend

#=========================================================

u_nd = np.load('/home/mike/Documents/GulfStream/Code/DATA/1L/' + str(FORCE) + '/' + str(BG) + '/u_nd_' + str(Fpos) + str(N) + '.npy');
v_nd = np.load('/home/mike/Documents/GulfStream/Code/DATA/1L/' + str(FORCE) + '/' + str(BG) + '/v_nd_' + str(Fpos) + str(N) + '.npy');
eta_nd = np.load('/home/mike/Documents/GulfStream/Code/DATA/1L/' + str(FORCE) + '/' + str(BG) + '/eta_nd_' + str(Fpos) + str(N) + '.npy');

# Calculate full zonal velocity and SSH
etaFull = np.zeros((N,N,Nt));
uFull = np.zeros((N,N,Nt));
for j in range(0,N):
	for ti in range(0,Nt):
		etaFull[j,:,ti] = eta_nd[j,:,ti] + H0_nd[j];
		uFull[j,:,ti] = u_nd[j,:,ti] + U0_nd[j];

# We want the time average of the energy
#eta_av = etaFull[:,:,0];
#u_av = uFull[:,:,0];
#v_av = v_nd[:,:,0];
#for ti in range(1,Nt):
#	eta_av[:,:,ti] = eta_av[:,:,ti-1] + etaFull[:,:,ti];
#	u_av[:,:,ti] = u_av[:,:,ti-1] + u[:,:,ti];
#	v_av[:,:,ti] = v_av[:,:,ti-1] + v[:,:,ti];
#eta_av = eta_av / nt;
#u_av = u_av / nt;

U0tilde = np.real(np.fft.fft(U0_nd));
H0tilde = np.real(np.fft.fft(H0_nd));

utilde = np.real(np.fft.fftshift(np.fft.fft2(uFull)));
vtilde = np.real(np.fft.fftshift(np.fft.fft2(v_nd)));
etatilde = np.real(np.fft.fftshift(np.fft.fft2(etaFull)));

# Define sets of wavenumbers
Kx = K_nd;
Ky = np.fft.fftfreq(N,Ly/N);
Ky = Ky * Ly;
Kabs = np.absolute(Kx);

# Kinetic energy
KE = 0.5 * etatilde * (utilde**2 + vtilde**2);
KE_BG = 0.5 * H0tilde * U0tilde**2;

KEprime = np.zeros((N,N,Nt));
for j in range(0,N):
	for ti in range(0,Nt):
		KEprime[j,:] = KE[j,:] - KE_BG[j];

# Average
KE_av = KEprime[:,:,0];
for ti in range(1,Nt):
	KE_av[:,:] = KE_av[:,:] + KEprime[:,:,ti];
KE_av = KE_av / Nt;

plt.contourf(KE_av);
plt.show();

NN = int((2*N**2)**0.5)+1;
KEspectral = np.zeros(NN);
for K in range(1,NN):
	print K;
	for i in range(0,N):
		Ki = Kabs[i]
		for j in range(0,N):
			if Ki**2 + Kabs[j]**2 == K:
				KEspectral[K] = KEspectral[K] + KE_av[j,i];
			

plt.plot(KEspectral[5:NN-1]);
plt.show()




