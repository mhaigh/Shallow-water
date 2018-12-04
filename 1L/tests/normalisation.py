# normalisation

#=======================================================

import sys

import numpy as np

import diagnostics
import PV
import buoy
import forcing_1L
import solver
import output
import energy
import plotting

from inputFile_1L import *

#=======================================================

u_nd = np.load('u_nd.npy');
v_nd = np.load('v_nd.npy');
eta_nd = np.load('eta_nd.npy');

u_norm = u_nd / AmpF_nd;
v_norm = v_nd / AmpF_nd;
eta_norm = eta_nd / AmpF_nd;

# In order to calculate the vorticities/energies of the system, we require full (i.e. BG + forced response) u and eta
eta_full = np.zeros((N,N,Nt));
u_full = np.zeros((N,N,Nt));
for j in range(0,N):
	eta_full[j,:,:] = eta_nd[j,:,:] + H0_nd[j];
	u_full[j,:,:] = u_nd[j,:,:] + U0_nd[j];

# In order to calculate the vorticities/energies of the system, we require full (i.e. BG + forced response) u and eta
eta_full_norm = np.zeros((N,N,Nt));
u_full_norm = np.zeros((N,N,Nt));
for j in range(0,N):
	eta_full_norm[j,:,:] = eta_norm[j,:,:] + H0_nd[j];
	u_full_norm[j,:,:] = u_norm[j,:,:] + U0_nd[j];

#=======================================================
#=======================================================

# Calculate using normalised sols.
PV_prime_norm, PV_full_norm, PV_BG = PV.potentialVorticity(u_norm,v_norm,eta_norm,u_full_norm,eta_full_norm,H0_nd,U0_nd,N,Nt,dx_nd,dy_nd,f_nd);
uq_norm, Uq_norm, uQ_norm, UQ_norm, vq_norm, vQ_norm = PV.fluxes(u_norm,v_norm,U0_nd,PV_prime_norm,PV_BG,N,Nt);

# Calculate using un-normalised sols, then normalise
PV_prime, PV_full, PV_BG = PV.potentialVorticity(u_nd,v_nd,eta_nd,u_full,eta_full,H0_nd,U0_nd,N,Nt,dx_nd,dy_nd,f_nd);
uq, Uq, uQ, UQ, vq, vQ = PV.fluxes(u_nd,v_nd,U0_nd,PV_prime,PV_BG,N,Nt);

PV_prime, PV_full, PV_BG = PV_prime / AmpF_nd, PV_full / AmpF_nd, PV_BG / AmpF_nd;
uq, Uq, uQ, UQ, vq, vQ = uq / AmpF_nd, Uq / AmpF_nd, uQ / AmpF_nd, UQ / AmpF_nd, vq / AmpF_nd, vQ / AmpF_nd

plt.figure(1);
plt.subplot(231);
plt.contourf(PV_prime[:,:,ts]);
plt.colorbar();
plt.subplot(234);
plt.contourf(PV_prime_norm[:,:,ts]);
plt.colorbar();
plt.show();





















