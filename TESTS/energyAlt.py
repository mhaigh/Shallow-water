# energyAlt
#=========================================================

# Code for calculation kinetic energy and potential energy for some solution (u,v,eta), found by RSW... code.
# When running this code, the variables in the input file must match those that are originally used to calculate (u,v,h),
# which are pre-solved and loaded below.
#=========================================================

import numpy as np
import matplotlib.pyplot as plt

import funcs

from inputFile_1L import *
from forcing_1L import *

#=========================================================

u_nd = np.load('/home/mike/Documents/GulfStream/Code/DATA/1L/' + str(FORCE) + '/' + str(BG) + '/u_nd_' + str(Fpos) + str(N) + '.npy');
v_nd = np.load('/home/mike/Documents/GulfStream/Code/DATA/1L/' + str(FORCE) + '/' + str(BG) + '/v_nd_' + str(Fpos) + str(N) + '.npy');
eta_nd = np.load('/home/mike/Documents/GulfStream/Code/DATA/1L/' + str(FORCE) + '/' + str(BG) + '/eta_nd_' + str(Fpos) + str(N) + '.npy');

I = np.complex(0,1);		# Define I = sqrt(-1)

# In order to calculate the energy forced into the system, we require the energy of the full system
etaFull = np.zeros((N,N2,Nt));
uFull = np.zeros((N,N2,Nt));
for j in range(0,N):
	etaFull[j,:,:] = eta_nd[j,:,:] + H0_nd[j];
	uFull[j,:,:] = u_nd[j,:,:] + U0_nd[j];


eta_x = np.zeros((N,N2,Nt));
eta_y = np.zeros((N,N2,Nt));
u_x = np.zeros((N,N2,Nt));
u_y = np.zeros((N,N2,Nt));
v_x = np.zeros((N,N2,Nt));
v_y = np.zeros((N,N2,Nt));
u_xx = np.zeros((N,N2,Nt));
u_yy = np.zeros((N,N2,Nt));
v_xx = np.zeros((N,N2,Nt));
v_yy = np.zeros((N,N2,Nt));
u_nd_x = np.zeros((N,N2,Nt));
u_nd_y = np.zeros((N,N2,Nt));
u2_eta_x = np.zeros((N,N2,Nt));
u2_eta_y = np.zeros((N,N2,Nt));

u2_eta = etaFull * (uFull**2 + v_nd**2);

F1_ndt = np.zeros((N,N2,Nt));
F2_ndt = np.zeros((N,N2,Nt));
F3_ndt = np.zeros((N,N2,Nt));

for ti in range(0,Nt):
	t = T_nd[ti];
	F1_ndt[:,:,ti] = F1_nd[:,:] * np.cos(omega_nd*t);
	F2_ndt[:,:,ti] = F2_nd[:,:] * np.cos(omega_nd*t);
	F3_ndt[:,:,ti] = F3_nd[:,:] * np.cos(omega_nd*t);
	

# Need some derivatives:
for ti in range(0,Nt):
	eta_x[:,:,ti] = funcs.diff(etaFull[:,:,ti],1,1,dx_nd);
	eta_y[:,:,ti] = funcs.diff(etaFull[:,:,ti],0,0,dy_nd);
	u_x[:,:,ti] = funcs.diff(uFull[:,:,ti],1,1,dx_nd);
	u_xx[:,:,ti] = funcs.diff(u_x[:,:,ti],1,1,dx_nd);
	u_y[:,:,ti] = funcs.diff(uFull[:,:,ti],0,0,dx_nd);
	u_yy[:,:,ti] = funcs.diff(u_y[:,:,ti],0,0,dx_nd);
	v_x[:,:,ti] = funcs.diff(v_nd[:,:,ti],1,1,dx_nd);
	v_xx[:,:,ti] = funcs.diff(v_x[:,:,ti],1,1,dx_nd);
	v_y[:,:,ti] = funcs.diff(v_nd[:,:,ti],0,0,dx_nd);
	v_yy[:,:,ti] = funcs.diff(v_y[:,:,ti],0,0,dx_nd);
	u_nd_x[:,:,ti] = funcs.diff(u_nd[:,:,ti],1,1,dx_nd);
	u_nd_y[:,:,ti] = funcs.diff(u_nd[:,:,ti],0,0,dy_nd);
	u2_eta_x[:,:,ti] = funcs.diff(u2_eta[:,:,ti],1,1,dx_nd);
	u2_eta_y[:,:,ti] = funcs.diff(u2_eta[:,:,ti],0,0,dy_nd);
	
# Break down the energy into parts, define each part separately.
E1 = - etaFull * (uFull * eta_x + v_nd * eta_y);	# Contains geostrophic components
E2 = (Ro / Re) * etaFull * (uFull * (u_xx + u_yy) + v_nd * (v_xx + v_yy));	# Viscous terms
E3 = - gamma_nd * etaFull * (uFull**2 + v_nd**2);						# Frictional terms
E4 = 0.5 * Ro * (etaFull * (u_x + v_y) * (uFull**2 + v_nd**2) + uFull * u2_eta_x + v_nd * u2_eta_y);					# Divergence terms
E5 = etaFull * uFull * F1_ndt + etaFull * v_nd * F2_ndt + 0.5 * Ro * (uFull**2 + v_nd**2) * F3_ndt;	# Forcing terms

KE = np.zeros((N,N2,Nt));
KEtot = np.zeros(Nt);
for ti in range(1,Nt):
	KE[:,:,ti] = KE[:,:,ti-1] +  E1[:,:,ti] + E2[:,:,ti] + E3[:,:,ti] + E4[:,:,ti] + E5[:,:,ti];
	KEtot[ti] = np.trapz(np.trapz(KE[:,:,ti],x_nd,dx_nd,1),y_nd,dy_nd,0);


plt.plot(KEtot);
plt.show()


