# P_test.py
#====================================================

import PV
import diagnostics
import numpy as np
import matplotlib.pyplot as plt
import plotting

from inputFile import *

#====================================================

# This test code computes the various components of the footprint at different stages of the operation.
# TEST1: Do zonal derivative and averaging cause uq to have small contribution?
# TEST2: Do zonal averages strongly depend on including the final x gridpoint?
# TEST3: What causes vq behaviour?

#====================================================
# Load the solution, this has already been normalised.
u_nd = np.load('u_nd.npy');
v_nd = np.load('v_nd.npy');
eta_nd = np.load('eta_nd.npy');
	
eta_full = np.zeros((N,N,Nt));
u_full = np.zeros((N,N,Nt));
for j in range(0,N):
	eta_full[j,:,:] = eta_nd[j,:,:] + H0_nd[j];
	u_full[j,:,:] = u_nd[j,:,:] + U0_nd[j];

PV_prime, PV_full, PV_BG = PV.potentialVorticity(u_nd,v_nd,eta_nd,u_full,eta_full,H0_nd,U0_nd,N,Nt,dx_nd,dy_nd,f_nd);
uq, Uq, uQ, UQ, vq, vQ = PV.fluxes(u_nd,v_nd,U0_nd,PV_prime,PV_BG,N,Nt);

P, P_uq, P_uQ, P_Uq, P_vq, P_vQ, P_xav, P_uq_xav, P_uQ_xav, P_Uq_xav, P_vq_xav, P_vQ_xav = PV.footprintComponents(uq,Uq,uQ,vq,vQ,x_nd,T_nd,dx_nd,dy_nd,N,Nt);
#plotting.footprintComponentsPlot(uq,Uq,uQ,vq,vQ,P,P_uq,P_Uq,P_uQ,P_vq,P_vQ,P_xav,P_uq_xav,P_uQ_xav,P_Uq_xav,P_vq_xav,P_vQ_xav,x_nd,y_nd,N,Nt);
#plotting.plotPrimaryComponents(P_uq,P_vq,P_uq_xav,P_vq_xav,x_nd,y_nd,FORCE,BG,Fpos,N);

# Why does the contribution from u^prime * q^prime disappear?
# Is it because <d/dx(u^prime * q^prime)> = c * (u_prime * q^prime)?
# This is periodic in x, and only need be evaluated at the endpoints in x.


TEST1 = False;
if TEST1:
	uq_tav = diagnostics.timeAverage(uq,T_nd,Nt); 		# Take the time-average
	uq_tav_xav = np.zeros(N);
	for j in range(0,N):
		uq_tav_xav[j] = - (uq_tav[j,N-1] - uq_tav[j,0]);

	plt.plot(uq_tav_xav);
	plt.plot(P_uq_xav);
	plt.show();

# We conclude:
# The zonal derivative and the zonal average operators cancel each other out.
# So the contribution from the zonal PV flux is just the difference between uq at either end of the domain,
# but the domain is periodic so this is small!
# The average of the derivative of a period function is zero.

"""!!! This test will now fail, improvements have been made based upon it !!!"""
TEST2 = False;
if TEST2:
	# Define a second PV flux vq2 which includes the final periodic gridpoint in x.
	P_vq2 = np.zeros((N,N+1));
	P_vq2[:,0:N] = P_vq[:,:];
	P_vq2[:,N] = P_vq[:,0];
	P_vq_xav2 = np.trapz(P_vq2,x_nd,dx_nd,axis=1);
	
	plt.plot(P_vq_xav2);
	plt.plot(-P_vq_xav);
	plt.show();	

# We conclude:
# Including the extra grid-point doesn't affect the all-important contribution from vq,
# but does improve the accuracy of the other fluxes (they're averages are closer to zero).


TEST3 = True;
if TEST3:	
	# NEED TO FINISH THIS TEST, BUT ITS NOT TOO IMPORTANT.
	# Take vq, and evaluate [vq]_y0^1/2. Then integrate in x.
	# Does this give the same result as the long-winded method?
	vq_tav = - diagnostics.timeAverage(vq,T_nd,Nt);

	P_n = np.zeros(N+1); P_s = np.zeros(N+1);
	P_y0n = np.zeros(N+1); P_y0s = np.zeros(N+1);
	P_n[0:N] = vq_tav[N-1,:]; P_s[0:N] = vq_tav[0,:];
	P_y0n[0:N] = vq_tav[y0_index+1,:]; P_y0s[0:N] = vq_tav[y0_index-1,:];
	P_n[N] = vq_tav[N-1,0];	P_s[N] = vq_tav[0,0];
	P_y0n[N] = vq_tav[y0_index+1,0]; P_y0s[N] = vq_tav[y0_index-1,0];


	P_n = P_n - P_y0n;	P_s = P_y0s - P_s;
	
	P_n = np.trapz(P_n,x_nd,dx_nd);	P_s = np.trapz(P_s,x_nd,dx_nd);
	print(P_n,P_s)

	E1 = PV.EEF_vq(P_vq_xav,P_n,P_s,y_nd,y0_nd,dy_nd,omega_nd,N);
	E2 = PV.EEF(P_vq_xav,y_nd,y0_nd,dy_nd,omega_nd,N);
		
	print(E1);
	print(E2);
	print(' ');
	print(E1[0] - E1[1]);
	print(E2[0] - E2[1]);






















	
			
