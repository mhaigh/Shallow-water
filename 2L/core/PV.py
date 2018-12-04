# PV
# A module containing footprint-related functions
#=====================================================================

import numpy as np
import matplotlib.pyplot as plt

from diagnostics import diff, extend, timeAverage

#=====================================================================

# vort
# Calculate potential vorticity - the algorithm is the same for each layer and so can be called twice, altering the input.
def vort(u,v,h,u_full,h_full,H,U,N,Nt,dx,dy,f):

	RV_full = np.zeros((N,N,Nt));
	RV_prime = np.zeros((N,N,Nt));
	for ti in range(0,Nt):
		# Define the relative vorticities (RV_full=RV_BG+RV_prime, can always check this numerically)
		RV_full[:,:,ti] = diff(v[:,:,ti],1,1,dx) - diff(u_full[:,:,ti],0,0,dy);
		RV_prime[:,:,ti] = diff(v[:,:,ti],1,1,dx) - diff(u[:,:,ti],0,0,dy);
	RV_BG = - diff(U,2,0,dy);	# This is defined outside the loop as it has no time-dependence.
	
	# Now define two of the PVs
	PV_full = np.zeros((N,N,Nt));
	PV_BG = np.zeros(N);
	for j in range(0,N):
		PV_BG[j] = (RV_BG[j] + f[j]) / H[j];
		for i in range(0,N):
			for ti in range(0,Nt):
				PV_full[j,i,ti] = (RV_full[j,i,ti] + f[j]) / h_full[j,i,ti];
		
	# Two options to define the PV induced in the forced system: (1) PV_full-PV_BG or (2) use the algebraic def. given in the report.
	PV_prime = np.zeros((N,N,Nt));
	for j in range(1,N-1):
		for i in range(0,N):
			for ti in range(0,Nt):
				PV_prime[j,i,ti] = PV_full[j,i,ti] - PV_BG[j];		# Option 1
	#PV_prime = np.zeros((N,N));	# Option 2 - keep this commented out, just use it as a check.
	#for j in range(0,N):
	#	for i in range(0,N):
	#		PV_prime[j,i] = (H[j] * RV_prime[j,i] - h[j,i] * (RV_BG[j] + f[j])) / (eta_full[j,i] * H[j]);

	return PV_prime, PV_full, PV_BG

#====================================================

# footprint
# A function that calculates the PV footprint of the 1L SW solution as produced by RSW_1L.py
def footprint(u_full,v,q_full,x,y,dx,dy,T,Nt):
# This code calculates the PV footprint, i.e. the PV flux convergence defined by
# P = -(div(u*q,v*q)-div((u*q)_av,(v*q)_av)), where _av denotees the time average.
# We will take the time average over one whole forcing period, and subtract this PV flux
# convergence from the PV flux convergence at the end of one period.
# To calculate footprints, we use the full PV and velocity profiles.

	uq = q_full * u_full;		# Zonal PV flux
	vq = q_full * v;		# Meridional PV flux

	# Next step: taking appropriate derivatives of the fluxes. To save space, we won't define new variables, but instead overwrite the old ones.
	# From these derivatives we can calculate the 

	# Time-averaging
	uq = timeAverage(uq,T,Nt);
	vq = timeAverage(vq,T,Nt);

	# Calculate the footprint.
	P = - diff(uq,1,1,dx) - diff(vq,0,0,dy);

	P = extend(P);
		
	# We are interested in the zonal average of the footprint
	P_xav = np.trapz(P,x,dx,axis=1); # / 1 not required.
	
	return P, P_xav

#====================================================

# EEF
def EEF(P_xav,y,y0,y0_index,dy,N):
# A function that calculates the equivalent eddy flux, given a zonally averaged footprint
# The code works by calculating six integrals (three each side of the forcing) that make up each component of the equivalent eddy flux:
# int1_north/south = int_{y > / < y0} P_xav dy;
# int2_north/south = int_{y > / < y0} |y| |P_xav| dy;
# int3_north/south = int_{y > / < y0} |P_xav| dy.

	# Define two y arrays, with all gridpoints north and south of the forcing location.
	y_north = y[y0_index:N];
	y_south = y[0:y0_index+1];
	# Define two corresponding P_xav arrays
	P_north = P_xav[y0_index:N];
	P_south = P_xav[0:y0_index+1];
	
	Pabs_north = abs(P_north);
	Pabs_south = abs(P_south);
	yabs_north = abs(y_north - y0);
	yabs_south = abs(y_south - y0);

	# Now calculate the 6 integrals
	int_north = np.trapz(P_north,y_north,dy);
	int_south = np.trapz(P_south,y_south,dy);

	l_north = np.trapz(Pabs_north*yabs_north,y_north,dy) /  np.trapz(Pabs_north,y_north,dy);
	l_south = np.trapz(Pabs_south*yabs_south,y_south,dy) /  np.trapz(Pabs_south,y_south,dy);	

	EEF_north = int_north * l_north;
	EEF_south = int_south * l_south;
	
	EEF_array = np.array([EEF_north, EEF_south]);
	l_array = np.array([l_north,l_south]);

	return EEF_array, l_array;

