# PV
# A module containing PV-related functions
#=====================================================================

import numpy as np
import matplotlib.pyplot as plt

from diagnostics import diff, extend, timeAverage

#=====================================================================

# potentialVorticity
def potentialVorticity(u,v,h,u_full,h_full,H0_nd,U0_nd,N,Nt,dx_nd,dy_nd,f_nd,Ro):
	'''
	Calculate potential vorticity	
	'''

	RV_full = np.zeros((N,N,Nt));
	RV_prime = np.zeros((N,N,Nt));
	for ti in range(0,Nt):
		# Define the relative vorticities (RV_full=RV_BG+RV_prime, can always check this numerically)
		RV_full[:,:,ti] = diff(v[:,:,ti],1,1,dx_nd) - diff(u_full[:,:,ti],0,0,dy_nd);
	RV_prime[:,:,ti] = diff(v[:,:,ti],1,1,dx_nd) - diff(u[:,:,ti],0,0,dy_nd);
	RV_BG = - diff(U0_nd,2,0,dy_nd);	# This is defined outside the loop as it has no time-dependence.

	PV_full = np.zeros((N,N,Nt));
	PV_BG = np.zeros(N);
	for j in range(0,N):
		PV_BG[j] = (RV_BG[j] + f_nd[j] / Ro) / H0_nd[j];
		for i in range(0,N):
			for ti in range(0,Nt):
				PV_full[j,i,ti] = (RV_full[j,i,ti] + f_nd[j] / Ro) / h_full[j,i,ti];
		
	# Two options to define the PV induced in the forced system: (1) PV_full-PV_BG or (2) use the algebraic def. given in the report.
	PV_prime = np.zeros((N,N,Nt));	
	for j in range(1,N-1):
		for i in range(0,N):
			for ti in range(0,Nt):
				PV_prime[j,i,ti] = PV_full[j,i,ti] - PV_BG[j];		# Option 1
	#PV_prime = np.zeros((N,N,Nt));	# Option 2 - keep this commented out, just use it as a check.
	#for j in range(0,N):
	#	for i in range(0,N):
	#		for ti in range(0,Nt):
	#			PV_prime[j,i,ti] = (RV_full[j,i,ti] - f[j]) / (h_full[j,i,ti]) - (f[j] - RV_BG[j]) / (H0_nd[j]);

	return PV_prime, PV_full, PV_BG

#=====================================================================

# PV_instant
def PV_instant(u,v,h,u_full,h_full,H0_nd,U0_nd,N,Nt,dx_nd,dy_nd,f_nd,Ro):
	'''
	Instantaneous PV
	'''

	RV_full = diff(v,1,1,dx_nd) - diff(u_full,0,0,dy_nd);
	RV_BG = - diff(U0_nd,2,0,dy_nd);
	RV = diff(v,1,1,dx_nd) - diff(u,0,0,dy_nd)
	Q = (RV_BG + f_nd / Ro) / H0_nd
	
	q = np.zeros((N,N))
	for j in range(0,N):	
		q[j,:] = (RV[j,:] - Q[j] * h[j,:]) / H0_nd[j]

	return q

#=====================================================================
# potentialVorticity_linear
def potentialVorticity_linear(u,v,h,H0_nd,U0_nd,N,Nt,dx_nd,dy_nd,f_nd,Ro):
	'''
	Calculate linear potential vorticity.
	Return three components of linear PV anomaly.
	'''

	RV_prime1 = np.zeros((N,N,Nt));
	RV_prime2 = np.zeros((N,N,Nt));
	for ti in range(0,Nt):
		RV_prime1[:,:,ti] = diff(v[:,:,ti],1,1,dx_nd);
		RV_prime2[:,:,ti] = - diff(u[:,:,ti],0,0,dy_nd);
	RV_BG = - diff(U0_nd,2,0,dy_nd);

	PV_BG = np.zeros(N);
	for j in range(0,N):
		PV_BG[j] = (RV_BG[j] + f_nd[j] / Ro) / H0_nd[j];

	PV_prime1 = np.zeros((N,N,Nt));
	PV_prime2 = np.zeros((N,N,Nt));
	PV_prime3 = np.zeros((N,N,Nt));
	for i in range(0,N):
		for ti in range(0,Nt):
			PV_prime1[:,i,ti] = RV_prime1[:,i,ti] / H0_nd[:];
			PV_prime2[:,i,ti] = RV_prime2[:,i,ti] / H0_nd[:];
			PV_prime3[:,i,ti] = - PV_BG[:] * h[:,i,ti] / H0_nd[:];

	return PV_prime1, PV_prime2, PV_prime3

#====================================================

# fluxes
def fluxes(u,v,U0_nd,PV_prime,PV_BG,N,Nt):
	'''
	Calculates 6 PV flux terms
	'''
	
	uq = u * PV_prime
	vq = v * PV_prime
		
	Uq = np.zeros((N,N,Nt));	
	uQ = np.zeros((N,N,Nt));
	vQ = np.zeros((N,N,Nt));
	for i in range(0,N):
		for ti in range(0,Nt):
			Uq[:,i,ti] = U0_nd[:] * PV_prime[:,i,ti];
			uQ[:,i,ti] = u[:,i,ti] * PV_BG[:];
			vQ[:,i,ti] = v[:,i,ti] * PV_BG[:];

	UQ = U0_nd * PV_BG;

	return uq, Uq, uQ, UQ, vq, vQ

#====================================================

# footprint
# A function that calculates the PV footprint of the 1L SW solution as produced by RSW_1L.py
def footprint(uq,Uq,uQ,UQ,vq,vQ,x_nd,T_nd,dx_nd,dy_nd,N,Nt):
	'''
	This code calculates the PV footprint, i.e. the PV flux convergence defined by
	P = -(div(u*q,v*q)), where _av denotees the time average.
	We will take the time average over one whole forcing period, and subtract this PV flux
	convergence from the PV flux convergence at the end of one period.
	To calculate footprints, we use the full PV and velocity profiles.
	'''
		
	uq_full = uq + Uq + uQ		# Total zonal PV flux
	#for j in range(0,N):
		#qu_full[:,j,:] = qu_full[:,j,:] + UQ[j];
	vq_full = vq + vQ			# Total meridional PV flux	
	
	# Time-averaging
	uq_full = timeAverage(uq_full,T_nd,Nt)
	vq_full = timeAverage(vq_full,T_nd,Nt)

	# Calculate the footprint.
	P = - diff(uq_full,1,1,dx_nd) - diff(vq_full,0,0,dy_nd)

	P = extend(P)
		
	# We are interested in the zonal average of the footprint
	P_xav = np.trapz(P,x_nd,dx_nd,axis=1) # / 1 not required.

	return P, P_xav

#====================================================

# footprintComponents
def footprintComponents(uq,Uq,uQ,vq,vQ,x_nd,T_nd,dx_nd,dy_nd,N,Nt):
	'''
	A function that calculates the PV footprint of the 1L SW solution in terms of its components, allowing for analysis.
	The function calculates the following terms: (1) uq, (2) Uq, (3) uQ, (4) UQ, (5) vq and (6) vQ. (UQ has zero zonal derivative.)
	The zonal/meridional derivative of the zonal/meridional PV flux is taken, averaged over one forcing period.
	Lastly, the zonal averages are calculated and everything useful returned.
	'''

	# Time averaging.
	uq = timeAverage(uq,T_nd,Nt);
	Uq = timeAverage(Uq,T_nd,Nt);
	uQ = timeAverage(uQ,T_nd,Nt);
	vq = timeAverage(vq,T_nd,Nt);
	vQ = timeAverage(vQ,T_nd,Nt);

	# Derivatives (no need to operate on UQ) and time-averaging.
	P_uq = - diff(uq,1,1,dx_nd);
	P_uQ = - diff(uQ,1,1,dx_nd);
	P_Uq = - diff(Uq,1,1,dx_nd);
	P_vQ = - diff(vQ,0,0,dy_nd);
	P_vq = - diff(vq,0,0,dy_nd);

	# Extend all arrays to include the final x gridpoint.
	P_uq = extend(P_uq);
	P_uQ = extend(P_uQ);
	P_Uq = extend(P_Uq);
	P_vQ = extend(P_vQ);
	P_vq = extend(P_vq);

	# Normalisation by AmpF_nd not needed if normalised quanities are passed into the function.

	P = P_uq + P_uQ + P_Uq + P_vq + P_vQ;

	# Zonal averaging 
	P_uq_xav = np.trapz(P_uq,x_nd,dx_nd,axis=1);
	P_uQ_xav = np.trapz(P_uQ,x_nd,dx_nd,axis=1);
	P_Uq_xav = np.trapz(P_Uq,x_nd,dx_nd,axis=1);
	P_vq_xav = np.trapz(P_vq,x_nd,dx_nd,axis=1);
	P_vQ_xav = np.trapz(P_vQ,x_nd,dx_nd,axis=1);
	
	#P_xav = np.trapz(P_tav,x_nd[:N],dx_nd,axis=1);
	P_xav = P_uq_xav + P_uQ_xav + P_Uq_xav + P_vq_xav + P_vQ_xav;
	# Tests confirm that the multiple approaches for calculating P_xav and P_tav yield the same results.

	return P, P_uq, P_uQ, P_Uq, P_vq, P_vQ, P_xav, P_uq_xav, P_uQ_xav, P_Uq_xav, P_vq_xav, P_vQ_xav;

#====================================================

# footprint_shift
def footprint_shift(P,y_nd,dy_nd,x_nd,dx_nd,N):
	'''
	This function finds the extent of the zonal shift of the footprint.
	'''

	P_av = np.trapz(np.abs(P),y_nd,dy_nd,axis=0);

	x_shift = np.trapz(P_av*x_nd,x_nd,dx_nd) / np.trapz(P_av,x_nd,dx_nd);

	#indices = np.argsort(-P_av);
	#index = indices[0];
	#x_shift = x_nd[index];

	return x_shift;	


#====================================================

# EEF
def EEF(P_xav,y_nd,y0_nd,y0_index,dy_nd,N):
	''' 
	A function that calculates the equivalent eddy flux, given a zonally averaged footprint
	The code works by calculating six integrals (three each side of the forcing) that make up each component of the equivalent eddy flux:
	int1_north/south = int_{y > / < y0} P_xav dy;
	int2_north/south = int_{y > / < y0} |y| |P_xav| dy;
	int3_north/south = int_{y > / < y0} |P_xav| dy.
	'''

	# Define two y arrays, with all gridpoints north and south of the forcing location.
	y_north = y_nd[y0_index:N];
	y_south = y_nd[0:y0_index+1];
	# Define two corresponding P_xav arrays
	P_north = P_xav[y0_index:N];
	P_south = P_xav[0:y0_index+1];
	
	Pabs_north = abs(P_north);
	Pabs_south = abs(P_south);
	yabs_north = abs(y_north - y0_nd);
	yabs_south = abs(y_south - y0_nd);

	# Now calculate the 6 integrals
	int_north = np.trapz(P_north,y_north,dy_nd);
	int_south = np.trapz(P_south,y_south,dy_nd);

	l_north = np.trapz(Pabs_north*yabs_north,y_north,dy_nd) /  np.trapz(Pabs_north,y_north,dy_nd);
	l_south = np.trapz(Pabs_south*yabs_south,y_south,dy_nd) /  np.trapz(Pabs_south,y_south,dy_nd);	

	EEF_north = int_north * l_north;
	EEF_south = int_south * l_south;
	
	EEF_array = np.array([EEF_north, EEF_south]);
	l_array = np.array([l_north,l_south]);

	return EEF_array, l_array;

#====================================================

# firstMoment
def firstMoment(P_xav,y_nd,y0_nd,dy_nd,N):

	fm = np.trapz(P_xav * (y_nd - y0_nd),y_nd,dy_nd)

	return fm

#====================================================


# EEF_components
def EEF_components(P_xav,P_uq_xav,P_uQ_xav,P_Uq_xav,P_vq_xav,P_vQ_xav,y_nd,y0_nd,y0_index,dy_nd,omega_nd,N):
	'''
	This function works in the same way as the EEF function, but instead takes as input each individual component of the zonally averaged footprint
	(of which there are five) and returns the EEF contribution in the north and south from each one.  
	The five footprint components are (1) uq_xav, (2) Uq_xav, (3) uQ_xav, (4) vq_xav and (5) vQ_xav (UQ doesn't contribute - zero zonal derivative).
	'''

	# Define two y arrays, with all gridpoints north and south of the forcing location
	# and define 12 corresponding EEF component arrays. Two arrays are needed for the full footprint,
	# so that we can calculate the normalisation integrals - the contribution from each footprint component
	# is normalised by the same normalisation constant: norm1 / norm2.

	# y
	y_north = y_nd[y0_index:N];
	y_south = y_nd[0:y0_index+1];
	# uq
	uq_north = P_uq_xav[y0_index:N];
	uq_south = P_uq_xav[0:y0_index+1];
	# Uq
	Uq_north = P_Uq_xav[y0_index:N];
	Uq_south = P_Uq_xav[0:y0_index+1];
	# uQ
	uQ_north = P_uQ_xav[y0_index:N];
	uQ_south = P_uQ_xav[0:y0_index+1];
	# vq
	vq_north = P_vq_xav[y0_index:N];
	vq_south = P_vq_xav[0:y0_index+1];
	# vQ
	vQ_north = P_vQ_xav[y0_index:N];
	vQ_south = P_vQ_xav[0:y0_index+1];
	# P
	P_north = P_xav[y0_index:N];
	P_south = P_xav[0:y0_index+1];
	
	Pabs_north = abs(P_north);
	Pabs_south = abs(P_south);
	yabs_north = abs(y_north - y0_nd);
	yabs_south = abs(y_south - y0_nd);

	# The normalisation constants: norm_north & norm_south (to multiply the integrals of uq_north/south,...)
	norm1_north = np.trapz(Pabs_north*yabs_north,y_north,dy_nd);
	norm2_north = np.trapz(Pabs_north,y_north,dy_nd);
	norm1_south = np.trapz(Pabs_south*yabs_south,y_south,dy_nd);
	norm2_south = np.trapz(Pabs_south,y_south,dy_nd);

	if norm2_north == 0:
		norm_north = 1;
	else:
		norm_north = norm1_north / norm2_north;

	if norm2_south == 0:
		norm_south = 1;
	else:
		norm_south = norm1_south / norm2_south;	

	# Now integrate uq_north/south etc. (overwrite the original variables, not needed), multiply by normalisation constant and forcing frequency.
	# uq
	uq_north = np.trapz(uq_north,y_north,dy_nd) * norm_north * omega_nd;
	uq_south = np.trapz(uq_south,y_south,dy_nd) * norm_south * omega_nd;
	# Uq
	Uq_north = np.trapz(Uq_north,y_north,dy_nd) * norm_north * omega_nd;
	Uq_south = np.trapz(Uq_south,y_south,dy_nd) * norm_south * omega_nd;
	# uQ
	uQ_north = np.trapz(uQ_north,y_north,dy_nd) * norm_north * omega_nd;
	uQ_south = np.trapz(uQ_south,y_south,dy_nd) * norm_south * omega_nd;
	# vq
	vq_north = np.trapz(vq_north,y_north,dy_nd) * norm_north * omega_nd;
	vq_south = np.trapz(vq_south,y_south,dy_nd) * norm_south * omega_nd;
	# vQ
	vQ_north = np.trapz(vQ_north,y_north,dy_nd) * norm_north * omega_nd;
	vQ_south = np.trapz(vQ_south,y_south,dy_nd) * norm_south * omega_nd;

	EEF_north = uq_north + Uq_north + uQ_north + vq_north + vQ_north;
	EEF_south = uq_south + Uq_south + uQ_south + vq_south + vQ_south;	

	# Define a single array to be returned by the function, containing all necessary information.	
	EEF_array = np.zeros((6,2));
	EEF_array[0,:] = [EEF_north, EEF_south]; EEF_array[1,:] = [uq_north,uq_south];
	EEF_array[2,:] = [Uq_north, Uq_south]; EEF_array[3,:] = [uQ_north,uQ_south];
	EEF_array[4,:] = [vq_north, vq_south]; EEF_array[5,:] = [vQ_north,vQ_south];

#= np.array((,[uq_north,uq_south],[Uq_north,Uq_south],[uQ_north,uQ_south],[vq_north,vq_south],[vQ_north,vQ_south]));

	return EEF_array;

#====================================================

# findCOM
def findCOM(f):

	from scipy.ndimage.measurements import center_of_mass

	# For a function f, find the latitude at which its center of mass lies.
	
	return center_of_mass(f)
	
	
	



