# thickness.py
# A module containing buoyancy-related functions
#=====================================================================

import numpy as np
import matplotlib.pyplot as plt

from diagnostics import diff, extend, timeAverage

#=====================================================================

# fluxes
def fluxes(u,v,h,U0_nd,H0_nd,N,Nt):
# Calculates 6 PV flux terms
	
	uh = u * h;
	vh = v * h;
		
	Uh = np.zeros((N,N,Nt));	
	uH = np.zeros((N,N,Nt));
	vH = np.zeros((N,N,Nt));
	for i in range(0,N):
		for ti in range(0,Nt):
			Uh[:,i,ti] = U0_nd[:] * h[:,i,ti];
			uH[:,i,ti] = u[:,i,ti] * H0_nd[:];
			vH[:,i,ti] = v[:,i,ti] * H0_nd[:];

	UH = U0_nd * H0_nd;

	return uh, uH, Uh, UH, vh, vH;

#=====================================================================

# footprint
# A function that calculates the thickness footprint of the 1L SW solution as produced by RSW_1L.py
def footprint(uh,uH,Uh,vh,vH,x_nd,y_nd,T_nd,dx_nd,dy_nd,dt_nd,N,Nt):

	bu_full = uh + uH + Uh;		# Zonal thickness flux
	bv_full = vh + vH;			# Meridional thickness flux

	bu_full = timeAverage(bu_full,T_nd,Nt);
	bv_full = timeAverage(bv_full,T_nd,Nt);

	# Calculate the footprint.
	B = - diff(bu_full,1,1,dx_nd) - diff(bv_full,0,0,dy_nd);

	B = extend(B);

	# We are interested in the zonal average of the footprint
	B_xav = np.trapz(B,x_nd,dx_nd,axis=1);

	return B, B_xav

#=====================================================================

# EEF
def EEF(B_xav,y_nd,y0_nd,y0_index,dy_nd,N):

	# Define two y arrays, with all gridpoints north and south of the forcing location.
	y_north = y_nd[y0_index:N];
	y_south = y_nd[0:y0_index+1];
	# Define two corresponding P_xav arrays
	B_north = B_xav[y0_index:N];
	B_south = B_xav[0:y0_index+1];
	
	Babs_north = abs(B_north);
	Babs_south = abs(B_south);
	yabs_north = abs(y_north - y0_nd);
	yabs_south = abs(y_south - y0_nd);

	# Now calculate the 6 integrals
	int1_north = np.trapz(B_north,y_north,dy_nd);
	int1_south = np.trapz(B_south,y_south,dy_nd);

	norm1_north = np.trapz(Babs_north*yabs_north,y_north,dy_nd);
	norm1_south = np.trapz(Babs_south*yabs_south,y_south,dy_nd);
	norm2_north = np.trapz(Babs_north,y_north,dy_nd);
	norm2_south = np.trapz(Babs_south,y_south,dy_nd);

	EEF_north = (int1_north * norm1_north / norm2_north);
	EEF_south = (int1_south * norm1_south / norm2_south);
	
	EEF_array = np.array([EEF_north, EEF_south]);

	return EEF_array;
