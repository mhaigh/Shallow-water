# PV
# A module containing PV-related functions
#=====================================================================

import numpy as np
import matplotlib.pyplot as plt

from diagnostics import diff, extend, timeAverage

#=====================================================================

# fluxes
def fluxes(u,v):
# Calculates momentum flux terms.
# For the purpose of EEFs of momentum, we only need three flux terms:
# uu, uv, and vv.
	
	uu = u * u;
	uv = u * v;
	vv = v * v;

	return uu, uv, vv

#====================================================

# footprint
def footprint(uu,uv,vv,x_nd,T_nd,dx_nd,dy_nd,N,Nt):
# A function that calculates the momentum footprint of the 1L SW solution as produced by RSW.py	
	
	# Time-averaging
	uu = timeAverage(uu,T_nd,Nt);
	uv = timeAverage(uv,T_nd,Nt);
	vv = timeAverage(vv,T_nd,Nt);

	# Two footprint terms to calculate
	Mu = - diff(uu,1,1,dx_nd) - diff(uv,0,0,dy_nd);
	Mv = - diff(uv,1,1,dx_nd) - diff(vv,0,0,dy_nd);

	Mu = extend(Mu);
	Mv = extend(Mv);
		
	# We are interested in the zonal average of the footprint
	Mu_xav = np.trapz(Mu,x_nd,dx_nd,axis=1);
	Mv_xav = np.trapz(Mv,x_nd,dx_nd,axis=1);

	return Mu, Mv, Mu_xav, Mv_xav

#====================================================

# EEF_mom
def EEF_mom(Mu_xav,Mv_xav,y_nd,y0_nd,y0_index,dy_nd,omega_nd,N):
# A function that calculates the equivalent eddy flux, given a zonally averaged footprint
# The code works by calculating six integrals (three each side of the forcing) that make up each component of the equivalent eddy flux:
# int1_north/south = int_{y > / < y0} P_xav dy;
# int2_north/south = int_{y > / < y0} |y| |P_xav| dy;
# int3_north/south = int_{y > / < y0} |P_xav| dy.

	# Define two y arrays, with all gridpoints north and south of the forcing location.
	y_north = y_nd[y0_index:N];
	y_south = y_nd[0:y0_index+1];
	# Define four corresponding momentum flux arrays.
	Mu_north = Mu_xav[y0_index:N];
	Mu_south = Mu_xav[0:y0_index+1];
	Mv_north = Mv_xav[y0_index:N];
	Mv_south = Mv_xav[0:y0_index+1];

	# Take absolute values.
	Mu_abs_north = abs(Mu_north);
	Mu_abs_south = abs(Mu_south);
	Mv_abs_north = abs(Mv_north);
	Mv_abs_south = abs(Mv_south);
	yabs_north = abs(y_north - y0_nd);
	yabs_south = abs(y_south - y0_nd);

	# Now calculate the 6 integrals for the u momentum fluxes.
	int1_north = np.trapz(Mu_north,y_north,dy_nd);
	int1_south = np.trapz(Mu_south,y_south,dy_nd);

	norm1_north = np.trapz(Mu_abs_north*yabs_north,y_north,dy_nd);
	norm1_south = np.trapz(Mu_abs_south*yabs_south,y_south,dy_nd);
	norm2_north = np.trapz(Mu_abs_north,y_north,dy_nd);
	norm2_south = np.trapz(Mu_abs_south,y_south,dy_nd);

	EEF_u_north = (int1_north * norm1_north / norm2_north) * omega_nd;
	EEF_u_south = (int1_south * norm1_south / norm2_south) * omega_nd;

	EEF_u = np.array([EEF_u_north,EEF_u_south]);

	# And do the same again for the v momentum fluxes.
	int1_north = np.trapz(Mv_north,y_north,dy_nd);
	int1_south = np.trapz(Mv_south,y_south,dy_nd);

	norm1_north = np.trapz(Mv_abs_north*yabs_north,y_north,dy_nd);
	norm1_south = np.trapz(Mv_abs_south*yabs_south,y_south,dy_nd);
	norm2_north = np.trapz(Mv_abs_north,y_north,dy_nd);
	norm2_south = np.trapz(Mv_abs_south,y_south,dy_nd);

	EEF_v_north = (int1_north * norm1_north / norm2_north) * omega_nd;
	EEF_v_south = (int1_south * norm1_south / norm2_south) * omega_nd;

	EEF_v = np.array([EEF_v_north, EEF_v_south]);

	return EEF_u, EEF_v


 
	
	



