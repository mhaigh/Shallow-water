# BG_state.py
#=======================================================

# This module contains functions which define the background flow and the background SSH.
# All BG flows here satisfy geostrophic balance.
# Note that in the 2-layer system, BOTH BG layer thicknesses depend on BOTH zonal background flows.

import numpy as np
import matplotlib.pyplot as plt

#=======================================================

def BG_uniform_none(Umag1,H1_flat,H2_flat,rho1_nd,rho2_nd,f0,beta,g,y,N):
	"(1) Uniform background flow and (2) no background flow"

	U1 = np.zeros(N);
	U2 = np.zeros(N);
	H1 = np.zeros(N);
	H2 = np.zeros(N);

	# Assign all values.
	for j in range(0,N):
		U1[j] = Umag1; 
		H1[j] = - U1[j] * (f0 * y[j] + beta * y[j]**2 / 2) / (g * rho2_nd);
		H2[j] = - rho1_nd * H1[j];

	# Add on the motionless layer thicknesses.
	H1 = H1 + H1_flat;
	H2 = H2 + H2_flat;

	return U1, U2, H1, H2
	
#=======================================================

def BG_Gaussian_none(Umag1,sigma,JET_POS,H1_flat,H2_flat,rho1_nd,rho2_nd,f0,beta,g,y,Ly,N):
	"Gaussian BG flow"

	from scipy.special import erf

	U1 = np.zeros(N);
	U2 = np.zeros(N);
	H1 = np.zeros(N);
	H2 = np.zeros(N);

	if JET_POS == 'CENTER':
		y0_jet = 0.0;
	elif JET_POS == 'NORTH':
		y0_jet = 0.25 * Ly;
	elif JET_POS == 'SOUTH':
		y0_jet = - 0.25 * Ly;
	
	l = Ly / 2;
	a = Umag1 / (np.exp(l**2 / (2. * sigma**2)) - 1.);	# Maximum BG flow velocity Umag1
	# Assign all values.
	for j in range(0,N):
		yy0 = y[j] - y0_jet;
		U1[j] = a * np.exp((l**2 - yy0**2) / (2. * sigma**2)) - a;		# -a ensures U0 is zero on the boundaries
		H1[j] = a * (beta * sigma**2 * np.exp((l**2 - yy0**2) / (2.0 * sigma**2))
					- np.sqrt(np.pi/2.) * sigma * y0_jet * beta * np.exp(l**2 / (2. * sigma**2)) * erf(yy0 / (np.sqrt(2.) * sigma))
					- np.sqrt(np.pi/2.) * sigma * f0 * np.exp(l**2 / (2. * sigma**2)) * erf(yy0 / (np.sqrt(2) * sigma))
					+ f0 * y[j] + beta * y[j]**2 / 2.) / (g * rho2_nd); #erf(0);
		H2[j] = - rho1_nd * H1[j];


	# Add on the motionless layer thicknesses.
	H1 = H1 + H1_flat;
	H2 = H2 + H2_flat;

	return U1, U2, H1, H2

#=======================================================

def BG_LapGauss(Umag,sigma,JET_POS,Hflat,rho1_nd,rho2_nd,f0,beta,g,y,Ly,N):
	"Laplacian-Gaussian background flow"

	from scipy.special import erf

	U0 = np.zeros(N);
	H0 = np.zeros(N);

	#if JET_POS == 'CENTER':
	#	y0_jet = 0;
	#elif JET_POS == 'NORTH':
	#	y0_jet = 0.25 * Ly;
	#elif JET_POS == 'SOUTH':
	#	y0_jet = - 0.25 * Ly;
	
	l = Ly / 2;
	a = Umag / (np.exp(l**2 / (2. * sigma**2)) - 1.);	# Maximum BG flow velocity Umag
	for j in range(0,N):
		U0[j] = a * (1. - y[j]**2 / (2.*sigma**2)) * (np.exp((l**2 - y[j]**2) / (2. * sigma**2)) - 1.);
		H0[j] = a * (beta * sigma**2 * np.exp((l**2 - y[j]**2) / (2.0 * sigma**2))
					- np.sqrt(np.pi/2.) * sigma * f0 * np.exp(l**2 / (2. * sigma**2)) * erf(y[j] / (np.sqrt(2) * sigma))
					# Extra recirculation terms herein.
					- 0.5 * f0 * y[j] * np.exp((l**2 - y[j]**2) / (2. * sigma**2))
					+ 0.5 * np.sqrt(np.pi/2.) * f0 * sigma * np.exp(l**2 / (2. * sigma**2)) * erf(y[j] / (np.sqrt(2) * sigma))
					- 0.5 * beta * (y[j]**2 + 2 * sigma**2) * np.exp((l**2 - y[j]**2) / (2. * sigma**2))
					# Adjusted Coriolis contribution herein.
					+ f0 * (y[j] - y[j]**3 / (6.0 * sigma**2)) + beta * (y[j]**2 / 2. - y[j]**4 / (8.0 * sigma**2))) / g + Hflat; #erf(0);

	H1 = H1 + H1_flat;
	H2 = H2 + H2_flat;

	return U0, H0
	

#=======================================================
	
def BG_none_none(H1_flat,H2_flat,N):
	"Zero BG flow"

	U1 = np.zeros(N);
	U2 = np.zeros(N);
	H1 = np.zeros(N);
	H2 = np.zeros(N);

	H1 = H1 + H1_flat;
	H2 = H2 + H2_flat;

	return U1, U2, H1, H2

#=======================================================
