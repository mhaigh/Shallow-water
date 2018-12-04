# BG_state.py
#=======================================================

# This module contains functions which define the background flow, the background SSH, and the background PV.

import numpy as np
import matplotlib.pyplot as plt

#=======================================================

def BG_uniform(Umag,Hflat,f0,beta,g,y,N):
	"Uniform background flow"

	U0 = np.zeros(N);
	H0 = np.zeros(N);	
	for j in range(0,N):
		U0[j] = Umag; 			# (m s-1)
		H0[j] = - (U0[j] / g) * (f0 * y[j] + beta * y[j]**2 / 2) + Hflat;

	return U0, H0
	
#=======================================================

def BG_shear(Umag,shear,Hflat,f0,beta,g,y,Ly,N):
	"Zonal shear background flow"

	U0 = np.zeros(N);
	H0 = np.zeros(N);
	l = Ly / 2;
	a = Umag / l;
	for j in range(0,N):
		U0[j] = a * y[j] + shear;
		H0[j] = - (f0 * shear * y[j] + 0.5 * (f0 * a + beta * shear) * y[j]**2 + a * beta * y[j]**3 / 3.) / g + Hflat;

	return U0, H0

#=======================================================

def BG_Gaussian(Umag,sigma,JET_POS,Hflat,f0,beta,g,y,Ly,N):
	"Gaussian BG flow"

	from scipy.special import erf

	U0 = np.zeros(N);
	H0 = np.zeros(N);

	if JET_POS == 'CENTER':
		y0_jet = 0;
	elif JET_POS == 'NORTH':
		y0_jet = 0.25 * Ly;
	elif JET_POS == 'SOUTH':
		y0_jet = - 0.25 * Ly;
	
	l = Ly / 2;
	a = Umag / (np.exp(l**2 / (2. * sigma**2)) - 1.);	# Maximum BG flow velocity Umag
	for j in range(0,N):
		yy0 = y[j] - y0_jet;
		U0[j] = a * np.exp((l**2 - yy0**2) / (2. * sigma**2)) - a;		# -a ensures U0 is zero on the boundaries
		H0[j] = a * (beta * sigma**2 * np.exp((l**2 - yy0**2) / (2.0 * sigma**2))
					- np.sqrt(np.pi/2.) * sigma * y0_jet * beta * np.exp(l**2 / (2. * sigma**2)) * erf(yy0 / (np.sqrt(2.) * sigma))
					- np.sqrt(np.pi/2.) * sigma * f0 * np.exp(l**2 / (2. * sigma**2)) * erf(yy0 / (np.sqrt(2) * sigma))
					+ f0 * y[j] + beta * y[j]**2 / 2.) / g + Hflat; #erf(0);

	return U0, H0
	

#=======================================================

def BG_LapGauss(Umag,sigma,JET_POS,Hflat,f0,beta,g,y,Ly,N):
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


	return U0, H0
	

#=======================================================

def BG_quadratic(Umag,Hflat,f0,beta,g,y,Ly,N):
	"Quadratic background flow"

	Umag = -1.
	a = 1./Ly**2; b = 1/Ly; c = 0.5 - a * Ly**2 / 4
	scale = 1#c - b**2 / (4 * a)

	U0 = np.zeros(N)
	H0 = np.zeros(N)

	for j in range(0,N):
		U0[j] = Umag * (a * y[j]**2 + b * y[j] + c) / scale
		H0[j] = - Umag * ((a * beta / 4.0) * y[j]**4 + ((f0 * a + beta * b) / 3.0) * y[j]**3 + ((f0 * b + c * beta) / 2.0) * y[j]**2 + f0 * c * y[j]) / (g * scale) + Hflat

	return U0, H0

#=======================================================
	
def BG_zero(Hflat,chi,N):
	"Zero BG flow"

	U0 = np.zeros(N);
	for j in range(0,N):
		H0[j] = Hflat;

	return U0, H0

#=======================================================
