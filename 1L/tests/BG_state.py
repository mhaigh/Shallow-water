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
