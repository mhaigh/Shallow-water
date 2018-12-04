# parallel_EIG_DECOMP.py
#=======================================================

# This code decomposes a 1L SW solution, as produced by RSW_1L.py into into eigenmodes components,
# producing a set of weights corresponding the to set of eigenmodes found by runnning modules located in eigSolver.py
# First it defines the solution, which can either be defined from a file (FILE) or by running RSW_1L.py (NEW).

#====================================================

import os
import sys
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import time

from eig import eigSolver, eigDiagnostics
from core import diagnostics, solver, forcing, energy
##from output import output, output_read
from parallel import parallelDiags

from inputFile import *

#====================================================

def EIG_DECOMP_main(U0_set,pi):

	# Need to loop over a set of background states.
	NU = len(U0_set)
	theta = np.zeros((NU,N,dim),dtype=complex) 	# Initialise the set of weights; these will be complex.
	val = np.zeros((NU,N,dim),dtype=complex)
	count = np.zeros((NU,N,dim),dtype=int)
	count2 = np.zeros((NU,N,dim),dtype=int)
	ratio = np.zeros((NU,N,dim))

	# Start the loop.
	for ui in range(0,NU):

		# Start by defining the background state.
		for j in range(0,N):
			U0[j] = U0_set[ui]
			H0[j] = - (U0[j] / g) * (f0 * y[j] + beta * y[j]**2 / 2) + Hflat
		U0_nd = U0 / U
		H0_nd = H0 / chi		
	
		# The 1L SW solution
		#====================================================

		I = np.complex(0.0,1.0);

		# Coefficients
		a1,a2,a3,a4,b4,c1,c2,c3,c4 = solver.SOLVER_COEFFICIENTS(Ro,Re,K_nd,f_nd,U0_nd,H0_nd,omega_nd,gamma_nd,dy_nd,N);
		# Solver (free-slip)
		solution = solver.FREE_SLIP_SOLVER4(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ro*Ftilde1_nd,Ro*Ftilde2_nd,Ftilde3_nd,N,N2);

		solution = eigDiagnostics.switchKsign(solution,N)

		# Analysis
		#====================================================

		# Loop over desired wavenumbers (for tests, this may not be the full r'ange of wavenumbers)
		# ii indexes arrays storing information at ALL wavenumbers k
		# i indexes arrays storing information ONLY at wavenumbers used in the decomposition.

		a1,a2,a3,a4,b1,b4,c1,c2,c3,c4 = eigSolver.EIG_COEFFICIENTS2(Ro,Re,K_nd,f_nd,U0_nd,H0_nd,gamma_nd,dy_nd,N);

		for i in range(0,N):	 	

			k = K_nd[i]
			#print('k = ' + str(k))
	
			# Eigenmodes, eigenvalues and count.
			#====================================================

			# Use free-slip solver
			val[ui,i,:], vec = eigSolver.FREE_SLIP_EIG(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,N,N2,i,False);

			# Each eigenmode is currently a unit vector, but we normalise so that each mode contains unit energy.
			#==
	
			# Extract the three components.
			u_vec, v_vec, h_vec = eigDiagnostics.vec2vecs(vec,N,dim,BC);	
		
			# Calculate the contained in each component.
			E = np.zeros(dim);	
			for wi in range(0,dim):
				EE = energy.E_anomaly_EIG(u_vec[:,wi],v_vec[:,wi],h_vec[:,wi],H0_nd,U0_nd,Ro,y_nd,dy_nd);
				# Normalise each vector by the square root of the energy.
				u_vec[:,wi], v_vec[:,wi], h_vec[:,wi] = u_vec[:,wi] / np.sqrt(EE), v_vec[:,wi] / np.sqrt(EE), h_vec[:,wi] / np.sqrt(EE);
			# Rebuild the vector. This should have unit energy perturbation. 
			# (There are more direct ways of executing this normalisation, but this method is the safest.)
			vec = eigDiagnostics.vecs2vec(u_vec,v_vec,h_vec,N,dim,BC);

			# Order modes by meridional pseudo wavenumber (count).
			count[ui,i,:], count2[ui,i,:], ratio[ui,i,:], i_count = eigDiagnostics.orderEigenmodes(u_vec,x_nd,k,N,dim)
			#count, i_count = eigDiagnostics.orderEigenmodes2(vec,val,N,False)

	
			# Comment out this line, depending on which EIG_COEFFICIENTS function is being called.
			#val = val / (2. * np.pi * I * Ro);

			#====================================================
		
			Phi = solution[:,i];		# 1. Assign the solution corresponding to wavenumber k=K_nd[ii].

			theta[ui,i,:] = np.linalg.solve(vec,Phi); 				# 2.

	np.save('theta_array'+str(pi),theta)
	np.save('val_array'+str(pi),val)
	np.save('count_array'+str(pi),count) 
	np.save('count2_array'+str(pi),count) 
	np.save('ratio_array'+str(pi),count)  
	
	# Return the weights, eigenvalues, and meridional wavenumber.
	#return theta, val, count
	return

# End function
#====================================================

# Preamble before looping through function.
#====================================================

# Define BG flow set and number of processors. 
pe = 4
nn = 61
U0_set = np.linspace(-0.1,0.1,nn);

# Split U0_set into pe sets for parallelisation.
sets = parallelDiags.splitDomain(U0_set,nn,pe)

# Define forcing. This is constant throughout if only BG flow varies.
F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing.forcing_cts2(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE,AmpF_nd,f_nd,f0_nd,bh,dx_nd,dy_nd)

# Initialise output arrays
#====================================================

# For each BG flow, for each wavenumber k, for each mode:
# initialise weights, eigenvalues, meridional wavenumber.  
theta_array = np.zeros((nn,N,dim),dtype=complex)
val_array = np.zeros((nn,N,dim),dtype=complex)
count_array = np.zeros((nn,N,dim),dtype=int)

# Main function
#====================================================

if __name__ == '__main__':
	jobs = [];
	for pi in range(0,pe):
		p = mp.Process(target=EIG_DECOMP_main,args=(sets[pi],pi));
		jobs.append(p);
		p.start();

	for p in jobs:
		p.join();


# Collect results from each processor into single array.
parallelDiags.buildArray('theta_array',nn,pe)
parallelDiags.buildArray('val_array',nn,pe)
parallelDiags.buildArray('count_array',nn,pe)
parallelDiags.buildArray('count2_array',nn,pe)
parallelDiags.buildArray('ratio_array',nn,pe)









	
	
