# EIG_DECOMP_y0.py
#=======================================================

# Solve 1L SW system with forcing at every possible latitude, for fixed BG flow.
# Unlike other DECOMP codes, here we only need to caclulate the eigenmodes once.
# When running in parallel, threads do not communicate, so each one needs to
# calculate the set of eigenmodes themselves. Could alternatively pass it to main.

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

def EIG_DECOMP_main(y0_set,pi):

	I = np.complex(0.0,1.0)
	
	# Need to loop over a set of background states.
	Ny = len(y0_set)
	theta = np.zeros((Ny,N,dim),dtype=complex) 	# Initialise the set of weights; these will be complex.

	# Only theta will have Ny dimension. Only need one set of eigenmodes, so no dependence on forcing latitude.
	val = np.zeros((N,dim),dtype=complex)
	count = np.zeros((N,dim),dtype=int)
	count2 = np.zeros((N,dim),dtype=int)
	ratio = np.zeros((N,dim))

	# Find solution at every y0-value and for all i. Store it.
	#========================================================

	# Coefficients - no dependence on y0
	a1,a2,a3,a4,b4,c1,c2,c3,c4 = solver.SOLVER_COEFFICIENTS(Ro,Re,K_nd,f_nd,U0_nd,H0_nd,omega_nd,gamma_nd,dy_nd,N);

	solution = np.zeros((Ny,dim,N),dtype=complex)

	# Start the loop.
	for yi in range(0,Ny):

		# Get forcing latitude.
		y0_nd = y0_set[yi];

		# Redefine forcing, moving one gridpoint each iteration.
		F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing.forcing_cts2(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE,AmpF_nd,f_nd,f0_nd,bh,dx_nd,dy_nd)

		# The 1L SW solution.

		# Solver (free-slip)
		solution[yi,] = solver.FREE_SLIP_SOLVER4(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ro*Ftilde1_nd,Ro*Ftilde2_nd,Ftilde3_nd,N,N2);

		solution[yi,] = eigDiagnostics.switchKsign(solution[yi,],N)

	# Done finding solution.

	#====================================================



	# Find eigenmodes and decompose solution.
	#====================================================

	# Loop over desired wavenumbers (for tests, this may not be the full r'ange of wavenumbers)

	# Eig coefficients don't change. Define them just once.
	a1,a2,a3,a4,b1,b4,c1,c2,c3,c4 = eigSolver.EIG_COEFFICIENTS2(Ro,Re,K_nd,f_nd,U0_nd,H0_nd,gamma_nd,dy_nd,N);

	for i in range(0,N):	 	

		k = K_nd[i]

		#print('k = ' + str(k))
	
		# Eigenmodes, eigenvalues and count.
		#====================================================

		# Use free-slip solver
		val[i,:], vec = eigSolver.FREE_SLIP_EIG(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,N,N2,i,False);


		# Each eigenmode is currently a unit vector, but we normalise so that each mode contains unit energy.
		#==
	
		# Extract the three components.
		u_vec, v_vec, h_vec = eigDiagnostics.vec2vecs(vec,N,dim,BC);	
	
		# Calculate the contained in each component.
		E = np.zeros(dim);	
		for wi in range(0,dim):
			EE = energy.E_anomaly_EIG(u_vec[:,wi],v_vec[:,wi],h_vec[:,wi],H0_nd,U0_nd,Ro,y_nd,dy_nd)
			# Normalise each vector by the square root of the energy.
			u_vec[:,wi], v_vec[:,wi], h_vec[:,wi] = u_vec[:,wi] / np.sqrt(EE), v_vec[:,wi] / np.sqrt(EE), h_vec[:,wi] / np.sqrt(EE)
		# Rebuild the vector. This should have unit energy perturbation. 
		# (There are more direct ways of executing this normalisation, but this method is the safest.)
		vec = eigDiagnostics.vecs2vec(u_vec,v_vec,h_vec,N,dim,BC)



		# Eigenmodes now have unit time-mean energy.
		#==

		# Meridional pseudo wavenumber (count).
		count[i,:], count2[i,:], ratio[i,:], i_count = eigDiagnostics.orderEigenmodes(u_vec,x_nd,k,N,dim)

		# Now decompose for each forcing latitude.		
		for yi in range(0,Ny):

			# Decompose.
			theta[yi,i,:] = np.linalg.solve(vec,solution[yi,:,i])

	# End of eigenmode calculation and decomposition.
	#========================================================
	
	np.save('theta_array'+str(pi),theta)
	np.save('val_array',val)
	np.save('count_array',count) 
	np.save('count2_array',count2) 
	np.save('ratio_array',count)  
	
	# Don't return anything, arrays are saved instead.
	return

# End function
#====================================================

# Preamble before looping through function.
#====================================================

# Define y0 set and number of processors. 
pe = 1
y0_set = np.array([lat for lat in y_nd if -1./6 <= lat <= 1./6])
ny = len(y0_set)

# Split U0_set into pe sets for parallelisation.
sets = parallelDiags.splitDomain(y0_set,ny,pe)

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
parallelDiags.buildArray('theta_array',ny,pe)









	
	
