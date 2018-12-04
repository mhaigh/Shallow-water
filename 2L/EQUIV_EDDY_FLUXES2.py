# EQUIV_EDDY_FLUXES.py
#=======================================================
# This is an executable code that solves the 1L shallow water system a number of times, each time storing the equivalent eddy flux.
#=======================================================

import os
import numpy as np
import multiprocessing as mp
import time
from scipy.ndimage.measurements import center_of_mass

from core import solver, PV, diagnostics, BG_state

from parallel import parallelDiags

from inputFile import *

#=======================================================

start = time.time()

# Can test against U0 or y0, or find the buoyancy vs U0 or y0.
TEST = 'U0'

#=======================================================

# Initialise tests

if TEST == 'U0':

	pe = 1		# Number of processors


	Nu = 20;

	#test_set = np.array((50.,60.,70.))
	#test_set = np.array((50.,100.,200.))
	#test_set = np.array((60000.,90000.,12000.)) # r0
	#test_set = np.linspace(0.015,0.045,Nu) * 3840000.0
	#test_set = np.linspace(0.4,1.2,Nu)
	test_set = np.linspace(-0.3,0.5,Nu);

	# Now split the input set into pe sample sets.
	sets = parallelDiags.splitDomain(test_set,Nu,pe)

	# Define forcing.
	F1, F2, F3, F4, F5, F6, Ftilde1, Ftilde2, Ftilde3, Ftilde4, Ftilde5, Ftilde6 = forcing.forcing_cts2(x,y,K,y0,r0,N,'BALANCED',AmpF_nd,rho1_nd,rho2_nd,f,1,bh,dx,dy)

if TEST == 'y0':
	y0_min = y[0] + r0;					# We want to keep the forcing at least one gridpoint away from the boundary
	y0_max = y[N-1] - r0;
	y0_set = [];						# Initialise an empty set of forcing latitudes
	y0_index_set = [];
	for j in range(0,N):
		if y0_min <= y[j] <= y0_max:
			y0_set.append(y[j])			# Build the set of forcing locations, all at least 1 gridpoint away from the boundary.	
			y0_index_set.append(j)
	y0_set = np.array(y0_set)
	y0_index_set = np.array(y0_index_set)
	nn = np.shape(y0_set)[0];
	a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,c5,d1,d3,d4,d5,e4,e5,f1,f2,f3,f4 = solver.SOLVER_COEFFICIENTS(Ro,Re,K,f,U1,U2,H1,H2,rho1_nd,rho2_nd,omega,gamma,dy,N)
		

#=======================================================


def EEF_main(set_,pi):

	NU = len(set_)
	
	# Initialise output arrays
	EEF_array = np.zeros((NU,2))

	# Now start the loop over each forcing index.
	for ui in range(0,NU):

		# Redefine U0 and H0.
		#sigma = set_[ui]
		Umag1 = set_[ui]
 		U1, U2, H1, H2 = BG_state.BG_uniform_none(Umag1,H1_flat,H2_flat,rho1_nd,rho2_nd,f0,beta,g,y,N);
		U1 = U1 / U;	U2 = U2 / U
		H1 = H1 / chi;	H2 = H2 / chi 

		# Solver coeffs depend on BG state.
		a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,c5,d1,d3,d4,d5,e4,e5,f1,f2,f3,f4 = solver.SOLVER_COEFFICIENTS(Ro,Re,K,f,U1,U2,H1,H2,rho1_nd,rho2_nd,omega,gamma,dy,N)
		
		# Solve.
		solution = solver.FREE_SLIP_SOLVER(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,c5,d1,d3,d4,d5,e4,e5,f1,f2,f3,f4,Ro*Ftilde1,Ro*Ftilde2,Ftilde3,Ro*Ftilde4,Ro*Ftilde5,Ftilde6,N,N2)

		# Extract flow components.
		utilde, vtilde, htilde = solver.extractSols(solution,N,N2,BC)
		u, v, h = solver.SPEC_TO_PHYS(utilde,vtilde,htilde,T,Nt,dx,omega,N)

		# Take real part.
		u = np.real(u)
		v = np.real(v)
		h = np.real(h)

		# For use in PV and footprint calculations: the 'full' zonal velocities and interface thicknesses.
		u_full = np.zeros((N,N,Nt,2))
		h_full = np.zeros((N,N,Nt,2))
		for j in range(0,N):
			u_full[j,:,:,0] = u[j,:,:,0] + U1[j]
			u_full[j,:,:,1] = u[j,:,:,1] + U2[j]
			h_full[j,:,:,0] = h[j,:,:,0] + H1[j]
			h_full[j,:,:,1] = h[j,:,:,1] + H2[j]

		# Call function calculate PV in upper layer.
		q = np.zeros((N,N,Nt,2)); q_full = np.zeros((N,N,Nt,2)); Q = np.zeros((N,2))
		q[:,:,:,0], q_full[:,:,:,0], Q[:,0] = PV.vort(u[:,:,:,0],v[:,:,:,0],h[:,:,:,0],u_full[:,:,:,0],h_full[:,:,:,0],H1,U1,N,Nt,dx,dy,f)
	
		# Calculate footprints using previously calculated PV.
		P, P_xav = PV.footprint(u_full[:,:,:,0],v[:,:,:,0],q_full[:,:,:,0],x,y,dx,dy,T,Nt)
	
		# EEF
		EEF_array[ui,:], l = PV.EEF(P_xav,y,y0,y0_index,dy,N)

		# End loop.

	# Save output from processor pi.
	np.save('EEF_array'+str(pi),EEF_array)

	# End function.


#=======================================================

if __name__ == '__main__':
	jobs = []
	for pi in range(0,pe):
		p = mp.Process(target=EEF_main,args=(sets[pi],pi))
		jobs.append(p)
		p.start()

	for p in jobs:
		p.join()
	
# Collect results, output in individual files
parallelDiags.buildArray('EEF_array',Nu,pe)

#=======================================================


elapsed = time.time() - start
elapsed = np.ones(1) * elapsed
print(elapsed)

#=======================================================



# If TEST==y0, matrix only needs to be defined once, but forcing must be defined each iteration.
#	if TEST == 'y0':
#		y0 = y0_set[ii];				# Redefine y0 and the forcing in each run.
#		y0_index = y0_index_set[ii];
#		y0_nd = y0 / L;
##		# Forcing redefined each iteration.
#		F1, F2, F3, F4, F5, F6, Ftilde1, Ftilde2, Ftilde3, Ftilde4, Ftilde5, Ftilde6 = forcing.forcing_cts(x,y,K,y0,r0,N,FORCE1,AmpF_nd,f,U,L,rho1_nd,rho2_nd,dx,dy);	
#
	
	
	
