# EQUIV_EDDY_FLUXES.py
#=======================================================

# Solve the SW system a number times, each time
# calculating the energy budget terms.

#=======================================================

import os
import numpy as np
import multiprocessing as mp
import time

from core import solver, diagnostics, BG_state, energy
from parallel import parallelDiags

from inputFile import *

#=======================================================

start = time.time();

# Can test against U0 or y0, or find the buoyancy vs U0 or y0.

#=======================================================

pe = 1		# Number of processors

# Initialise tests

Nu = 20;

#test_set = np.array((50.,60.,70.))
#test_set = np.array((50.,100.,200.))
#test_set = np.array((60000.,90000.,12000.)) # r0
#test_set = np.linspace(0.015,0.045,Nu) * 3840000.0
#test_set = np.linspace(0.4,1.2,Nu)
test_set = np.linspace(-0.1,0.1,Nu);

# Now split the input set into pe sample sets.
sets = parallelDiags.splitDomain(test_set,Nu,pe)

#=======================================================

# Forcing
F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing.forcing_cts2(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE,AmpF_nd,f_nd,f0_nd,bh,dx_nd,dy_nd)

def EEF_main(set_,pi):

	NU = len(set_)
	
	# Initialise output arrays
	Ef_av_array = np.zeros((NU,N,N))
	Ed_av_array = np.zeros((NU,N,N))

	# Now start the loop over each forcing index.
	for ui in range(0,NU):
		#print(ui)

		# Redefine U0 and H0
		#sigma = set_[ui]
		Umag = set_[ui]
 
		U0, H0 = BG_state.BG_uniform(Umag,Hflat,f0,beta,g,y,N)
		U0_nd = U0 / U
		H0_nd = H0 / chi

		# Solution
		a1,a2,a3,a4,b4,c1,c2,c3,c4 = solver.SOLVER_COEFFICIENTS(Ro,Re,K_nd,f_nd,U0_nd,H0_nd,omega_nd,gamma_nd,dy_nd,N)		
		solution = solver.FREE_SLIP_SOLVER(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ro*Ftilde1_nd,Ro*Ftilde2_nd,Ftilde3_nd,N,N2)
		utilde_nd, vtilde_nd, etatilde_nd = solver.extractSols(solution,N,N2,BC)
		u, v, h = solver.SPEC_TO_PHYS(utilde_nd,vtilde_nd,etatilde_nd,T_nd,dx_nd,omega_nd,N)
		
		# Take real part. Don't normalise by forcing ampltiude, as we want to compare energy input by forcing.
		u = np.real(u)
		v = np.real(v)
		h = np.real(h)
	
		# Calculate full flow quantities.
		u_full = diagnostics.fullFlow(u,U0_nd)
		h_full = diagnostics.fullFlow(h,H0_nd)

		#==

		# Energy budget terms.
		Ed, Ed_av_array[ui,] = energy.budgetDissipation3(U0_nd,H0_nd,u,v,h,Ro,Re,gamma_nd,dx_nd,dy_nd,T_nd,Nt,N)
		Ef, Ef_av_array[ui,] = energy.budgetForcing(u_full,v,h_full,F1_nd,F2_nd,F3_nd,Ro,N,T_nd,omega_nd,Nt)

		# End loop.

	np.save('Ed_av_array'+str(pi),Ed_av_array)
	np.save('Ef_av_array'+str(pi),Ef_av_array)

if __name__ == '__main__':
	jobs = []
	for pi in range(0,pe):
		p = mp.Process(target=EEF_main,args=(sets[pi],pi))
		jobs.append(p)
		p.start()

	for p in jobs:
		p.join()

# Collect results, output in individual files
parallelDiags.buildArray('Ef_av_array',Nu,pe)
parallelDiags.buildArray('Ed_av_array',Nu,pe)

elapsed = time.time() - start
elapsed = np.ones(1) * elapsed
print(elapsed)

	
	
	
