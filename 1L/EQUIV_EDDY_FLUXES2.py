# EQUIV_EDDY_FLUXES.py
#=======================================================
# This is an executable code that solves the 1L shallow water system a number of times, each time storing the equivalent eddy flux.
#=======================================================

import os
import numpy as np
import multiprocessing as mp
import time
from scipy.ndimage.measurements import center_of_mass

from core import solver, PV, momentum, diagnostics, BG_state, corr, energy
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
test_set = np.linspace(-0.3,0.5,Nu);

# Now split the input set into pe sample sets.
sets = parallelDiags.splitDomain(test_set,Nu,pe)

#=======================================================

# Forcing
F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing.forcing_cts2(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE,AmpF_nd,f_nd,f0_nd,bh,dx_nd,dy_nd)

def EEF_main(set_,pi):

	NU = len(set_)
	
	# Initialise output arrays
	EEF_array = np.zeros((NU,2))
	E_array = np.zeros((NU))	
	M_array = np.zeros((NU,N,N))
	N_array = np.zeros((NU,N,N))

	# Now start the loop over each forcing index.
	for ui in range(0,NU):

		# Redefine U0 and H0.
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
		
		# Take real part.
		u = np.real(u)
		v = np.real(v)
		h = np.real(h)
	
		# Normalise all solutions by the (non-dimensional) forcing amplitude. 
		u = u / AmpF_nd
		v = v / AmpF_nd
		h = h / AmpF_nd
		
		# In order to calculate the vorticities of the system, we require full (i.e. BG + forced response) u and eta.
		h_full = np.zeros((N,N,Nt))
		u_full = np.zeros((N,N,Nt))
		for j in range(0,N):
			h_full[j,:,:] = h[j,:,:] + H0_nd[j]
			u_full[j,:,:] = u[j,:,:] + U0_nd[j]

		#==

		# Correlations (can do before or after energy)
		M_array[ui,:,:] = corr.M(u,v,T_nd)
		N_array[ui,:,:] = corr.N(u,v,T_nd)

		#==

		# Energy
		#KE_BG, KE_BG_tot, PE_BG, PE_BG_tot = energy.energy_BG(U0_nd,H0_nd,Ro,y_nd,dy_nd,N)
		#KE, KE_tot = energy.KE(u_full,v,h_full,x_nd,y_nd,dx_nd,dy_nd,N)
		#PE, PE_tot = energy.PE(h_full,Ro,x_nd,y_nd,dx_nd,dy_nd,N)
		#E_tot = KE_tot + PE_tot - KE_BG_tot - PE_BG_tot
		# Use time-mean KE omitting h for now. Find a better way to do this
		KE = u**2 + v**2
		KE_tot = np.zeros(Nt)
		for ti in range(0,Nt):
			KE_tot[ti] = diagnostics.domainInt(KE[:,:,ti],x_nd,dx_nd,y_nd,dy_nd)

		E_array[ui] = diagnostics.timeAverage1D(KE_tot,T_nd,Nt) 

		#import matplotlib.pyplot as plt
		#plt.plot(KE_tot);plt.show()
		# Normalise by energy 
		#u = u / np.sqrt(E_array[ui]); v = v / np.sqrt(E_array[ui]); h = h / np.sqrt(E_array[ui])

		#==
	
		# Calculate PV fields and PV fluxes.
		PV_prime, PV_full, PV_BG = PV.potentialVorticity(u,v,h,u_full,h_full,H0_nd,U0_nd,N,Nt,dx_nd,dy_nd,f_nd,Ro)
		uq, Uq, uQ, UQ, vq, vQ = PV.fluxes(u,v,U0_nd,PV_prime,PV_BG,N,Nt)
		P, P_xav = PV.footprint(uq,Uq,uQ,UQ,vq,vQ,x_nd,T_nd,dx_nd,dy_nd,N,Nt)	

		EEF_array[ui,:], l_PV = PV.EEF(P_xav,y_nd,y_nd[y0_index],y0_index,dy_nd,N)

		#==

		u = u / np.sqrt(E_array[ui]); v = v / np.sqrt(E_array[ui]); h = h / np.sqrt(E_array[ui])

		# In order to calculate the vorticities of the system, we require full (i.e. BG + forced response) u and eta.
		h_full = np.zeros((N,N,Nt))
		u_full = np.zeros((N,N,Nt))
		for j in range(0,N):
			h_full[j,:,:] = h[j,:,:] + H0_nd[j]
			u_full[j,:,:] = u[j,:,:] + U0_nd[j]

		#==
	
		# Calculate PV fields and PV fluxes.
		PV_prime, PV_full, PV_BG = PV.potentialVorticity(u,v,h,u_full,h_full,H0_nd,U0_nd,N,Nt,dx_nd,dy_nd,f_nd,Ro)
		uq, Uq, uQ, UQ, vq, vQ = PV.fluxes(u,v,U0_nd,PV_prime,PV_BG,N,Nt)
		P, P_xav = PV.footprint(uq,Uq,uQ,UQ,vq,vQ,x_nd,T_nd,dx_nd,dy_nd,N,Nt)	

		EEFnorm_array[ui,:], l_PV = PV.EEF(P_xav,y_nd,y_nd[y0_index],y0_index,dy_nd,N)

	np.save('EEF_array'+str(pi),EEF_array)
	np.save('EEFnorm_array'+str(pi),EEFnorm_array)
	np.save('M_array'+str(pi),M_array)
	np.save('N_array'+str(pi),N_array)
	np.save('E_array'+str(pi),E_array)
	

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
parallelDiags.buildArray('EEFnorm_array',Nu,pe)
parallelDiags.buildArray('M_array',Nu,pe)
parallelDiags.buildArray('N_array',Nu,pe)
parallelDiags.buildArray('E_array',Nu,pe)

elapsed = time.time() - start
elapsed = np.ones(1) * elapsed
print(elapsed)

	
	
	
