# This function shows that the zonal-mean footprint calculated from
# summing contributions from each wavenumber separately is equal to
# calculating the footprint from the full solution in the regular
# way. In other words, modes of different zonal wavenumber do not 
# interact with one another, due to Fourier mode orthogonality.

import time

import sys

import numpy as np
import matplotlib.pyplot as plt

from core import solver, PV, momentum, thickness, energy, diagnostics, corr
from output import plotting, plotting_bulk

from inputFile import *

# 1L SW Solver
#====================================================
#====================================================

start = time.time()

def RSW_main():
	# Forcing

	#plotting.forcingPlot_save(x_grid,y_grid,F3_nd[:,0:N],FORCE,BG,Fpos,N);

	#F1_nd, F2_nd, F3_nd = forcing.forcingInv(Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,x_nd,y_nd,dx_nd,N);
	#F1_nd, F2_nd = forcing.F12_from_F3(F3_nd,f_nd,dx_nd,dy_nd,N,N);
	#plotting.forcingPlots(x_nd[0:N],y_nd,Ro*F1_nd,Ro*F2_nd,F3_nd,Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,N);

#	sys.exit();
	
	# Coefficients
	a1,a2,a3,a4,b4,c1,c2,c3,c4 = solver.SOLVER_COEFFICIENTS(Ro,Re,K_nd,f_nd,U0_nd,H0_nd,omega_nd,gamma_nd,dy_nd,N)
	# Solver
	if BC == 'NO-SLIP':
		solution = solver.NO_SLIP_SOLVER(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ro*Ftilde1_nd,Ro*Ftilde2_nd,Ftilde3_nd,N,N2)
	if BC == 'FREE-SLIP':
		#solution = solver.FREE_SLIP_SOLVER(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ro*Ftilde1_nd,Ro*Ftilde2_nd,Ftilde3_nd,N,N2)
		solution = solver.FREE_SLIP_SOLVER4(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ro*Ftilde1_nd,Ro*Ftilde2_nd,Ro*Ftilde3_nd,N,N2)

	utilde_nd, vtilde_nd, etatilde_nd = solver.extractSols(solution,N,N2,BC);
	u, v, h = solver.SPEC_TO_PHYS(utilde_nd,vtilde_nd,etatilde_nd,T_nd,dx_nd,omega_nd,N);


	PP = np.zeros(N)
	EEF = 0.

	for iji in range(0,N):
		print(iji)

		ut = np.zeros((N,N),dtype=complex)
		vt = np.zeros((N,N),dtype=complex)
		ht = np.zeros((N,N),dtype=complex)
		
		ut[:,iji] = utilde_nd[:,iji]
		vt[:,iji] = vtilde_nd[:,iji]
		ht[:,iji] = etatilde_nd[:,iji]

		u, v, h = solver.SPEC_TO_PHYS(ut,vt,ht,T_nd,dx_nd,omega_nd,N);

		u = np.real(u)
		v = np.real(v)
		h = np.real(h)
	
		# Normalise all solutions by the (non-dimensional) forcing amplitude. 
		u = u / AmpF_nd
		v = v / AmpF_nd
		h = h / AmpF_nd

		# In order to calculate the vorticities/energies of the system, we require full (i.e. BG + forced response) u and eta.
		h_full = np.zeros((N,N,Nt))
		u_full = np.zeros((N,N,Nt))
		for j in range(0,N):
			h_full[j,:,:] = h[j,:,:] + H0_nd[j]
			u_full[j,:,:] = u[j,:,:] + U0_nd[j]

		PV_prime, PV_full, PV_BG = PV.potentialVorticity(u,v,h,u_full,h_full,H0_nd,U0_nd,N,Nt,dx_nd,dy_nd,f_nd,Ro)
		uq, Uq, uQ, UQ, vq, vQ = PV.fluxes(u,v,U0_nd,PV_prime,PV_BG,N,Nt)
		P, P_xav = PV.footprint(uq,Uq,uQ,UQ,vq,vQ,x_nd,T_nd,dx_nd,dy_nd,N,Nt)

		PP += P_xav

		from scipy.ndimage.measurements import center_of_mass
		iii = center_of_mass(np.abs(P_xav))[0]
		com = y_nd[int(iii)]
		EEF, l = PV.EEF(P_xav,y_nd,com,int(iii),dy_nd,N)
		EEF_north = EEF[0]; EEF_south = EEF[1];
		EEF_tmp = EEF_north - EEF_south;

		EEF += EEF_tmp

	plt.plot(PP)
	plt.show()
	print(EEF)

if __name__ == '__main__':
	RSW_main()






















	
		
