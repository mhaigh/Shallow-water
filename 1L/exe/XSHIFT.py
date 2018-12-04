# RSW.py
#=======================================================

# This code solves the single-layer shallow water equations (centered-in-space finite difference), with external forcing terms on each of the three equations.
# The equations are solved in a beta-plane zonally periodic channel, with no-normal flow BCs at the northern and southern boundaries.
# The model includes simple linear Rayleigh drag on the ocean's bottom and viscosity.
# Also included is a latitude-dependent zonal BG flow and corresponding geostrophic BG sea-surface height, around which the equations are linearised.
# The original governing equations are simplified by implementing the zonal Fourier transform and assuming time-periodicity.

# This means solving a system of the form:
# a1 * u + a2 * u_yy + a3 * v + a4 * eta = Ro * F1,
# b1 * u + b2 * v + b3 * v_yy + b4 * eta_y = Ro * F2,
# c1 * u + c2 * v + c3 * v_y + c4 * eta = F3,
# where (u,v) is the horizontal velocity vector, eta is the interface height, Fi are the forcings with correpsonding amplitude alphai
# and ai, bi, ci are k- and y-dependent coefficients.

#====================================================

import sys

import numpy as np

import diagnostics
import PV
import momentum
import forcing_1L
import solver
import output
import energy
import plotting

from inputFile import *

# 1L SW Solver
#====================================================
#====================================================


# Forcing
if FORCE_TYPE == 'CTS':
	F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing_1L.forcing_cts(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE,AmpF_nd,f_nd,f0_nd,dx_nd,dy_nd);
elif FORCE_TYPE == 'DCTS':
	F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing_1L.forcing_dcts(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE,AmpF_nd,f_nd,f0_nd,dx_nd,dy_nd);
elif FORCE_TYPE == 'DELTA':
	F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing_1L.forcing_delta(AmpF_nd,y0_index,dx_nd,N);
else:
	sys.exit('ERROR: Invalid forcing option selected.');
	#plotting.forcingPlot_save(x_grid,y_grid,F3_nd[:,0:N],FORCE,BG,Fpos,N);


nn = 51;
U0_set = np.linspace(-0.5,0.5,nn);
xshift = np.zeros(nn);
for ii in range(0,nn):
	print(ii);
	# Redefine U0 and H0 in each run.
	for j in range(0,N):
		U0[j] = U0_set[ii];
		H0[j] = - (U0[j] / g) * (f0 * y[j] + beta * y[j]**2 / 2) + Hflat;
	U0_nd = U0 / U;
	H0_nd = H0 / chi; 

	# Coefficients
	a1,a2,a3,a4,b4,c1,c2,c3,c4 = solver.SOLVER_COEFFICIENTS(Ro,Re,K_nd,f_nd,U0_nd,H0_nd,omega_nd,gamma_nd,dy_nd,N);
	# Solver
	if BC == 'NO-SLIP':
		solution = solver.NO_SLIP_SOLVER(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ro*Ftilde1_nd,Ro*Ftilde2_nd,Ftilde3_nd,N,N2);
	if BC == 'FREE-SLIP':
		#solution = solver.FREE_SLIP_SOLVER(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ro*Ftilde1_nd,Ro*Ftilde2_nd,Ftilde3_nd,N,N2);
		solution = solver.FREE_SLIP_SOLVER4(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ro*Ftilde1_nd,Ro*Ftilde2_nd,Ro*Ftilde3_nd,N,N2)

	utilde_nd, vtilde_nd, etatilde_nd = solver.extractSols(solution,N,N2,BC);
	u, v, h = solver.SPEC_TO_PHYS(utilde_nd,vtilde_nd,etatilde_nd,T_nd,dx_nd,omega_nd,N);

	u = np.real(u);
	v = np.real(v);
	h = np.real(h);
	
	# Normalise all solutions by the (non-dimensional) forcing amplitude. 
	u = u / AmpF_nd;
	v = v / AmpF_nd;
	h = h / AmpF_nd;

	# In order to calculate the vorticities/energies of the system, we require full (i.e. BG + forced response) u and eta
	h_full = np.zeros((N,N,Nt));
	u_full = np.zeros((N,N,Nt));
	for j in range(0,N):
		h_full[j,:,:] = h[j,:,:] + H0_nd[j];
		u_full[j,:,:] = u[j,:,:] + U0_nd[j];

	#np.save('u.npy',u);
	#np.save('v.npy',v);
	#np.save('h.npy',h);

	#plt.subplot(121);
	#plt.pcolor(x_grid,y_grid,u_full[:,:,ts],cmap='bwr');
	#plt.colorbar();
	#plt.subplot(122);
	#plt.pcolor(x_grid,y_grid,h_full[:,:,ts],cmap='bwr');
	#plt.colorbar();
	#plt.show();

	#sys.exit();


	# PV and PV footprints
	#====================================================

	# Calculate PV fields, footprints and equivalent eddy fluxes (EEFs)
	if doPV:
		PV_prime, PV_full, PV_BG = PV.potentialVorticity(u,v,h,u_full,h_full,H0_nd,U0_nd,N,Nt,dx_nd,dy_nd,f_nd,Ro);
		uq, Uq, uQ, UQ, vq, vQ = PV.fluxes(u,v,U0_nd,PV_prime,PV_BG,N,Nt);
		# Keep these next two lines commented out unless testing effects of normalisation.
		# uq, Uq, uQ, UQ, vq, vQ = uq/AmpF_nd**2, Uq/AmpF_nd**2, uQ/AmpF_nd**2, UQ/AmpF_nd**2, vq/AmpF_nd**2, vQ/AmpF_nd**2;
		# PV_prime, PV_full = PV_prime/AmpF_nd, PV_full/AmpF_nd;

		if doFootprints:
			if footprintComponents: 
				P, P_uq, P_uQ, P_Uq, P_vq, P_vQ, P_xav, P_uq_xav, P_uQ_xav, P_Uq_xav, P_vq_xav, P_vQ_xav = PV.footprintComponents(uq,Uq,uQ,vq,vQ,x_nd,T_nd,dx_nd,dy_nd,N,Nt);
				#plotting.footprintComponentsPlot(uq,Uq,uQ,vq,vQ,P,P_uq,P_Uq,P_uQ,P_vq,P_vQ,P_xav,P_uq_xav,P_uQ_xav,P_Uq_xav,P_vq_xav,P_vQ_xav,x_nd,y_nd,N,Nt);
				#plotting.plotPrimaryComponents(P_uq,P_vq,P_uq_xav,P_vq_xav,x_nd,y_nd,FORCE,BG,Fpos,N);
			else: 
				P, P_xav = PV.footprint(uq,Uq,uQ,UQ,vq,vQ,x_nd,T_nd,dx_nd,dy_nd,N,Nt);	

	xshift[ii] = PV.footprint_shift(P,y_nd,dy_nd,x_nd,dx_nd,N);		
			


plt.plot(xshift);
plt.show();

np.save('xshift.npy',xshift);


