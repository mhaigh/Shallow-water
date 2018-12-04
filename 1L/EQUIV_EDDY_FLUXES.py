# EQUIV_EDDY_FLUXES.py
#=======================================================
# This is an executable code that solves the 1L shallow water system a number of times, each time storing the equivalent eddy flux.
#=======================================================

import os
import numpy as np
import time

from core import solver, PV, momentum, diagnostics

from inputFile import *

#=======================================================

start = time.time();

filename = 'EEF_PV';
filename_u = 'EEF_u';
filename_v = 'EEF_v';
filename_eta = 'EEF_eta';

# Can test against U0 or y0, or find the buoyancy vs U0 or y0.
TEST = 'y0';

#=======================================================

# Initialise tests

if TEST == 'U0':
	nn = 321;
	U0_set = np.linspace(-0.3,0.5,nn);
	if FORCE_TYPE == 'CTS':
		F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing.forcing_cts(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE,AmpF_nd,f_nd,f0_nd,dx_nd,dy_nd)
	elif FORCE_TYPE == 'CTS2':
		F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing.forcing_cts2(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE,AmpF_nd,f_nd,f0_nd,bh,dx_nd,dy_nd)
	elif FORCE_TYPE == 'DCTS':
		F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing.forcing_dcts(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE,AmpF_nd,f_nd,f0_nd,dx_nd,dy_nd);
	else:
		sys.exit('ERROR: Invalid forcing option selected.');

if TEST == 'y0':
	y0_min = y[0] + r0;					# We want to keep the forcing at least one gridpoint away from the boundary
	y0_max = y[N-1] - r0;
	y0_set = [];						# Initialise an empty set of forcing latitudes
	y0_index_set = [];
	for j in range(0,N):
		if y0_min <= y[j] <= y0_max:
			y0_set.append(y[j]);		# Build the set of forcing locations, all at least 1 gridpoint away from the boundary.	
			y0_index_set.append(j);
	y0_set = np.array(y0_set);
	y0_index_set = np.array(y0_index_set);
	nn = np.shape(y0_set)[0];
	#print(nn)
	a1,a2,a3,a4,b4,c1,c2,c3,c4 = solver.SOLVER_COEFFICIENTS(Ro,Re,K_nd,f_nd,U0_nd,H0_nd,omega_nd,gamma_nd,dy_nd,N);
	

# Initialise the array which stores the EEF values.
if os.path.isfile(filename + '.npy'):
	EEF_PV = np.load(filename + '.npy');
else:
	EEF_PV = np.zeros((nn,2));
	l_PV = np.zeros((nn,2));
	P_xav = np.zeros((nn,N));
	P1 = np.zeros((nn,N));
	P2 = np.zeros((nn,N));

#EEF_u = np.zeros((nn,2));
#EEF_v = np.zeros((nn,2));
#EEF_eta = np.zeros((nn,2));

# Correlation
corr_vy_uy = np.zeros((nn,Nt));
corr_v_uyy = np.zeros((nn,Nt));
cs = N / 4; 
ce = N - N / 4;
	
# Amplitude
uAmp = np.zeros((nn,Nt));
vAmp = np.zeros((nn,Nt));

# EEF
EEF = np.zeros((nn,2));
fm = np.zeros((nn));
l = np.zeros((nn,2));

#=======================================================

# Now start the loop over each forcing index.
for ii in range(0,nn):
	
	if EEF_PV[ii,0] == 0:

		# If TEST==U0, linear problem has to be redefined each iteration.
		if TEST == 'U0':
			# Redefine U0 and H0 in each run.
			for j in range(0,N):
				U0[j] = U0_set[ii];
				H0[j] = - (U0[j] / g) * (f0 * y[j] + beta * y[j]**2 / 2) + Hflat;
			U0_nd = U0 / U;
			H0_nd = H0 / chi; 

			a1,a2,a3,a4,b4,c1,c2,c3,c4 = solver.SOLVER_COEFFICIENTS(Ro,Re,K_nd,f_nd,U0_nd,H0_nd,omega_nd,gamma_nd,dy_nd,N);
				
		# If TEST==y0, matrix only needs to be defined once, but forcing must be defined each iteration.
		if TEST == 'y0':
			y0 = y0_set[ii];				# Redefine y0 and the forcing in each run.
			y0_index = y0_index_set[ii];
			y0_nd = y0 / L;
			# Forcing
			if FORCE_TYPE == 'CTS':
				F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing_1L.forcing_cts(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE,AmpF_nd,f_nd,f0_nd,dx_nd,dy_nd)
			elif FORCE_TYPE == 'CTS2':
				F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing.forcing_cts2(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE,AmpF_nd,f_nd,f0_nd,bh,dx_nd,dy_nd)
			elif FORCE_TYPE == 'DCTS':
				F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing_1L.forcing_dcts(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE,AmpF_nd,f_nd,f0_nd,dx_nd,dy_nd);
			else:
				sys.exit('ERROR: Invalid forcing option selected.');	
	
		# Solver
		if BC == 'NO-SLIP':
			solution = solver.NO_SLIP_SOLVER(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ro*Ftilde1_nd,Ro*Ftilde2_nd,Ro*Ftilde3_nd,N,N2)
		if BC == 'FREE-SLIP':
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

		#print(np.max(u))

		# In order to calculate the vorticities of the system, we require full (i.e. BG + forced response) u and eta.
		h_full = np.zeros((N,N,Nt));
		u_full = np.zeros((N,N,Nt));
		for j in range(0,N):
			h_full[j,:,:] = h[j,:,:] + H0_nd[j];
			u_full[j,:,:] = u[j,:,:] + U0_nd[j];
	
		# Calculate PV fields and PV fluxes.
		PV_prime, PV_full, PV_BG = PV.potentialVorticity(u,v,h,u_full,h_full,H0_nd,U0_nd,N,Nt,dx_nd,dy_nd,f_nd,Ro)
		uq, Uq, uQ, UQ, vq, vQ = PV.fluxes(u,v,U0_nd,PV_prime,PV_BG,N,Nt)
		P, P_xav[ii,:] = PV.footprint(uq,Uq,uQ,UQ,vq,vQ,x_nd,T_nd,dx_nd,dy_nd,N,Nt)			
		EEF_PV[ii,:], l_PV[ii,:] = PV.EEF(P_xav[ii,:],y_nd,y0_nd,y0_index,dy_nd,N)
		fm[ii] = PV.firstMoment(P_xav[ii,:],y_nd,y0_nd,dy_nd,N)
		print(fm)
		#===

		# These next few lines compute dominant components of the footprint zonal average.
		# Comment out if not required.

		# Take relevant derivatives
		#v_y = np.zeros((N,N,Nt));
		#u_y = np.zeros((N,N,Nt));
		#u_yy = np.zeros((N,N,Nt));
		#for ti in range(0,Nt):
		#	v_y[:,:,ti] = diagnostics.diff(v[:,:,ti],0,0,dy_nd);
		#	u_y[:,:,ti] = diagnostics.diff(u[:,:,ti],0,0,dy_nd);
		#	u_yy[:,:,ti] = diagnostics.diff(u_y[:,:,ti],0,0,dy_nd);
			# Correlation at each time step
		#	corr_vy_uy[ii,ti] = diagnostics.arrayCorr(u_y[cs:ce,cs:ce,ti],v_y[cs:ce,cs:ce,ti]);
		#	corr_v_uyy[ii,ti] = diagnostics.arrayCorr(u_yy[cs:ce,cs:ce,ti],v[cs:ce,cs:ce,ti]);
			# Amplitude
		#	uAmp[ii,ti] = max(u[cs:ce,cs:ce,ti].max(),u[cs:ce,cs:ce,ti].min(),key=abs);
		#	vAmp[ii,ti] = max(v[cs:ce,cs:ce,ti].max(),v[cs:ce,cs:ce,ti].min(),key=abs);
	
		# Define initial footprint contributions (include SSH terms later)
		#P1_tmp = diagnostics.timeAverage(v_y*u_y,T_nd,Nt);
		#P2_tmp = diagnostics.timeAverage(v*u_yy,T_nd,Nt);

		#P1_tmp = diagnostics.extend(P1_tmp);
		#P2_tmp = diagnostics.extend(P2_tmp);

		#P1[ii,:] = np.trapz(P1_tmp,x_nd,dx_nd,axis=1) / H0_nd;
		#P2[ii,:] = np.trapz(P2_tmp,x_nd,dx_nd,axis=1) / H0_nd;
	
		#===

		# Buoyancy EEF
		#uh, uH, Uh, UH, vh, vH = thickness.fluxes(u,v,h,U0_nd,H0_nd,N,Nt);
		#B, B_xav = thickness.footprint(uh,uH,Uh,vh,vH,x_nd,y_nd,T_nd,dx_nd,dy_nd,dt_nd,N,Nt);			
		#EEF_eta[ii,:] = thickness.EEF(B_xav,y_nd,y0_nd,y0_index,dy_nd,N);

		# Calculate momentum fluxes and footprints
		#uu, uv, vv = momentum.fluxes(u,v);
		#Mu, Mv, Mu_xav, Mv_xav = momentum.footprint(uu,uv,vv,x_nd,T_nd,dx_nd,dy_nd,N,Nt);
		#EEF_u[ii,:], EEF_v[ii,:] = momentum.EEF_mom(Mu_xav,Mv_xav,y_nd,y0_nd,y0_index,dy_nd,omega_nd,N);
		
plt.plot(fm); plt.show()
#np.save(filename,EEF_PV);
np.save('EEF_l',l_PV);
np.save('P_xav',P_xav);
#np.save('P1',P1);
#np.save('P2',P2);
#np.save(filename_u,EEF_u);
#np.save(filename_v,EEF_v);
#np.save(filename_eta,EEF_eta);

#np.save('output/corr_vy_uy',corr_vy_uy);
#np.save('output/corr_v_uyy',corr_v_uyy);
#np.save('output/uAmp',uAmp);
#np.save('output/vAmp',vAmp);

#np.save('output/EEF_vy_uy',EEF_vy_uy);
#np.save('output/EEF_v_uyy',EEF_v_uyy);
#np.save('output/l_vy_uy',l_vy_uy);
#np.save('output/l_v_uyy',l_v_uyy);
	
elapsed = time.time() - start;
elapsed = np.ones(1) * elapsed;
print(elapsed);

	
	
	
