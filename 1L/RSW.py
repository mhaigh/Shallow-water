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
	#F12, F22 = forcing.F12_from_F3(F3_nd,f_nd,dx_nd,dy_nd,N,N);
	#plotting.forcingPlots(x_nd[0:N],y_nd,Ro*F1_nd,Ro*F2_nd,F3_nd,Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,N);
	
	# Coefficients
	a1,a2,a3,a4,b4,c1,c2,c3,c4 = solver.SOLVER_COEFFICIENTS(Ro,Re,K_nd,f_nd,U0_nd,H0_nd,omega_nd,gamma_nd,dy_nd,N)
	# Solver
	if BC == 'NO-SLIP':
		solution = solver.NO_SLIP_SOLVER(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ro*Ftilde1_nd,Ro*Ftilde2_nd,Ftilde3_nd,N,N2)
	if BC == 'FREE-SLIP':
		#solution = solver.FREE_SLIP_SOLVER(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ro*Ftilde1_nd,Ro*Ftilde2_nd,Ftilde3_nd,N,N2)
		solution = solver.FREE_SLIP_SOLVER4(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ro*Ftilde1_nd,Ro*Ftilde2_nd,Ro*Ftilde3_nd,N,N2)

	utilde_nd, vtilde_nd, etatilde_nd = solver.extractSols(solution,N,N2,BC)

	#utilde_nd, vtilde_nd, etatilde_nd = diagnostics.selectModes(utilde_nd,vtilde_nd,etatilde_nd,6,False,N)
	u, v, h = solver.SPEC_TO_PHYS(utilde_nd,vtilde_nd,etatilde_nd,T_nd,dx_nd,omega_nd,N)

	plt.contourf(np.angle(h[:,:,ts])); plt.colorbar(); plt.show()

	u = np.real(u)
	v = np.real(v)
	h = np.real(h)

	#diagnostics.twoMode(utilde_nd,vtilde_nd,x_nd,N)
	#KEspec = energy.KEspectrum(u,v,K_nd,y_nd,T_nd,Nt,N)
	#sys.exit()
	
	# Normalise all solutions by the (non-dimensional) forcing amplitude. 
	u = u / AmpF_nd
	v = v / AmpF_nd
	h = h / AmpF_nd

	#np.save('u.npy',u)
	#np.save('v.npy',v)
	#np.save('h.npy',h)
	
	#sys.exit()



	# In order to calculate the vorticities/energies of the system, we require full (i.e. BG + forced response) u and eta.
	h_full = np.zeros((N,N,Nt))
	u_full = np.zeros((N,N,Nt))
	for j in range(0,N):
		h_full[j,:,:] = h[j,:,:] + H0_nd[j]
		u_full[j,:,:] = u[j,:,:] + U0_nd[j]

	#====================================================

	# Energy

	if doEnergy:
		KE_BG, KE_BG_tot, PE_BG, PE_BG_tot = energy.energy_BG(U0_nd,H0_nd,Ro,y_nd,dy_nd,N)
		KE, KE_tot = energy.KE(u_full,v,h_full,x_nd,y_nd,dx_nd,dy_nd,N)
		PE, PE_tot = energy.PE(h_full,Ro,x_nd,y_nd,dx_nd,dy_nd,N)

		E = KE + PE
		#E_tot = KE_tot + PE_tot

		Ef, Ef_av = energy.budgetForcing(u_full,v,h_full,F1_nd,F2_nd,F3_nd,Ro,N,T_nd,omega_nd,Nt)
		Ed, Ed_av = energy.budgetForcing2(U0_nd,H0_nd,u,v,h,F1_nd,F2_nd,F3_nd,Ro,N,T_nd,omega_nd,Nt)
		#Ed, Ed_av = energy.budgetDissipation(u_full,v,h_full,Ro,Re,gamma_nd,dx_nd,dy_nd,T_nd,Nt)
		Ed, Ed_av = energy.budgetDissipation2(U0_nd,H0_nd,u,v,h,Ro,Re,gamma_nd,dx_nd,dy_nd,T_nd,Nt,N)
		Eflux, Eflux_av = energy.budgetFlux(u_full,v,h_full,Ro,dx_nd,dy_nd,T_nd,Nt)


		plt.subplot(221)
		plt.contourf(Ef_av); plt.colorbar()

		plt.subplot(222)
		plt.contourf(Ed_av); plt.colorbar()

		plt.subplot(223)
		plt.contourf(Eflux_av); plt.colorbar()

		plt.subplot(224)
		plt.contourf(-Ed_av-Ef_av); plt.colorbar();
		plt.show()


		#uE, vE = energy.flux(KE,u,v)
		#Econv, Econv_xav = energy.conv(uE,vE,T_nd,Nt,x_nd,dx_nd,y_nd,dy_nd)
		#vE_av = diagnostics.timeAverage(vE,T_nd,Nt)
		#plt.contourf(vE_av); plt.colorbar(); plt.show()

		#plt.subplot(121); plt.contourf(Econv); plt.colorbar(); 
		#plt.subplot(122); plt.plot(Econv_xav); plt.show()
		
		#quit()

	#====================================================

	if doCorr:
			
		M = corr.M(u,v,T_nd)
		N_ = corr.N(u,v,T_nd)
		K = corr.K(u,v,T_nd)
		D = corr.D(u,v,1,dx_nd,dy_nd)
		Curl_uD = corr.Curl_uD(u,v,D,T,dx_nd,dy_nd)
		theta = corr.orientation(M,N_)

		Mnorm = corr.Mnorm(u,v,T_nd)
		Nnorm = corr.Nnorm(u,v,T_nd)

		Dv, Du = corr.Curl_uD_components(u,v,D,T,dx_nd,dy_nd)

		#corr.plotComponents(x_nd,y_nd,M,N_,K,Du)

		plotting_bulk.plotKMN(K,Mnorm,Nnorm,x_grid,y_grid,N,0,2,'')
		plt.show()
		quit()
		#N_ /= diagnostics.domainInt(K,x_nd,dx_nd,y_nd,dy_nd)

		N_av = np.trapz(diagnostics.extend(N_),x_nd,dx_nd,axis=1)
		Nyy = diagnostics.diff(diagnostics.diff(N_,0,0,dy_nd),0,0,dy_nd)
		Nyy_av = np.trapz(diagnostics.extend(Nyy),x_nd,dx_nd,axis=1)
		Curl_uD_av = np.trapz(diagnostics.extend(Curl_uD),x_nd,dx_nd,axis=1)

		#np.save('M',M); np.save('N',N_); np.save('Nyy',Nyy)
		
		lim = np.max(np.abs(N_)) / 2.
		plt.figure(figsize=[12,6])
		plt.subplot(121)
		plt.pcolor(x_grid,y_grid,N_,cmap='bwr',vmin=-lim,vmax=lim)
		plt.xlim(-0.1,0.1)
		plt.ylim(-.1,0.1)
		plt.text(-0.08,0.08,'N',fontsize=18)
		plt.xlabel('x',fontsize=18)
		plt.ylabel('y',fontsize=18)
		plt.grid()
		plt.colorbar()

		plt.subplot(122)
		plt.plot(N_av,y_nd)
		plt.ylim(-.1,.1)
		plt.xlabel('<N>',fontsize=18)
		plt.grid()

		plt.tight_layout()
		plt.show()

		#plt.figure(figsize=[12,6])
		#plt.subplot(121)
		#plt.contourf(x_nd[0:N],y_nd,Nyy)
		#plt.xlim(-0.1,0.1)
		#plt.ylim(-0.1,0.1)
		#plt.colorbar()
		#plt.subplot(122)
		#plt.plot(Nyy_av,y_nd)
		#plt.show()

		#corr.plotOrientation(theta,K,x_nd,y_nd)
		#uav = np.trapz(diagnostics.extend(Du),x_nd,dx_nd,axis=1)
		#vav = np.trapz(diagnostics.extend(Dv),x_nd,dx_nd,axis=1)

		#plt.plot(uav,label='u')
		#plt.plot(vav,label='v')
		#plt.plot(Curl_uD_av,label='full')	
		#plt.legend()
		#plt.show()

		# Correlation.
		# Central half?
		#cs = N / 4; 
		#ce = N - N / 4;
		#corr = corr.arrayCorrTime(u[cs:ce,cs:ce,:],v[cs:ce,cs:ce,:]);
		#print corr


		#quit()

	#====================================================
	
	# Error - if calculated, should be done before real part of solution is taken
	if errorPhys:
		e1, e2, e3 = diagnostics.error(u,v,h,dx_nd,dy_nd,dt_nd,U0_nd,H0_nd,Ro,gamma_nd,Re,f_nd,F1_nd,F2_nd,F3_nd,T_nd,ts,omega_nd,N)
		e = np.sqrt((e1**2 + e2**2 + e3**2) / 3.0)
		print 'Error = ' + str(e) + '. Error split = ' + str(e1) + ', ' + str(e2) + ', ' + str(e3)
	if errorSpec:
		error_spec = np.zeros((3,N));	# An array to save the spectral error at each wavenumber for each equation.
		for i in range(0,N):
			error_spec[:,i] = diagnostics.specError(utilde_nd[:,i],vtilde_nd[:,i],etatilde_nd[:,i],Ftilde1_nd[:,i],Ftilde2_nd[:,i],Ftilde3_nd[:,i],a1[:,i],a2,a3,a4[i],\
	b4,c1[:,i],c2,c3,c4[:,i],f_nd,Ro,K_nd[i],H0_nd,y_nd,dy_nd,N)
		for eq in range(0,3):
			error = sum(error_spec[eq,:]) / N
			print('Error' + str(int(eq+1)) + '=' + str(error))

	#====================================================

	# Momentum footprints
	#====================================================
	
	if doMomentum:
		uu, uv, vv = momentum.fluxes(u,v);
		Mu, Mv, Mu_xav, Mv_xav = momentum.footprint(uu,uv,vv,x_nd,T_nd,dx_nd,dy_nd,N,Nt);
		#plotting.MomFootprints(Mu,Mv,Mu_xav,Mv_xav);
		
		Mumax = np.max(Mu_xav)
	
		plt.plot(Mu_xav/Mumax,y_nd,linewidth=2.,color='k')
		plt.text(-0.4,0.4,str(Mumax))
		plt.xlabel('Zonal mom. flux convergence',fontsize=18)
		plt.ylabel('y',fontsize=18)
		plt.ylim([-.5,.5])
		plt.yticks((-1./2,-1./4,0,1./4,1./2));
		plt.grid()
		plt.show()
	
		if False:

			plt.subplot(121);
			plt.pcolor(x_grid,y_grid,Mu,cmap='bwr', vmin=-.5, vmax=.5);
			plt.xticks((-1./2,-1./4,0,1./4,1./2));
			plt.yticks((-1./2,-1./4,0,1./4,1./2));	
			plt.xlabel('x',fontsize=16);
			plt.ylabel('y',fontsize=16);
			plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
			plt.grid(b=True, which='both', color='0.65',linestyle='--');
			plt.colorbar()

			plt.subplot(122);
			plt.plot(Mu_xav,y_nd);
			plt.xlabel('y')
			plt.show()

			plt.subplot(121);
			plt.pcolor(x_grid,y_grid,Mv,cmap='bwr', vmin=-1., vmax=1.);
			plt.xticks((-1./2,-1./4,0,1./4,1./2));
			plt.yticks((-1./2,-1./4,0,1./4,1./2));	
			plt.xlabel('x',fontsize=16);
			plt.ylabel('y',fontsize=16);
			plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
			plt.grid(b=True, which='both', color='0.65',linestyle='--');

			plt.subplot(122);
			plt.plot(Mv_xav,y_nd);

			plt.tight_layout();
			plt.show();
	
		EEF_u, EEF_v = momentum.EEF_mom(Mu_xav,Mv_xav,y_nd,y0_nd,y0_index,dy_nd,omega_nd,N);
		
		#print(EEF_u, EEF_v);

	# PV and PV footprints
	#====================================================

	# Calculate PV fields, footprints and equivalent eddy fluxes (EEFs)
	if doPV:
		PV_prime, PV_full, PV_BG = PV.potentialVorticity(u,v,h,u_full,h_full,H0_nd,U0_nd,N,Nt,dx_nd,dy_nd,f_nd,Ro)
		#PV_prime1, PV_prime2, PV_prime3 = PV.potentialVorticity_linear(u,v,h,H0_nd,U0_nd,N,Nt,dx_nd,dy_nd,f_nd,Ro)
		uq, Uq, uQ, UQ, vq, vQ = PV.fluxes(u,v,U0_nd,PV_prime,PV_BG,N,Nt)
		# Keep these next two lines commented out unless testing effects of normalisation.
		# uq, Uq, uQ, UQ, vq, vQ = uq/AmpF_nd**2, Uq/AmpF_nd**2, uQ/AmpF_nd**2, UQ/AmpF_nd**2, vq/AmpF_nd**2, vQ/AmpF_nd**2
		# PV_prime, PV_full = PV_prime/AmpF_nd, PV_full/AmpF_nd

		if doFootprints:
			if footprintComponents: 
				P, P_uq, P_uQ, P_Uq, P_vq, P_vQ, P_xav, P_uq_xav, P_uQ_xav, P_Uq_xav, P_vq_xav, P_vQ_xav = PV.footprintComponents(uq,Uq,uQ,vq,vQ,x_nd,T_nd,dx_nd,dy_nd,N,Nt);
				#plotting.footprintComponentsPlot(uq,Uq,uQ,vq,vQ,P,P_uq,P_Uq,P_uQ,P_vq,P_vQ,P_xav,P_uq_xav,P_uQ_xav,P_Uq_xav,P_vq_xav,P_vQ_xav,x_nd,y_nd,N,Nt);
				#plotting.plotPrimaryComponents(P_uq,P_vq,P_uq_xav,P_vq_xav,x_nd,y_nd,FORCE,BG,Fpos,N);
			else: 
				P, P_xav = PV.footprint(uq,Uq,uQ,UQ,vq,vQ,x_nd,T_nd,dx_nd,dy_nd,N,Nt);	

			if doEEFs:

				from scipy.ndimage.measurements import center_of_mass
				iii = center_of_mass(np.abs(P_xav))[0]
				i1 = int(iii); i2 = int(i1 + 1); r = iii - i1
				#print(iii,i1,i2,r)

				com = y_nd[int(iii)]
				#print(y0_index-iii)
				if footprintComponents:
					EEF_array = PV.EEF_components(P_xav,P_uq_xav,P_uQ_xav,P_Uq_xav,P_vq_xav,P_vQ_xav,y_nd,y0_nd,y0_index,dy_nd,omega_nd,N);
					# This returns EEF_array, an array with the following structure:
					# EEF_array = ([EEF_north,EEF_south],[uq_north,uq_south],[Uq_north,Uq_south],[uQ_north,uQ_south],[vq_north,vq_south],[vQ_north,vQ_south]).
					EEF_north = EEF_array[0,0]; EEF_south = EEF_array[0,1];
				else:
					EEF, l = PV.EEF(P_xav,y_nd,com,int(iii),dy_nd,N)
					# These lines for Gaussian EEFs, when com jumps from 1 grid point to next, need to smooth EEF.
					#EEF1, l = PV.EEF(P_xav,y_nd,y_nd[i1],i1,dy_nd,N)
					#EEF2, l = PV.EEF(P_xav,y_nd,y_nd[i2],i2,dy_nd,N)
					#EEF_ = (1 - r) * EEF1 + r * EEF2
					EEF_north = EEF[0]; EEF_south = EEF[1];
					EEF = EEF_north - EEF_south;
					print(EEF)
	

		
		Pmax = np.max(abs(P_xav));

		plt.plot(P_xav/Pmax,y_nd,linewidth=2.,color='k')
		plt.text(-0.4,0.4,str(Pmax))
		plt.xlabel('PV flux convergence',fontsize=18)
		plt.ylabel('y',fontsize=18)
		plt.ylim([-.5,.5])
		plt.yticks((-1./2,-1./4,0,1./4,1./2));
		plt.grid()
		plt.show()

	# Buoyancy footprints
	#====================================================
	
	if doThickness:
		# Should these be zero, according to conservation of mass?
		Pb, Pb_xav = thickness.footprint(u_full,v,h_full,x_nd,y_nd,T_nd,dx_nd,dy_nd,dt_nd,N,Nt);

	#output.ncSave(utilde_nd,vtilde_nd,etatilde_nd,u,v,h,x_nd,y_nd,K_nd,T_nd,PV_full,PV_prime,PV_BG,Pq,EEFq,N,Nt);

	#sys.exit()
	
	#====================================================

	if False:

		# Take relevant derivatives
		v_y = np.zeros((N,N,Nt));
		u_y = np.zeros((N,N,Nt));
		u_yy = np.zeros((N,N,Nt));
		for ti in range(0,Nt):
			v_y[:,:,ti] = diagnostics.diff(v[:,:,ti],0,0,dy_nd);
			u_y[:,:,ti] = diagnostics.diff(u[:,:,ti],0,0,dy_nd);
			u_yy[:,:,ti] = diagnostics.diff(u_y[:,:,ti],0,0,dy_nd);
	
		uv1 = v_y * u_y;
		uv2 = v * u_yy;
		uv3 = v * u_y


		plt.subplot(131);
		plt.contourf(x_nd[0:N],y_nd,v[:,:,ts]);
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.colorbar();
		plt.subplot(132);
		plt.contourf(x_nd[0:N],y_nd,u_yy[:,:,ts]);
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.colorbar();
		plt.subplot(133);
		plt.contourf(x_nd[0:N],y_nd,uv2[:,:,ts]);
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.colorbar();
		plt.show()

		plt.subplot(221);
		plt.contourf(x_nd[0:N],y_nd,uv1[:,:,20]);
		plt.colorbar();
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.subplot(222);
		plt.contourf(x_nd[0:N],y_nd,uv1[:,:,100]);
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.colorbar();
		plt.subplot(223);
		plt.contourf(x_nd[0:N],y_nd,uv2[:,:,20]);
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.colorbar();
		plt.subplot(224);
		plt.contourf(x_nd[0:N],y_nd,uv2[:,:,100]);
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.colorbar();
		plt.show();

		# Define initial footprint contributions (include SSH terms later)
		P1 = diagnostics.timeAverage(uv1,T_nd,Nt);
		P2 = diagnostics.timeAverage(uv2,T_nd,Nt);
		P3 = diagnostics.timeAverage(uv3,T_nd,Nt);
	
	
		P1 = diagnostics.extend(P1);
		P2 = diagnostics.extend(P2);
		P3 = diagnostics.extend(P3);

		plt.subplot(121);
		plt.contourf(x_nd,y_nd,P1);
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.colorbar();
		plt.subplot(122);	
		plt.contourf(x_nd,y_nd,P2);
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.colorbar();
		plt.show();

		# Account for H0_nd terms
		#H0_y = diagnostics.diff(H0_nd,2,0,dy_nd);
		#for i in range(0,N):
		#	P1[:,i] = P1[:,i] / H0_nd[:];
		#	P2[:,i] = P2[:,i] / H0_nd[:];
		#	P3[:,i] = P3[:,i] * H0_y[:] / H0_nd[:]**2;
	
		P1 = np.trapz(P1,x_nd,dx_nd,axis=1);
		P2 = np.trapz(P2,x_nd,dx_nd,axis=1);
		P3 = np.trapz(P3,x_nd,dx_nd,axis=1);

		plt.subplot(121);
		#plt.plot(P1,label='P1');
		plt.plot(P2,label='P2');
		plt.legend();
		plt.subplot(122);
		plt.plot(H0_nd*P_xav);
		plt.plot(P1+P2+P3);
		plt.show();

	# Plots
	#====================================================
	#====================================================
	

	# Call the function that plots the forcing in physical and physical-spectral space.
	if plotForcing:
		plotting.forcingPlots(x_nd,y_nd,F1_nd,F2_nd,F3_nd,Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,N);
		#forcing_1L.forcingInv(Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,x_nd,y_nd,dx_nd,N); # For diagnostic purposes
	
	# Background state plots (inc. BG SSH, BG flow, BG PV)
	if plotBG:
		plotting.bgPlots(y_nd,H0_nd,U0_nd,PV_BG);
	
	# Soltuion Plots
	if plotSol:
		plotting.solutionPlots(x_nd,y_nd,u,v,h,ts,FORCE,BG,Fpos,N,x_grid,y_grid,True);
		#plotting.solutionPlots_save(x_nd,y_nd,u,v,h,ts,FORCE,BG,Fpos,N,x_grid,y_grid,True);
		#plotting.solutionPlotsDim(x,y,u,v,eta,ts,L,FORCE,BG,Fpos,N);
	
	# Plots of PV and zonally averaged PV
	if doPV:
		if plotPV:
			#plotting.pvPlots(PV_full,PV_prime,x_nd,y_nd);
			plotting.pvPlots_save(PV_full,PV_prime,P,P_xav,x_nd,y_nd,ts,FORCE,BG,Fpos,N,U0_str,x_grid,y_grid,True);
		if plotPV_av:
			plotting.PV_avPlots(x_nd,y_nd,PV_prime,PV_BG,PV_full,ts,FORCE,BG,Fpos,N);
		if doFootprints:
			if plotFootprint:
				plotting.footprintPlots(x_nd,y_nd,P,P_xav,Fpos,BG,FORCE,nu,r0,period_days,U0_nd,U,N);
	
	# Phase and amplitude
	if plotPhaseAmp:
		plotting.solutionPlotsAmp(x_nd,y_nd,u,v,h,ts,FORCE,BG,Fpos,N);
		plotting.solutionPlotsPhase(x_nd,y_nd,u,v,h,ts,FORCE,BG,Fpos,N);

if __name__ == '__main__':
	RSW_main()

end = time.time()

print(end-start)

## 


