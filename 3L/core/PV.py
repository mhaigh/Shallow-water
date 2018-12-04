# PV
# A module containing footprint-related functions
#=====================================================================

import numpy as np
import matplotlib.pyplot as plt

from diagnostics import diff, extend

#=====================================================================

# vort
# Calculate potential vorticity - the algorithm is the same for each layer and so can be called twice, altering the input.
def vort(u_nd,v_nd,h_nd,u_full,h_full,H_nd,U_nd,N,Nt,dx_nd,dy_nd,f_nd):

	RV_full = np.zeros((N,N,Nt));
	RV_prime = np.zeros((N,N,Nt));
	for ti in range(0,Nt):
		# Define the relative vorticities (RV_full=RV_BG+RV_prime, can always check this numerically)
		RV_full[:,:,ti] = diff(v_nd[:,:,ti],1,1,dx_nd) - diff(u_full[:,:,ti],0,0,dy_nd);
		RV_prime[:,:,ti] = diff(v_nd[:,:,ti],1,1,dx_nd) - diff(u_nd[:,:,ti],0,0,dy_nd);
	RV_BG = - diff(U_nd,2,0,dy_nd);	# This is defined outside the loop as it has no time-dependence.
	
	# Now define two of the PVs
	PV_full = np.zeros((N,N,Nt));
	PV_BG = np.zeros(N);
	for j in range(0,N):
		PV_BG[j] = (RV_BG[j] + f_nd[j]) / H_nd[j];
		for i in range(0,N):
			for ti in range(0,Nt):
				PV_full[j,i,ti] = (RV_full[j,i,ti] + f_nd[j]) / h_full[j,i,ti];
		
	# Two options to define the PV induced in the forced system: (1) PV_full-PV_BG or (2) use the algebraic def. given in the report.
	PV_prime = np.zeros((N,N,Nt));
	for j in range(1,N-1):
		for i in range(0,N):
			for ti in range(0,Nt):
				PV_prime[j,i,ti] = PV_full[j,i,ti] - PV_BG[j];		# Option 1
	#PV_prime = np.zeros((N,N));	# Option 2 - keep this commented out, just use it as a check.
	#for j in range(0,N):
	#	for i in range(0,N):
	#		PV_prime[j,i] = (H_nd[j] * RV_prime[j,i] - h_nd[j,i] * (RV_BG[j] + f_nd[j])) / (eta_full[j,i] * H_nd[j]);

	return PV_prime, PV_full, PV_BG

#====================================================

# footprint
# A function that calculates the PV footprint of the 1L SW solution as produced by RSW_1L.py
def footprint(u_full,v_nd,PV_full,U_nd,U,x_nd,y_nd,dx_nd,dy_nd,AmpF_nd,FORCE1,r0,nu,BG1,Fpos,ts,period_days,N,Nt,GAUSS):
# This code calculates the PV footprint, i.e. the PV flux convergence defined by
# P = -(div(u*q,v*q)-div((u*q)_av,(v*q)_av)), where _av denotees the time average.
# We will take the time average over one whole forcing period, and subtract this PV flux
# convergence from the PV flux convergence at the end of one period.
# To calculate footprints, we use the full PV and velocity profiles.

	qu = PV_full * u_full;		# Zonal PV flux
	qv = PV_full * v_nd;		# Meridional PV flux

	# Next step: taking appropriate derivatives of the fluxes. To save space, we won't define new variables, but instead overwrite the old ones.
	# From these derivatives we can calculate the 
	P = - diff(qu[:,:,0],1,1,dx_nd) - diff(qv[:,:,0],0,0,dy_nd);		# Initialise the footprint with the first time-step
	for ti in range(1,Nt):
		P[:,:] = P[:,:] - diff(qu[:,:,ti],1,1,dx_nd) - diff(qv[:,:,ti],0,0,dy_nd);
	P = P / Nt;
	#P_av = np.trapz(P,T_nd[:Nt],dt_nd,axis=2) / T_nd[Nt-1];
	
	# Normalisation
	P = P / AmpF_nd**2;
	
	# We are interested in the zonal average of the footprint
	P_xav = np.trapz(P,x_nd[:N],dx_nd,axis=1);
	
	P = extend(P);
	
	#import scipy.io as sio
	#sio.savemat('/home/mike/Documents/GulfStream/Code/DATA/1L/' + str(FORCE) + '/' + str(BG) +  '/P_' + str(Fpos) + str(N),{'P':P});
	
	PLOT = 1;
	if PLOT == 1:
		Plim = np.max(abs(P));
		plt.figure(1,figsize=(15,7))
		plt.subplot(121)
		#plt.contourf(x_nd,y_nd,P,cmap='coolwarm')
		plt.contourf(x_nd,y_nd,P)
		plt.text(0.1,0.4,'PV FOOTPRINT',fontsize=18);
		#plt.text(0.25,0.4,str(Fpos),fontsize=18);		# Comment out this line if text on the plot isn't wanted.
		#plt.text(0.15,0.4,'r0 = '+str(r0/1000) + ' km' ,fontsize=18);	
		#plt.text(0.25,0.4,str(int(period_days))+' days',fontsize=18)
		#plt.text(0.25,0.4,'U0 = ' + str(U*U0_nd[0]),fontsize=18);
		#plt.text(0.25,0.4,r'$\nu$ = ' + str(int(nu)),fontsize=18);
		plt.xticks((-1./2,0,1./2))
		plt.yticks((-1./2,0,1./2))
		plt.xlabel('x');
		plt.ylabel('y');
		plt.clim(-Plim,Plim)
		plt.colorbar()
		plt.subplot(122)
		plt.plot(P_xav,y_nd,linewidth=2)
		plt.text(40,0.4,'ZONAL AVERAGE',fontsize=18)
		plt.yticks((-1./2,0,1./2))
		plt.ylim(-0.5,0.5)
		plt.xlim(-1.1*np.max(abs(P_xav)),1.1*np.max(abs(P_xav)));
		plt.tight_layout()
		#plt.savefig('/home/mike/Documents/GulfStream/Code/IMAGES/1L/' + str(FORCE1) + '/' + str(BG) +  '/FOOTPRINT_nu=' + str(nu) + '.png');
		plt.show()

	
	
		# These if loops are for constantly altering depending on the test being done.
		if BG1 == 'GAUSSIAN':
			plt.figure(2)
			plt.contourf(x_nd,y_nd,P)
			#plt.plot(U*U1_nd/(1.5*Umag)-0.5,y_nd,'k--',linewidth=2);
			plt.text(0.25,0.4,str(GAUSS),fontsize=18);
			#plt.text(0.25,0.4,str(period_days)+' days',fontsize=18)
			#plt.text(0.25,0.4,str(Fpos),fontsize=18);
			#plt.plot(P_xav[:,ts],y_nd,linewidth=2)
			#plt.text(0.25,0.4,'r0 = '+str(r0/1000),fontsize=18);	
			plt.colorbar()
			plt.ylim([-0.5,0.5]);
			plt.xticks((-1./2,0,1./2));
			plt.yticks((-1./2,0,1./2));
			plt.xlabel('x');
			plt.ylabel('y');
			plt.tight_layout()
			#plt.savefig('/home/mike/Documents/GulfStream/Code/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/TEST/FOOTPRINT_' + str(GAUSS) + '.png');
		
		if BG1 == 'UNIFORM':
			plt.figure(2)
			plt.contourf(x_nd,y_nd,P)
			#plt.text(0.25,0.4,'U1 = ' + str(U*U0_nd[0]),fontsize=18);
			plt.colorbar()
			plt.xticks((-1./2,0,1./2))
			plt.yticks((-1./2,0,1./2))
			plt.tight_layout()
			#plt.savefig('/home/mike/Documents/GulfStream/Code/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/FOOTPRINT_U0=' + str(U*U0_nd[0]) + '.png');
		

	return P, P_xav

