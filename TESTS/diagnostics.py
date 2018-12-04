# diagnostics.py
# File containing functions to be called by the master script RSW_visc_1L.py.
# Diagnostic functions include plotting tools, differentiation operators and error calculators.
#====================================================

import numpy as np
import matplotlib.pyplot as plt

#====================================================

# solutionPlots
# Solution plots
def solutionPlots(x_nd,y_nd,u_nd,v_nd,eta_nd,ts,FORCE,BG,Fpos,N):

	u_nd = extend(u_nd);
	v_nd = extend(v_nd);
	eta_nd = extend(eta_nd);

	ulim = np.max(abs(u_nd[:,:,ts]));
	vlim = np.max(abs(v_nd[:,:,ts]));
	etalim = np.max(abs(eta_nd[:,:,ts]));
	
	plt.figure(1,figsize=(22,6.4));

	plt.subplot(131);
	plt.contourf(x_nd,y_nd,u_nd[:,:,ts]);
	plt.text(0.3,0.45,'u',fontsize=22);
	plt.xticks((-1./2,0,1./2));
	plt.yticks((-1./2,0,1./2));	
	plt.xlabel('x');
	plt.ylabel('y');
	plt.clim(-ulim,ulim);
	plt.colorbar();

	plt.subplot(132);
	plt.contourf(x_nd,y_nd,v_nd[:,:,ts]);
	plt.text(0.3,0.45,'v',fontsize=22);
	plt.xticks((-1./2,0,1./2));
	plt.yticks((-1./2,0,1./2));
	plt.clim(-vlim,vlim);
	plt.colorbar();

	plt.subplot(133);
	plt.contourf(x_nd,y_nd,eta_nd[:,:,ts]);
	plt.text(0.3,0.45,'eta',fontsize=22);
	plt.xticks((-1./2,0,1./2));
	plt.yticks((-1./2,0,1./2));
	plt.clim(-etalim,etalim);
	plt.colorbar();

	plt.tight_layout();
	plt.savefig('/home/mike/Documents/GulfStream/Code/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/' + str(Fpos) + '_'  + str(N) + '.png');
	plt.show();

	#cmap='coolwarm'

#====================================================

# solutionPlotsDim
# Plots of the dimensional solutions
def solutionPlotsDim(x,y,u,v,eta,ts,L,FORCE,BG,Fpos,N):

	u = extend(u);
	v = extend(v);
	eta = extend(eta);
	
	plt.figure(1,figsize=(22,6));

	plt.subplot(131);
	plt.contourf(x,y,u[:,:,ts]);
	plt.text(y[N-N/6],x[N-N/12],'u',fontsize=22);
	plt.xticks((-L/2,0,L/2),['-L/2','0','L/2']);
	plt.yticks((-L/2,0,L/2),['-L/2','0','L/2']);
	plt.colorbar();

	plt.subplot(132);
	plt.contourf(x,y,v[:,:,ts]);
	plt.text(y[N-N/6],x[N-N/12],'v',fontsize=22);
	plt.xticks((-L/2,0,L/2),['-L/2','0','L/2']);
	plt.yticks((-L/2,0,L/2),['-L/2','0','L/2']);
	plt.colorbar();

	plt.subplot(133);
	plt.contourf(x,y,eta[:,:,ts]);
	plt.text(y[N-N/6],x[N-N/12],'eta',fontsize=22);
	plt.xticks((-L/2,0,L/2),['-L/2','0','L/2']);
	plt.yticks((-L/2,0,L/2),['-L/2','0','L/2']);
	plt.colorbar();

	plt.tight_layout();
	plt.savefig('/home/mike/Documents/GulfStream/Code/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/' + str(Fpos) + '_'  + str(N) + '.png');
	plt.show();


#====================================================

# solutionPlotsAmp
# Plots of amplitude 
def solutionPlotsAmp(x_nd,y_nd,u_nd,v_nd,eta_nd,ts,FORCE,BG,Fpos,N):

	u_nd = extend(u_nd);
	v_nd = extend(v_nd);
	eta_nd = extend(eta_nd);
	
	plt.figure(1);

	plt.subplot(131);
	plt.contourf(x_nd,y_nd,np.absolute(u_nd[:,:,ts]));
	plt.xticks((-1./2,0,1./2));
	plt.yticks((-1./2,0,1./2));
	plt.colorbar();

	plt.subplot(132);
	plt.contourf(x_nd,y_nd,np.absolute(v_nd[:,:,ts]));
	plt.xticks((-1./2,0,1./2));
	plt.yticks((-1./2,0,1./2));
	plt.colorbar();

	plt.subplot(133);
	plt.contourf(x_nd,y_nd,np.absolute(eta_nd[:,:,ts]));
	plt.xticks((-1./2,0,1./2));
	plt.yticks((-1./2,0,1./2));
	plt.colorbar();

	plt.tight_layout()
	plt.show()

#====================================================

# solutionPlotsPhase
# Plots of phase 
def solutionPlotsPhase(x_nd,y_nd,u_nd,v_nd,eta_nd,ts,FORCE,BG,Fpos,N):

	u_nd = extend(u_nd);
	v_nd = extend(v_nd);
	eta_nd = extend(eta_nd);

	plt.figure(1)

	plt.subplot(131);
	plt.contourf(x_nd,y_nd,np.angle(u_nd[:,:,ts]));
	plt.text(0.05,0.4,'u PHASE',color='w',fontsize=12);
	plt.xticks((-1./2,0,1./2));
	plt.yticks((-1./2,0,1./2));

	plt.subplot(132);
	plt.contourf(x_nd,y_nd,np.angle(v_nd[:,:,ts]));
	plt.xticks((-1./2,0,1./2));
	plt.yticks((-1./2,0,1./2));

	plt.subplot(133);
	plt.contourf(x_nd,y_nd,np.angle(eta_nd[:,:,ts]));
	plt.xticks((-1./2,0,1./2));
	plt.yticks((-1./2,0,1./2));

	plt.tight_layout()
	plt.show()

#====================================================

# PV_avPlots
# Plots of the zonal averages of PV, both the BG state and the forced state
def PV_avPlots(x_nd,y_nd,PV_prime,PV_BG,PV_full,ts,FORCE,BG,Fpos,N):

	PV_prime = extend(PV_prime);
	PV_full = extend(PV_full);

	PV_full_av = np.trapz(PV_full[:,:,ts],x_nd,x_nd[1]-x_nd[0],1);

	plt.figure(1)

	plt.plot(PV_full_av,y_nd);
	plt.plot(PV_BG,y_nd);

	plt.tight_layout()
	plt.show()


#====================================================

# bgPlots
# Background state plots
def bgPlots(y_nd,H0_nd,U0_nd,PV_BG):

	plt.figure(2);
	plt.subplot(131);
	plt.plot(H0_nd,y_nd);
	plt.yticks((-1./2,0,1./2));
	plt.title('BG SSH');
	plt.subplot(132);
	plt.plot(U0_nd,y_nd);
	plt.yticks((-1./2,0,1./2));
	plt.title('BG flow U0');
	plt.subplot(133);
	plt.plot(PV_BG,y_nd);
	plt.yticks((-1./2,0,1./2));
	plt.title('BG PV');
	plt.show()

#====================================================

# pvPlots
# Plots of PV and footprint
def pvPlots(PV_full,PV_prime,P,x_nd,y_nd):

	PV_full = extend(PV_full);
	PV_prime = extend(PV_prime);
	
	plt.figure(1,figsize=[21,6]);

	plt.subplot(131);
	plt.contourf(x_nd,y_nd,PV_full);
	plt.text(0.05,0.4,'PV FULL',color='k',fontsize=22);
	plt.xticks((-1./2,0,1./2));
	plt.yticks((-1./2,0,1./2));
	plt.colorbar();

	plt.subplot(132);
	plt.contourf(x_nd,y_nd,PV_prime);
	plt.text(0.05,0.4,'PV PRIME',color='k',fontsize=22);
	plt.xticks((-1./2,0,1./2));
	plt.yticks((-1./2,0,1./2));
	plt.colorbar();

	plt.subplot(133);
	plt.contourf(x_nd,y_nd,P);
	plt.text(0.05,0.4,'P',color='k',fontsize=22);
	plt.xticks((-1./2,0,1./2));
	plt.yticks((-1./2,0,1./2));
	plt.colorbar();

	plt.tight_layout()
	plt.show()

#====================================================

# forcingPlots
# Forcing plots 
def forcingPlots(x_nd,y_nd,F1_nd,F2_nd,F3_nd,Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,N):
# Function that plots the forcing, and its Fourier representation.

	plt.figure(1);

	plt.subplot(331);
	plt.contourf(x_nd,y_nd,F1_nd);
	plt.xticks((-1./2,0,1./2));
	plt.yticks((-1./2,0,1./2));
	plt.colorbar();
	plt.subplot(332);
	plt.contourf(x_nd,y_nd,F2_nd);
	plt.xticks((-1./2,0,1./2));
	plt.yticks((-1./2,0,1./2));
	plt.colorbar();
	plt.subplot(333);
	plt.contourf(x_nd,y_nd,F3_nd);
	plt.xticks((-1./2,0,1./2));
	plt.yticks((-1./2,0,1./2));
	plt.colorbar()

	plt.subplot(334);
	plt.contourf(np.real(Ftilde1_nd));
	plt.colorbar()
	plt.subplot(335);
	plt.contourf(np.real(Ftilde2_nd));
	plt.colorbar()
	plt.subplot(336);
	plt.contourf(np.real(Ftilde3_nd));
	plt.colorbar()

	plt.subplot(337);
	plt.contourf(np.imag(Ftilde1_nd));
	plt.colorbar()
	plt.subplot(338);
	plt.contourf(np.imag(Ftilde2_nd));
	plt.colorbar()
	plt.subplot(339);
	plt.contourf(np.imag(Ftilde3_nd));
	plt.colorbar()

	plt.show();

#====================================================

# diff
# Function for differentiating a vector
def diff(f,d,p,delta):

# f[y,x] is the function to be differentiated.
# d is the direction in which the differentiation is to be taken:
# d=0 for differentiation over the first index, d=1 for the second.
# d=2 for a 1D vector
# p is a periodic switch:
# p=1 calculates periodic derivative.
	
	if d != 2:
		dimx = np.shape(f)[1];		# Finds the number of gridpoints in the x and y directions
		dimy = np.shape(f)[0];
		df = np.zeros((dimy,dimx),dtype=f.dtype);
	else:
		dimy = np.shape(f)[0];
		df = np.zeros(dimy,dtype=f.dtype);	
	
	if p == 0:
		# Solid boundary derivative.
		# Note multiplication by 2 of boundary terms are to
		# invert the division by 2 at the end of the module.
		if d == 0:
		
			df[1:dimy-1,:] = f[2:dimy,:] - f[0:dimy-2,:];
		
			df[0,:] = 2 * (f[1,:] - f[0,:]);
			df[dimy-1,:] = 2 * (f[dimy-1,:] - f[dimy-2,:]);
	
		elif d == 1:

			df[:,1:dimx-1] = f[:,2:dimx] - f[:,0:dimx-2];

			df[:,0] = 2 * (f[:,1] - f[:,0]);
			df[:,dimx-1] = 2 * (f[:,0] - f[:,dimx-2]);
		
		elif d == 2:

			df[1:dimy-1] = f[2:dimy] - f[0:dimy-2];

			df[0] = 2 * (f[1] - f[0]);
			df[dimy-1] = 2 * (f[dimy-1] - f[dimy-2]);

		else:
			print('error')

	elif p == 1:
		# Periodic option

		if d == 0:

			df[1:dimy-1,:] = f[2:dimy,:] - f[0:dimy-2,:];	

			df[0,:] = f[1,:] - f[dimy-1,:];
			df[dimy-1,:] = f[0,:] - f[dimy-2,:];
	
		elif d == 1:

			df[:,1:dimx-1] = f[:,2:dimx]-f[:,0:dimx-2];

			df[:,0] = f[:,1] - f[:,dimx-1];
			df[:,dimx-1] = f[:,0] - f[:,dimx-2];

		elif d == 2:

			df[1:dimy-1]=f[2:dimy] - f[0:dimy-2];

			df[0] = f[1] - f[0];
			df[dimy-1] = f[0] - f[dimy-2];
		
		else:
			print('error')

	else:
		print('error')

	df = 0.5 * df / delta;

	return df

#====================================================

# ddt
# A time-derivative function.
def ddt(f,delta):
# Only takes two inputs, the function itself and the size of the timestep as defined in input_File_1L.
# f should be a 3-D vector, where the 3rd index is the time index.
	dimx, dimy, dimt = np.shape(f);
	df = np.zeros((dimy,dimx,dimt),dtype=f.dtype);
	
	df[:,:,1:dimt-1] = f[:,:,2:dimt] - f[:,:,0:dimt-2];	# centered fd

	df[:,:,0] = f[:,:,1] - f[:,:,dimt-1];				# The two boundary terms
	df[:,:,dimt-1] = f[:,:,0] - f[:,:,dimt-2];

	df = 0.5 * df / delta;
	
	return df

#====================================================

# error
def error(u_nd,v_nd,eta_nd,dx_nd,dy_nd,dt_nd,U0_nd,H0_nd,Ro,gamma_nd,Re,f_nd,F1_nd,F2_nd,F3_nd,T_nd,ts,omega_nd,N):
# This function calculates the error of the 1L SW solutions, and is to be used in the main code RSW_visc_1L.py.

	I = np.complex(0,1);
	ts = 12;
	# Now we calculate all the relevant x and y derivatives
	u_y = diff(u_nd[:,:,ts],0,0,dy_nd);
	u_yy = diff(u_y[:,:],0,0,dy_nd);
	u_x = diff(u_nd[:,:,ts],1,1,dx_nd);
	u_xx = diff(u_x[:,:],1,1,dx_nd);

	v_y = diff(v_nd[:,:,ts],0,0,dy_nd);
	v_yy = diff(v_y[:,:],0,0,dy_nd);
	v_x = diff(v_nd[:,:,ts],1,1,dx_nd);
	v_xx = diff(v_x[:,:],1,1,dx_nd);

	eta_x = diff(eta_nd[:,:,ts],1,1,dx_nd);
	eta_y = diff(eta_nd[:,:,ts],0,0,dy_nd);

	U0_y = diff(U0_nd,2,0,dy_nd);
	H0_y = diff(H0_nd,2,0,dy_nd);

	# t derivatives
	u_t = ddt(u_nd,dt_nd);
	v_t = ddt(v_nd,dt_nd);
	eta_t = ddt(eta_nd,dt_nd);
	
	e11 = np.zeros((N,N));
	e12 = np.zeros((N,N));
	e13 = np.zeros((N,N));
	e14 = np.zeros((N,N));
	e15 = np.zeros((N,N));
	e16 = np.zeros((N,N));

	for i in range(0,N):
		for j in range(0,N):
			e11[j,i] = Ro * (u_t[j,i,ts] + U0_nd[j] * u_x[j,i]);
			e12[j,i] = gamma_nd * u_nd[j,i,ts];
			e13[j,i] = - Ro * (u_xx[j,i] + u_yy[j,i]) / Re;
			e14[j,i] = (Ro * U0_y[j] - f_nd[j]) * v_nd[j,i,ts];
			e15[j,i] = eta_x[j,i];
			e16[j,i] = - F1_nd[j,i] * np.exp(2. * np.pi * I * omega_nd * T_nd[ts]);
	error1 = e11 + e12 + e13 + e14 + e15 + e16;


	for i in range(0,N):
		for j in range(0,N):
			e11[j,i] = Ro * (v_t[j,i,ts] + U0_nd[j] * v_x[j,i]);
			e12[j,i] = gamma_nd * v_nd[j,i,ts];
			e13[j,i] = - Ro * (v_xx[j,i] + v_yy[j,i]) / Re;
			e14[j,i] = f_nd[j] * u_nd[j,i,ts];
			e15[j,i] = eta_y[j,i];
			e16[j,i] = - F2_nd[j,i] * np.exp(2. * np.pi * I * omega_nd * T_nd[ts]);
	error2 = e11 + e12 + e13 + e14 + e15 + e16;

	PLOT = 0;
	if PLOT == 1:
		plt.subplot(241);
		plt.contourf(e11);
		plt.colorbar();
		plt.title('ADV')
		plt.subplot(242);
		plt.contourf(e12);
		plt.colorbar();
		plt.title('FRIC')
		plt.subplot(243);
		plt.contourf(e13);
		plt.colorbar();
		plt.title('VISC');
		plt.subplot(244);
		plt.contourf(e14);
		plt.colorbar();
		plt.title('COR')
		plt.subplot(245);
		plt.contourf(e15);
		plt.colorbar();
		plt.title('SSH');
		plt.subplot(246);
		plt.contourf(e16);
		plt.colorbar();
		plt.title('FORCE');
		plt.subplot(247);
		plt.contourf(e14+e15);
		plt.colorbar();
		plt.title('CORR+SSH')
		plt.subplot(248);
		plt.contourf(error2);
		plt.colorbar();
		plt.title('TOTAL');
		plt.show();
	

	for i in range(0,N):
		for j in range(0,N):
			e11[j,i] = eta_t[j,i,ts] + U0_nd[j] * eta_x[j,i];
			e12[j,i] = H0_nd[j] * (u_x[j,i] + v_y[j,i]);
			e13[j,i] = H0_y[j] * v_nd[j,i,ts];
			e14[j,i] = - F3_nd[j,i] * np.exp(2. * np.pi * I * omega_nd * T_nd[ts]);
	error3 = e11 + e12 + e13 + e14;

	PLOT = 0;
	if PLOT == 1:
		plt.subplot(231);
		plt.contourf(e11);
		plt.colorbar();
		plt.title('ADV')
		plt.subplot(232);
		plt.contourf(e12);
		plt.colorbar();
		plt.title('DIV1')
		plt.subplot(233);
		plt.contourf(e13);
		plt.colorbar();
		plt.title('DIV2');
		plt.subplot(234);
		plt.contourf(e12+e13);
		plt.colorbar();
		plt.subplot(235);	
		plt.contourf(e14);
		plt.title('FORCE');
		plt.colorbar();
		plt.subplot(236)
		plt.contourf(error3)
		plt.title('TOTAL');
		plt.colorbar();
		plt.show();
	
	error1 = np.real(error1);
	error2 = np.real(error2);
	error3 = np.real(error3);
	
	e1 = np.sqrt((error1**2).mean())
	e2 = np.sqrt((error2**2).mean())
	e3 = np.sqrt((error3**2).mean())

	return e1, e2, e3

#====================================================

# specError
def specError(utilde_nd,vtilde_nd,etatilde_nd,Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,a1,a2,a3,a4,b4,c1,c2,c3,c4,K_nd,H0_nd,dy_nd,N,i):
# An alternative error metric. Calculates the error associated with the three 1-D spectral equations for a given wavenumber K_nd[i].

	# Need relevant derivatives
	utilde_y = diff(utilde_nd[:,i],2,0,dy_nd);
	utilde_yy = diff(utilde_y,2,0,dy_nd);
	vtilde_y = diff(vtilde_nd[:,i],2,0,dy_nd);
	vtilde_yy = diff(vtilde_y,2,0,dy_nd);
	etatilde_y = diff(etatilde_nd[:,i],2,0,dy_nd);
	
	error1 = a1[:,i] * utilde_nd[:,i] + a2 * utilde_yy + a3 * vtilde_nd[:,i] + a4[i] * etatilde_nd[:,i];# - Ftilde3_nd[:,i];
	plt.plot(error1);
	plt.show();

	error1 = np.sqrt((np.real(error1)**2).mean());

	return error1;
	

#====================================================

# extend
def extend(f):
# A function used to replace the extra x-gridpoint on a solution.

	dimx = np.shape(f)[1];
	dimy = np.shape(f)[0];
	if f.size != dimx * dimy:
		dimt = np.shape(f)[2];

		f_new = np.zeros((dimy,dimx+1,dimt),dtype=f.dtype);
		for i in range(0,dimx):
			f_new[:,i,:] = f[:,i,:];
	
		f_new[:,dimx,:] = f[:,0,:];
	
	else:
		f_new = np.zeros((dimy,dimx+1),dtype=f.dtype);
		for i in range(0,dimx):
			f_new[:,i] = f[:,i];
	
		f_new[:,dimx] = f[:,0];

	return f_new	 







