# forcing_1L
#=======================================================

import numpy as np
import matplotlib.pyplot as plt
from diagnostics import diff, extend

# Forcing 
#=======================================================

# This function is used to define the three forcing terms on the 1-L SW equations.
# F1 is the forcing in physical space applied to the zonal momentum equation.
# F2 is the forcing in physical space applied to the meridional momentum equation.
# F3 is the forcing in physical space applied to the continuity equation.
# These forcings should be defined in normalised form (i.e. sum(Fi=1)), so that they are
# dimensionless and their amplitudes and dimensions are stored in the alphai coefficients.
# For geostrophically balanced forcing, the alphai are functionally related.

#=======================================================

def forcing_dcts(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE,AmpF_nd,f_nd,f0_nd,dx_nd,dy_nd):
	
	Nx = N;
	
	I = np.complex(0,1);
	F1_nd = np.zeros((N,Nx));
	F2_nd = np.zeros((N,Nx));
	F3_nd = np.zeros((N,Nx));

	# Balanced
	if FORCE == 'BALANCED':
		mass = 0;
		for i in range(0,Nx):
			for j in range(0,N):
				r_nd = np.sqrt(x_nd[i]**2 + (y_nd[j]-y0_nd)**2);
				if r_nd < r0_nd:
					if r_nd == 0:
						F1_nd[j,i] = 0;
						F2_nd[j,i] = 0;						
					else:	
						F1_nd[j,i] = AmpF_nd * np.pi * (y_nd[j]-y0_nd) / (2 * r0_nd * f_nd[j] * r_nd) * np.sin((np.pi / 2) * r_nd / r0_nd);
						F2_nd[j,i] = - AmpF_nd * np.pi * x_nd[i] / (2 * r0_nd * f_nd[j] * r_nd) * np.sin((np.pi / 2) * r_nd / r0_nd);
					F3_nd[j,i] = AmpF_nd * np.cos((np.pi / 2) * r_nd / r0_nd);
					mass = mass + F3_nd[j,i];
		mass = mass / (N*Nx)
		F3_nd = F3_nd - mass;
		#F3x = diff(F3,1,1,dx);
		#F3y = diff(F3,0,0,dy);
		#for j in range(0,N):
		#	F1[j,:] = - g * F3y[j,:] / f[j];
		#	F2[j,:] = g * F3x[j,:] / f[j];
			

	# Buoyancy only
	if FORCE == 'BUOYANCY':
		mass = 0;
		for i in range(0,Nx):
			for j in range(0,N):
				r_nd = np.sqrt(x_nd[i]**2 + (y_nd[j]-y0_nd)**2);
				if r_nd < r0_nd:
					F3_nd[j,i] = AmpF_nd * np.cos((np.pi / 2) * r_nd / r0_nd);
					mass = mass + F3_nd[j,i];
		mass = mass / (N*Nx);
		for i in range(0,N):
			for j in range(0,N):
				r_nd = np.sqrt(x_nd[i]**2 + (y_nd[j]-y0_nd)**2);
				if r_nd >= r0_nd:
					F3_nd[j,i] = - mass;

	# Vorticity only
	if FORCE == 'VORTICITY':
		for i in range(0,N):
			for j in range(0,N):
				r_nd = np.sqrt(x_nd[i]**2 + (y_nd[j]-y0_nd)**2);
				if r_nd < r0_nd:
					F1_nd[j,i] = AmpF_nd * np.pi * (y_nd[j]-y0_nd) / (2 * r0_nd * f_nd[j] * r_nd) * np.sin((np.pi / 2) * r_nd / r0_nd);
					F2_nd[j,i] = - AmpF_nd * np.pi * x_nd[i] / (2 * r0_nd * f_nd[j] * r_nd) * np.sin((np.pi / 2) * r_nd / r0_nd);

	
	
	# Lastly, Fourier transform the three forcings in the x-direction
		
	Ftilde1_nd = dx_nd * np.fft.hfft(F1_nd,N,axis=1);	# Multiply by dx_nd as FFT differs by this factor compared to FT.
	Ftilde3_nd = dx_nd * np.fft.hfft(F3_nd,N,axis=1); 
	Ftilde2_nd = dx_nd * np.fft.fft(F2_nd,axis=1);
	

	return F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd;

#=======================================================

def forcing_cts(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE,AmpF_nd,f_nd,f0_nd,dx_nd,dy_nd):
	
	Nx = N	

	I = np.complex(0,1)
	F1_nd = np.zeros((N,Nx))
	F2_nd = np.zeros((N,Nx))
	F3_nd = np.zeros((N,Nx))

	# Balanced
	if FORCE == 'BALANCED':
		mass = 0
		for i in range(0,Nx):
			for j in range(0,N):
				r_nd = np.sqrt(x_nd[i]**2 + (y_nd[j]-y0_nd)**2)
				if r_nd < r0_nd:
					if r_nd == 0:
						F1_nd[j,i] = 0.0
						F2_nd[j,i] = 0.0						
					else:	
						F1_nd[j,i] = 0.5 * AmpF_nd * np.pi * (y_nd[j]-y0_nd) / (r0_nd * f_nd[j] * r_nd) * np.sin(np.pi * r_nd / r0_nd)
						F2_nd[j,i] = - 0.5 * AmpF_nd * np.pi * x_nd[i] / (r0_nd * f_nd[j] * r_nd) * np.sin(np.pi * r_nd / r0_nd)
					F3_nd[j,i] = 0.5 * AmpF_nd * (1.0 + np.cos(np.pi * r_nd / r0_nd))
					mass = mass + F3_nd[j,i]
		mass = mass / (N*Nx)
		F3_nd = F3_nd - mass

	# Buoyancy only
	if FORCE == 'BUOYANCY':
		mass = 0;
		for i in range(0,Nx):
			for j in range(0,N):
				r_nd = np.sqrt(x_nd[i]**2 + (y_nd[j]-y0_nd)**2);
				if r_nd < r0_nd:
					F3_nd[j,i] = 0.5 * AmpF_nd * (1.0 + np.cos(np.pi * r_nd / r0_nd))
					mass = mass + F3_nd[j,i]
		mass = mass / (N*Nx);
		F3_nd = F3_nd - mass;
	
	
	PLOT = False;
	if PLOT:
		aa = 1./24;
		Fmax = np.max(np.max(F3_nd,axis=0));
		Flim = np.max(abs(F3_nd/(1.05*Fmax)));
		#F3[0,0] = - Fmax;
		L = 3840000.
		plt.contourf(x_nd[0:Nx],y_nd,F3_nd/(1.1*Fmax));
		plt.xlabel('x',fontsize=22);
		plt.ylabel('y',fontsize=22);
		plt.text(y_nd[N-130],x_nd[N-130],'F3',color='k',fontsize=22);
		plt.arrow(-aa,2*aa+0.25,2*aa,0,head_width=0.7e5/L, head_length=0.7e5/L,color='k');
		plt.arrow(2*aa,aa+.25,0,-2*aa,head_width=0.7e5/L, head_length=0.7e5/L,color='k');
		plt.arrow(aa,-2*aa+.25,-2*aa,0,head_width=0.7e5/L, head_length=0.7e5/L,color='k');
		plt.arrow(-2*aa,-aa+.25,0,2*aa,head_width=0.7e5/L, head_length=0.7e5/L,color='k');
		plt.xticks((-1./2,0,1./2),['-0.5','0','0.5'],fontsize=14);
		plt.yticks((-1./2,0,1./2),['-0.5','0','0.5'],fontsize=14);
		plt.clim(-Flim,Flim);
		plt.colorbar();
		plt.tight_layout();
		plt.show();

	# Voriticity only
	if FORCE == 'VORTICITY':
		for i in range(0,N):
			for j in range(0,N):
				r_nd = np.sqrt(x_nd[i]**2 + (y_nd[j]-y0_nd)**2)
				if r_nd < r0_nd:
					if r_nd == 0:
						F1_nd[j,i] = 0.0
						F2_nd[j,i] = 0.0						
					else:	
						F1_nd[j,i] = 0.5 * AmpF_nd * np.pi * (y_nd[j]-y0_nd) / (r0_nd * f_nd[j] * r_nd) * np.sin(np.pi * r_nd / r0_nd)
						F2_nd[j,i] = - 0.5 * AmpF_nd * np.pi * x_nd[i] / (r0_nd * f_nd[j] * r_nd) * np.sin(np.pi * r_nd / r0_nd)
	
	# Lastly, Fourier transform the three forcings in the x-direction
		
	Ftilde1_nd = dx_nd * np.fft.hfft(F1_nd,N,axis=1);	# Multiply by dx_nd as FFT differs by this factor compared to FT.
	Ftilde3_nd = dx_nd * np.fft.hfft(F3_nd,N,axis=1); 
	Ftilde2_nd = dx_nd * np.fft.fft(F2_nd,axis=1);
	#Ftilde2_nd = np.zeros((N,Nx),dtype=complex);
	#for j in range(0,N):
	#	for i in range(0,Nx):
	#		Ftilde2_nd[j,i] = 2 * np.pi * I * K_nd[i] * Ftilde3_nd[j,i] / f_nd[j];

	return F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd;

#=======================================================

def forcing_cts2(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE,AmpF_nd,f_nd,f0_nd,beta_nd,dx_nd,dy_nd):
# The same as forcing_cts, but the momentum forcing is constant with latitude.

	Nx = N	

	I = np.complex(0,1)
	F1_nd = np.zeros((N,Nx))
	F2_nd = np.zeros((N,Nx))
	F3_nd = np.zeros((N,Nx))

	# Balanced
	if FORCE == 'BALANCED':
		mass = 0
		for i in range(0,Nx):
			for j in range(0,N):
				r_nd = np.sqrt(x_nd[i]**2 + (y_nd[j]-y0_nd)**2)
				if r_nd < r0_nd:
					if r_nd == 0:
						F1_nd[j,i] = 0.0
						F2_nd[j,i] = 0.0						
					else:	
						F1_nd[j,i] = AmpF_nd * (np.pi * (y_nd[j]-y0_nd) / (r0_nd * r_nd) * np.sin(np.pi * r_nd / r0_nd) - beta_nd * (1.0 + np.cos(np.pi * r_nd / r0_nd)) / f_nd[j])
						F2_nd[j,i] = - AmpF_nd * np.pi * x_nd[i] / (r0_nd * r_nd) * np.sin(np.pi * r_nd / r0_nd)
					F3_nd[j,i] = AmpF_nd * f_nd[j] * (1.0 + np.cos(np.pi * r_nd / r0_nd))
					mass = mass + F3_nd[j,i]
		mass = mass / (N*Nx)
		F3_nd = F3_nd - mass

	# Buoyancy only
	if FORCE == 'BUOYANCY':
		mass = 0;
		for i in range(0,Nx):
			for j in range(0,N):
				r_nd = np.sqrt(x_nd[i]**2 + (y_nd[j]-y0_nd)**2);
				if r_nd < r0_nd:
					F3_nd[j,i] = AmpF_nd * f_nd[j] * (1.0 + np.cos(np.pi * r_nd / r0_nd))
					mass = mass + F3_nd[j,i]
		mass = mass / (N*Nx);
		F3_nd = F3_nd - mass;
	
	# Voriticity only
	if FORCE == 'VORTICITY':
		for i in range(0,N):
			for j in range(0,N):
				r_nd = np.sqrt(x_nd[i]**2 + (y_nd[j]-y0_nd)**2)
				if r_nd < r0_nd:
					if r_nd == 0:
						F1_nd[j,i] = 0.0
						F2_nd[j,i] = 0.0						
					else:	
						F1_nd[j,i] = AmpF_nd * np.pi * (y_nd[j]-y0_nd) / (r0_nd * r_nd) * np.sin(np.pi * r_nd / r0_nd)
						F2_nd[j,i] = - AmpF_nd * np.pi * x_nd[i] / (r0_nd * r_nd) * np.sin(np.pi * r_nd / r0_nd)

	PLOT = False;
	if PLOT:
		aa = 1./24;
		Fmax = np.max(np.max(F3_nd,axis=0));
		Flim = np.max(abs(F3_nd/(1.05*Fmax)));
		#F3[0,0] = - Fmax;
		L = 3840000.
		plt.contourf(x_nd[0:Nx],y_nd,F3_nd/(1.1*Fmax));
		plt.xlabel('x',fontsize=22);
		plt.ylabel('y',fontsize=22);
		plt.text(y_nd[N-130],x_nd[N-130],'F3',color='k',fontsize=22);
		plt.arrow(-aa,2*aa+0.25,2*aa,0,head_width=0.7e5/L, head_length=0.7e5/L,color='k');
		plt.arrow(2*aa,aa+.25,0,-2*aa,head_width=0.7e5/L, head_length=0.7e5/L,color='k');
		plt.arrow(aa,-2*aa+.25,-2*aa,0,head_width=0.7e5/L, head_length=0.7e5/L,color='k');
		plt.arrow(-2*aa,-aa+.25,0,2*aa,head_width=0.7e5/L, head_length=0.7e5/L,color='k');
		plt.xticks((-1./2,-1./4,0,1./4,1./2),['-0.5','-0.25','0','0.25','0.5'],fontsize=14);
		plt.yticks((-1./2,-1./4,0,1./4,1./2),['-0.5','-0.25','0','0.25','0.5'],fontsize=14);
		plt.clim(-Flim,Flim);
		plt.colorbar();
		plt.tight_layout();
		plt.grid()
		plt.show();
	
	# Lastly, Fourier transform the three forcings in the x-direction
		
	Ftilde1_nd = dx_nd * np.fft.hfft(F1_nd,N,axis=1);	# Multiply by dx_nd as FFT differs by this factor compared to FT.
	Ftilde3_nd = dx_nd * np.fft.hfft(F3_nd,N,axis=1); 
	Ftilde2_nd = dx_nd * np.fft.fft(F2_nd,axis=1);
	#Ftilde2_nd = np.zeros((N,Nx),dtype=complex);
	#for j in range(0,N):
	#	for i in range(0,Nx):
	#		Ftilde2_nd[j,i] = 2 * np.pi * I * K_nd[i] * Ftilde3_nd[j,i] / f_nd[j];

	return F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd;

#=======================================================

# forcing_delta
def forcing_delta(AmpF_nd,y0_index,dx_nd,N):
	
	F1_nd = np.zeros((N,N));
	F2_nd = np.zeros((N,N));
	Ftilde1_nd = np.zeros((N,N));
	Ftilde2_nd = np.zeros((N,N));

	F3_nd = np.zeros((N,N));
	F3_nd[y0_index,int(N/2)] = AmpF_nd;
	
	Ftilde3_nd = np.ones((N,N));
	#Ftilde3_nd[y0_index,:] = 1.0;
	Ftilde3_nd = np.fft.hfft(F3_nd,N,axis=1) * dx_nd;

	return F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd;

#=======================================================

# forcingInv
def forcingInv(Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,x_nd,y_nd,dx_nd,N):
# A function that calculates the inverse of the forcing the check that the original forcing is found.

	F1 = np.fft.ifft(Ftilde1_nd,axis=1) / dx_nd;
	F2 = np.fft.ifft(Ftilde2_nd,axis=1) / dx_nd;
	F3 = np.fft.ifft(Ftilde3_nd,axis=1) / dx_nd;

	#F1_nd = np.zeros((N,N));
	#F3_nd = np.zeros((N,N));
	#for i in range(0,N/2):
	#	F1_nd[:,i] = F1i[:,i];
	#	F1_nd[:,N-1-i] = F1i[:,i]; 
	#	F3_nd[:,i] = F3i[:,i];
	#	F3_nd[:,N-1-i] = F3i[:,i];

	return F1, F2, F3;

#=======================================================

# forcingDiff
def forcingDiff(Ftilde_nd,y_nd,dy_nd,N,i):
# Plots the y-derivatives of the 1D forcing at a given wavenumber.
# A function used to test the diff algortithm, improve it, and resolve the error issue.

	Ftilde_y = diff(Ftilde_nd[:,i],2,0,dy_nd);
	Ftilde_yy = diff(Ftilde_y,2,0,dy_nd);

	plt.subplot(131);
	plt.plot(Ftilde_nd[:,i],y_nd);
	plt.subplot(132);
	plt.plot(Ftilde_y,y_nd);
	plt.subplot(133);
	plt.plot(Ftilde_yy,y_nd);
	plt.show();

	import sys
	sys.exit();

#=======================================================

# F12_from_F3
def F12_from_F3(F3_nd,f_nd,dx_nd,dy_nd,N,nx):
# Finds F1 and F3 numerically from F3.

	F1_nd = np.zeros((N,nx),dtype=complex);
	F2_nd = np.zeros((N,nx),dtype=complex);
	for i in range(0,nx):
		F1_nd[:,i] = - diff(F3_nd[:,i],2,0,dy_nd)/f_nd[:];
	for j in range(0,N):
		F2_nd[j,:] = diff(F3_nd[j,:],2,0,dx_nd)/(f_nd[j]);

	return F1_nd, F2_nd;

#=======================================================


