# forcing
#=======================================================

# This file contains functions used to define the six forcing terms in the 2-L SW equations.
# F1,4 are the forcings in physical space applied to the zonal momentum equations.
# F2,5 are the forcings in physical space applied to the meridional momentum equations.
# F3,6 are the forcings in physical space applied to the continuity equations.
# The appropriate forcing function is called in RSW.py, depending on options defined in inputFile.py.

#=======================================================

import numpy as np
import matplotlib.pyplot as plt
from diagnostics import diff, extend

#=======================================================

# forcing_cts
def forcing_cts(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE1,AmpF_nd,f_nd,U,L,rho1_nd,rho2_nd,dx_nd,dy_nd):
# Takes input defined in the 2-layer input file.

	I = np.complex(0,1);

	Nx = N;		# This variable can be changed for testing purposes.

	# Initialise empty forcing arrays.
	# F4 and F5 are just empty arrays, and will be defined at the end.
	F1_nd = np.zeros((N,Nx));	# u1
	F2_nd = np.zeros((N,Nx));	# v1
	F3_nd = np.zeros((N,Nx));	# h1
	F6_nd = np.zeros((N,Nx));	# h2

	#=======================================================

	if FORCE1 == 'BALANCED':
		mass = 0.0;
		for i in range(0,Nx):
			for j in range(0,N):
				r_nd = np.sqrt(x_nd[i]**2 + (y_nd[j]-y0_nd)**2);
				if r_nd < r0_nd:
					if r_nd == 0:
						F1_nd[j,i] = 0.0;
						F2_nd[j,i] = 0.0;						
					else:	
						F1_nd[j,i] = 0.5 * AmpF_nd * np.pi * (y_nd[j]-y0_nd) / (r0_nd * f_nd[j] * r_nd) * np.sin(np.pi * r_nd / r0_nd);
						F2_nd[j,i] = - 0.5 * AmpF_nd * np.pi * x_nd[i] / (r0_nd * f_nd[j] * r_nd) * np.sin(np.pi * r_nd / r0_nd);
					F3_nd[j,i] = 0.5 * AmpF_nd * (1.0 + np.cos(np.pi * r_nd / r0_nd)) / rho2_nd;

 					mass = mass + F3_nd[j,i];
		mass = mass / (N*Nx);
		print(mass)
		F3_nd = F3_nd - mass;

		F6_nd = - rho1_nd * F3_nd;

	#=======================================================
			
	# Buoyancy only.
	if FORCE1 == 'BUOYANCY':
		mass = 0.0;
		for i in range(0,N):
			for j in range(0,N):
				r_nd = np.sqrt(x_nd[i]**2 + (y_nd[j]-y0_nd)**2);
				if r_nd < r0_nd:
					F3_nd[j,i] = 0.5 * AmpF_nd * (1.0 + np.cos(np.pi * r_nd  / r0_nd)) / rho2_nd;
					mass = mass + F3[j,i];
		mass = mass / (N*Nx);
		F3_nd = F3_nd - mass;

		F6_nd = - rho1_nd * F3_nd;
		
	#=======================================================

	# Vorticity only.
	if FORCE1 == 'VORTICITY':
		for i in range(0,Nx):
			for j in range(0,N):
				r_nd = np.sqrt(x_nd[i]**2 + (y_nd[j]-y0_nd)**2);
				if r_nd < r0_nd:
					if r_nd == 0:
						F1_nd[j,i] = 0.0;
						F2_nd[j,i] = 0.0;						
					else:	
						F1_nd[j,i] = 0.5 * AmpF_nd * np.pi * (y_nd[j]-y0_nd) / (r0_nd * f_nd[j] * r_nd) * np.sin(np.pi * r_nd / r0_nd);
						F2_nd[j,i] = - 0.5 * AmpF_nd * np.pi * x_nd[i] / (r0_nd * f_nd[j] * r_nd) * np.sin(np.pi * r_nd / r0_nd);
		
	#=======================================================	
	
	# Lastly, Fourier transform the three forcings in the x-direction
		
	Ftilde1_nd = dx_nd * np.fft.hfft(F1_nd,N,axis=1);	# Multiply by dx_nd as FFT differs by this factor compared to FT.
	Ftilde3_nd = dx_nd * np.fft.hfft(F3_nd,N,axis=1); 
	Ftilde2_nd = dx_nd * np.fft.fft(F2_nd,axis=1);
	Ftilde6_nd = - rho1_nd * Ftilde3_nd;

	#=======================================================

	# Define arrays of zeros.
	F4_nd = np.zeros((N,Nx));
	F5_nd = np.zeros((N,Nx));
	Ftilde4_nd = np.zeros((N,Nx));
	Ftilde5_nd = np.zeros((N,Nx));

	return F1_nd, F2_nd, F3_nd, F4_nd, F5_nd, F6_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd, Ftilde4_nd, Ftilde5_nd, Ftilde6_nd;

#=======================================================

def forcing_cts2(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE1,AmpF_nd,rho1_nd,rho2_nd,f_nd,f0_nd,beta_nd,dx_nd,dy_nd):
# The same as forcing_cts, but the momentum forcing is constant with latitude.

	I = np.complex(0,1);

	Nx = N;		# This variable can be changed for testing purposes.

	# Initialise empty forcing arrays.
	# F4 and F5 are just empty arrays, and will be defined at the end.
	F1_nd = np.zeros((N,Nx));	# u1
	F2_nd = np.zeros((N,Nx));	# v1
	F3_nd = np.zeros((N,Nx));	# h1
	F6_nd = np.zeros((N,Nx));	# h2

	#=======================================================

	if FORCE1 == 'BALANCED':
		mass = 0.0;
		for i in range(0,Nx):
			for j in range(0,N):
				r_nd = np.sqrt(x_nd[i]**2 + (y_nd[j]-y0_nd)**2);
				if r_nd < r0_nd:
					if r_nd == 0:
						F1_nd[j,i] = 0.0;
						F2_nd[j,i] = 0.0;						
					else:	
						F1_nd[j,i] = AmpF_nd * (np.pi * (y_nd[j]-y0_nd) / (r0_nd * r_nd) * np.sin(np.pi * r_nd / r0_nd) - beta_nd * (1.0 + np.cos(np.pi * r_nd / r0_nd)) / f_nd[j])
						F2_nd[j,i] = - AmpF_nd * np.pi * x_nd[i] / (r0_nd * r_nd) * np.sin(np.pi * r_nd / r0_nd)
					F3_nd[j,i] = 0.5 * AmpF_nd * (1.0 + np.cos(np.pi * r_nd / r0_nd)) / rho2_nd;

 					mass = mass + F3_nd[j,i];
		mass = mass / (N*Nx);

		F3_nd = F3_nd - mass;

		F6_nd = - rho1_nd * F3_nd;

	#=======================================================
			
	# Buoyancy only.
	if FORCE1 == 'BUOYANCY':
		mass = 0.0;
		for i in range(0,N):
			for j in range(0,N):
				r_nd = np.sqrt(x_nd[i]**2 + (y_nd[j]-y0_nd)**2);
				if r_nd < r0_nd:
					F3_nd[j,i] = 0.5 * AmpF_nd * (1.0 + np.cos(np.pi * r_nd  / r0_nd)) / rho2_nd;
					mass = mass + F3[j,i];
		mass = mass / (N*Nx);
		F3_nd = F3_nd - mass;

		F6_nd = - rho1_nd * F3_nd;
		
	#=======================================================

	# Vorticity only.
	if FORCE1 == 'VORTICITY':
		for i in range(0,Nx):
			for j in range(0,N):
				r_nd = np.sqrt(x_nd[i]**2 + (y_nd[j]-y0_nd)**2);
				#print(r_nd - r0_nd)
				if r_nd < r0_nd:
					if r_nd == 0:
						F1_nd[j,i] = 0.0;
						F2_nd[j,i] = 0.0;						
					else:	
						F1_nd[j,i] = AmpF_nd * (np.pi * (y_nd[j]-y0_nd) / (r0_nd * r_nd) * np.sin(np.pi * r_nd / r0_nd) - beta_nd * (1.0 + np.cos(np.pi * r_nd / r0_nd)) / f_nd[j])
						F2_nd[j,i] = - AmpF_nd * np.pi * x_nd[i] / (r0_nd * r_nd) * np.sin(np.pi * r_nd / r0_nd)
		
	#=======================================================	
	
	# Lastly, Fourier transform the three forcings in the x-direction
		
	Ftilde1_nd = dx_nd * np.fft.hfft(F1_nd,N,axis=1);	# Multiply by dx_nd as FFT differs by this factor compared to FT.
	Ftilde3_nd = dx_nd * np.fft.hfft(F3_nd,N,axis=1); 
	Ftilde2_nd = dx_nd * np.fft.fft(F2_nd,axis=1);
	Ftilde6_nd = - rho1_nd * Ftilde3_nd;

	#=======================================================

	# Define arrays of zeros.
	F4_nd = np.zeros((N,Nx));
	F5_nd = np.zeros((N,Nx));
	Ftilde4_nd = np.zeros((N,Nx));
	Ftilde5_nd = np.zeros((N,Nx));


	return F1_nd, F2_nd, F3_nd, F4_nd, F5_nd, F6_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd, Ftilde4_nd, Ftilde5_nd, Ftilde6_nd;

#=======================================================
#=======================================================


def forcing_cts2_PV0(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE1,AmpF_nd,rho1_nd,rho2_nd,f_nd,f0_nd,beta_nd,dx_nd,dy_nd):
	''' The same as cts2, but no geostrophic balance in the lower layer. 
	Instead we impose zero mom and zero buoyancy forcing, so that we have zero PV forcing.'''

	I = np.complex(0,1);

	Nx = N;		# This variable can be changed for testing purposes.

	# Initialise empty forcing arrays.
	# F4 and F5 are just empty arrays, and will be defined at the end.
	F1_nd = np.zeros((N,Nx));	# u1
	F2_nd = np.zeros((N,Nx));	# v1
	F3_nd = np.zeros((N,Nx));	# h1
	F6_nd = np.zeros((N,Nx));	# h2

	#=======================================================

	if FORCE1 == 'BALANCED':
		mass = 0.0
		for i in range(0,Nx):
			for j in range(0,N):
				r_nd = np.sqrt(x_nd[i]**2 + (y_nd[j]-y0_nd)**2);
				if r_nd < r0_nd:
					if r_nd == 0:
						F1_nd[j,i] = 0.0;
						F2_nd[j,i] = 0.0;						
					else:	
						F1_nd[j,i] = AmpF_nd * (np.pi * (y_nd[j]-y0_nd) / (r0_nd * r_nd) * np.sin(np.pi * r_nd / r0_nd) - beta_nd * (1.0 + np.cos(np.pi * r_nd / r0_nd)) / f_nd[j])
						F2_nd[j,i] = - AmpF_nd * np.pi * x_nd[i] / (r0_nd * r_nd) * np.sin(np.pi * r_nd / r0_nd)
					F3_nd[j,i] = 0.5 * AmpF_nd * (1.0 + np.cos(np.pi * r_nd / r0_nd))

 					mass = mass + F3_nd[j,i]

		mass = mass / (N*Nx)

		F3_nd = F3_nd - mass

	#=======================================================
			
	# Buoyancy only.
	if FORCE1 == 'BUOYANCY':
		mass = 0.0;
		for i in range(0,N):
			for j in range(0,N):
				r_nd = np.sqrt(x_nd[i]**2 + (y_nd[j]-y0_nd)**2);
				if r_nd < r0_nd:
					F3_nd[j,i] = 0.5 * AmpF_nd * (1.0 + np.cos(np.pi * r_nd  / r0_nd)) / rho2_nd;
					mass = mass + F3[j,i];
		mass = mass / (N*Nx);
		F3_nd = F3_nd - mass;

		F6_nd = - rho1_nd * F3_nd;
		
	#=======================================================

	# Vorticity only.
	if FORCE1 == 'VORTICITY':
		for i in range(0,Nx):
			for j in range(0,N):
				r_nd = np.sqrt(x_nd[i]**2 + (y_nd[j]-y0_nd)**2);
				#print(r_nd - r0_nd)
				if r_nd < r0_nd:
					if r_nd == 0:
						F1_nd[j,i] = 0.0;
						F2_nd[j,i] = 0.0;						
					else:	
						F1_nd[j,i] = AmpF_nd * (np.pi * (y_nd[j]-y0_nd) / (r0_nd * r_nd) * np.sin(np.pi * r_nd / r0_nd) - beta_nd * (1.0 + np.cos(np.pi * r_nd / r0_nd)) / f_nd[j])
						F2_nd[j,i] = - AmpF_nd * np.pi * x_nd[i] / (r0_nd * r_nd) * np.sin(np.pi * r_nd / r0_nd)
		
	#=======================================================	
	
	# Lastly, Fourier transform the three forcings in the x-direction
		
	Ftilde1_nd = dx_nd * np.fft.hfft(F1_nd,N,axis=1);	# Multiply by dx_nd as FFT differs by this factor compared to FT.
	Ftilde3_nd = dx_nd * np.fft.hfft(F3_nd,N,axis=1); 
	Ftilde2_nd = dx_nd * np.fft.fft(F2_nd,axis=1);
	Ftilde6_nd = - rho1_nd * Ftilde3_nd;

	#=======================================================

	# Define arrays of zeros.
	F4_nd = np.zeros((N,Nx));
	F5_nd = np.zeros((N,Nx));
	Ftilde4_nd = np.zeros((N,Nx));
	Ftilde5_nd = np.zeros((N,Nx));


	return F1_nd, F2_nd, F3_nd, F4_nd, F5_nd, F6_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd, Ftilde4_nd, Ftilde5_nd, Ftilde6_nd;

#=======================================================
#=======================================================



#=======================================================
#=======================================================


# forcing_dcts
def forcing_dcts(x,y,K,y0,r0,N,FORCE1,FORCE2,AmpF,g,f,f0,U,L,rho1_nd,rho2_nd,dx,dy):
# Takes input defined in the 2-layer input file.

	I = np.complex(0,1);

	Nx = N;		# This variable can be changed for testing purposes.

	# Initialise empty forcing arrays.
	# F4 and F5 are just empty arrays, and will be defined at the end.
	F1_nd = np.zeros((N,Nx));	# u1
	F2_nd = np.zeros((N,Nx));	# v1
	F3_nd = np.zeros((N,Nx));	# h1
	F6_nd = np.zeros((N,Nx));	# h2

	#=======================================================

	if FORCE1 == 'BALANCED':
		mass = 0;
		for i in range(0,Nx):
			for j in range(0,N):
				r = np.sqrt(x[i]**2 + (y[j]-y0)**2);
				if r < r0:	
					count = count + 1;
					if r == 0:
						F1[j,i] = 0;
						F2[j,i] = 0;
					else:
						F1[j,i] = AmpF * np.pi * g * (y[j]-y0) / (2 * r0 * f[j] * r) * np.sin((np.pi / 2) * r / r0);
						F2[j,i] = - AmpF * np.pi * g * x[i] / (2 * r0 * f[j] * r) * np.sin((np.pi / 2) * r / r0);
					F3[j,i] = AmpF * np.cos((np.pi / 2) * r / r0) / rho2_nd;
					mass = mass + F3[j,i];
		mass = mass / (N*(Nx) - count);
		for i in range(0,Nx):
			for j in range(0,N):
				r = np.sqrt(x[i]**2 + (y[j]-y0)**2);
				if r >= r0:
					F3[j,i] = - mass;
		#F3x = diff(F3,1,1,dx);
		#F3y = diff(F3,0,0,dy);
		#for j in range(0,N):
		#	F1[j,:] = - g * F3y[j,:] / f[j];
		#	F2[j,:] = g * F3x[j,:] / f[j];

	F6 = - rho1_nd * F3

	#=======================================================
			
	# Buoyancy only.
	if FORCE1 == 'BUOYANCY':
		count = 0;
		mass = 0;
		for i in range(0,Nx):
			for j in range(0,N):
				r = np.sqrt(x[i]**2 + (y[j]-y0)**2);
				if r<r0:
					count = count + 1;
					F3[j,i] = AmpF * np.cos((np.pi / 2) * r / r0);
					mass = mass + F3[j,i];
		mass = mass / (N*(N+1) - count);
		for i in range(0,N+1):
			for j in range(0,N):
				r = np.sqrt(x[i]**2 + (y[j]-y0)**2);
				if r >= r0:
					F3[j,i] = - mass;
		
	#=======================================================

	# Vorticity only.
	if FORCE1 == 'VORTICITY':
		for i in range(0,N+1):
			for j in range(0,N):
				r = np.sqrt(x[i]**2 + (y[j]-y0)**2);
				if r<r0:
					F1[j,i] = AmpF * np.pi * g * (y[j]-y0) / (2 * r0 * f[j] * r) * np.sin((np.pi / 2) * r / r0);
					F2[j,i] = - AmpF * np.pi * g * (x[i]-x0) / (2 * r0 * f[j] * r) * np.sin((np.pi / 2) * r / r0);
	
	#=======================================================
	
	# Lastly, Fourier transform the three forcings in the x-direction
		
	Ftilde1 = dx * np.fft.hfft(F1,N,axis=1);	# Multiply by dx_nd as FFT differs by this factor compared to FT.
	Ftilde3 = dx * np.fft.hfft(F3,N,axis=1); 
	Ftilde2 = np.zeros((N,N),dtype=complex);
	for j in range(0,N):
		for i in range(0,N):
			Ftilde2[j,i] = 2 * np.pi * rho2_nd * g * I * K[i] * Ftilde3[j,i] / f[j];
	Ftilde6 = - rho1_nd * Ftilde3;

	# Nondimensionalise forcing terms
	#=======================================================

	F1_nd = F1 / (f0 * U);
	F2_nd = F2 / (f0 * U);
	F3_nd = F3 * g / (f0 * U**2); 
	F6_nd = F6 * g / (f0 * U**2);

	Ftilde1_nd = Ftilde1 / (f0 * U * L);
	Ftilde2_nd = Ftilde2 / (f0 * U * L);
	Ftilde3_nd = Ftilde3 * g / (f0 * U**2 * L);
	Ftilde6_nd = Ftilde6 * g / (f0 * U**2 * L);
	

	#=======================================================

	# Define arrays of zeros.
	F4_nd = np.zeros((N,Nx));
	F5_nd = np.zeros((N,Nx));
	Ftilde4_nd = np.zeros((N,Nx));
	Ftilde5_nd = np.zeros((N,Nx));


	return F1_nd, F2_nd, F3_nd, F4_nd, F5_nd, F6_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd, Ftilde4_nd, Ftilde5_nd, Ftilde6_nd;

#=======================================================

# forcingTest
def forcingTest(F1_nd,F2_nd,F3_nd,F6_nd,f_nd,rho1_nd,rho2_nd,dy_nd,dx_nd,N):
# This function takes as input the forcing terms defined in a previous function, as tests (visually) whether or not they are in geostrophic balance.	
	
	print('Max F1 = ' + str(np.max(F1_nd)));
	
	# First calculate all the terms we require.
	F0 = F3_nd + F6_nd; # F0 can be interpreted as the SSH forcing.

	F0_x = diff(F0,1,1,dx_nd);
	F0_y = diff(F0,0,0,dy_nd);
	F6_x = diff(F6_nd,1,1,dx_nd);
	F6_y = diff(F6_nd,0,0,dy_nd);

	geo1 = np.zeros((N,N))
	geo2 = np.zeros((N,N))
	MOM1 = np.zeros((N,N))
	for j in range(0,N):
		for i in range(0,N):
			MOM1[j,i] = f_nd[j] * F2_nd[j,i]
			geo1[j,i] = MOM1[j,i] - F0_x[j,i];
			geo2[j,i] = f_nd[j] * F1_nd[j,i] + F0_y[j,i];
	geo3 = rho1_nd * F0_x + rho2_nd * F6_x; 
	geo4 = rho1_nd * F0_y + rho2_nd * F6_y; 

	#====

	plt.subplot(221);
	plt.contourf(F0);
	plt.colorbar();
	plt.title('F0')

	plt.subplot(222);
	plt.contourf(F0_x);
	plt.colorbar();
	plt.title('F0_x')

	plt.subplot(223);
	plt.contourf(MOM1);
	plt.colorbar();
	plt.title('f*F2')

	plt.subplot(224);
	plt.contourf(geo1);
	plt.colorbar();	
	plt.title('Geo bal 1')

	plt.tight_layout();
	plt.show();

	#===

	plt.subplot(221);
	plt.contourf(F6_x);
	plt.colorbar();
	plt.title('F6_x')

	plt.subplot(222);
	plt.contourf(F6_y);
	plt.colorbar();
	plt.title('F6_y')

	plt.subplot(223)
	plt.contourf(geo3)
	plt.colorbar()
	plt.title('v2 geo bal')
	
	plt.subplot(224)
	plt.contourf(geo4)
	plt.colorbar()
	plt.title('u2 geo bal')

	plt.tight_layout();
	plt.show();
	




#=======================================================

# Copy the code below into inputFile.py to test the forcing.
#from output import plotting
#F0 = F3_nd + F6_nd;
#plt.contourf(F0);
#plt.colorbar();
#plt.show();
#plotting.plotForcing(x_grid,y_grid,F1_nd,F2_nd,F3_nd,F6_nd);
#forcing.forcingTest(F1_nd,F2_nd,F3_nd,F6_nd,f_nd,rho1_nd,rho2_nd,dy_nd,dx_nd,N);

