# diagnostics.py
# File containing functions to be called by the master script RSW_visc_1L.py.
# Diagnostic functions include plotting tools, differentiation operators and error calculators.
#====================================================

import numpy as np

#====================================================

# diff_fwd_4th
def diff_fwd_2nd(f,d,p,delta):
# 4th-order centered in space finite difference derivative of a function f
# Assuming that f is a 2-D array, d=0 differentiates the first index (usually y),
# and d=1 differentiates the second index (usually x). d=2 for 1-D array.
# p is a periodic switch. p=0 for solid wall BCs, p=1 for periodic.
# delta is the space step.
# -2	-1 	 0	  1     2
# 1/12 -2/3  0	 2/3  -1/12
# f = f[0...dimy-1]
	if d != 2:
		dimx = np.shape(f)[1];		# Finds the number of gridpoints in the x and y directions
		dimy = np.shape(f)[0];
		df = np.zeros((dimy,dimx),dtype=f.dtype);
	else:
		dimy = np.shape(f)[0];
		df = np.zeros(dimy,dtype=f.dtype);
	
	if p == 0:
		# Solid boundary derivative.
		if d == 0:
		
			df[2:dimy,:] = 3.0 * f[2:dimy,:] / 2.0 - 2.0 * f[1:dimy-1,:] + f[0:dimy-2,:] / 2.0;
		
			# Southern boundary
			df[1,:] = 0.5 * f[2,:] - 0.5 * f[0,:];
			df[0,:] = 3.0 * f[0,:] / 2.0 - 2 * f[1,:] + f[2,:] / 2.0;
	
		elif d == 1:

			df[:,2:dimy] = 3.0 * f[:,2:dimy] / 2.0 - 2.0 * f[:,1:dimy-1] + f[:,0:dimy-2] / 2.0;
		
			# Southern boundary
			df[:,1] = 0.5 * f[:,2] - 0.5 * f[:,0];
			df[:,0] = 3.0 * f[:,0] / 2.0 - 2 * f[:,1] + f[:,2] / 2.0;
		
		elif d == 2:

			df[2:dimy] = 3.0 * f[2:dimy] / 2.0 - 2.0 * f[1:dimy-1] + f[0:dimy-2] / 2.0;
		
			# Southern boundary
			df[1] = 0.5 * f[2] - 0.5 * f[0];
			df[0] = 3.0 * f[0] / 2.0 - 2 * f[1] + f[2] / 2.0;

		else:
			print('error')

	elif p == 1:
		# Periodic option

		if d == 0:
		
			df[2:dimy,:] = 3.0 * f[2:dimy,:] / 2.0 - 2.0 * f[1:dimy-1,:] + f[0:dimy-2,:] / 2.0;
		
			# Southern boundary
			df[1,:] = 3.0 * f[1,:] / 2.0 - 2.0 * f[0,:] + f[dimy-1,:] / 2.0;
			df[0,:] = 3.0 * f[0,:] / 2.0 - 2.0 * f[dimy-1,:] + f[dimy-2,:] / 2.0;
	
		elif d == 1:

			df[:,2:dimy] = 3.0 * f[:,2:dimy] / 2.0 - 2.0 * f[:,1:dimy-1] + f[:,0:dimy-2] / 2.0;
		
			# Southern boundary
			df[:,1] = 3.0 * f[:,1] / 2.0 - 2.0 * f[:,0] + f[:,dimy-1] / 2.0;
			df[:,0] = 3.0 * f[:,0] / 2.0 - 2.0 * f[:,dimy-1] + f[:,dimy-2] / 2.0;

		elif d == 2:

			df[2:dimy] = 3.0 * f[2:dimy] / 2.0 - 2.0 * f[1:dimy-1] + f[0:dimy-2] / 2.0;
		
			# Southern boundary
			df[1] = 3.0 * f[1] / 2.0 - 2.0 * f[0] + f[dimy-1] / 2.0;
			df[0] = 3.0 * f[0] / 2.0 - 2.0 * f[dimy-1] + f[dimy-2] / 2.0;
			
			#print(str(df[0])+'='+str(f[1])+'+'+str(f[dimy-1]));
			#print(str(df[dimy-1])+'='+str(f[0])+'+'+str(f[dimy-2]));
		
		else:
			print('error')

	else:
		print('error')

	df = df / delta;

	return df

#====================================================

# diff_center_4th
def diff_center_4th(f,d,p,delta):
# 4th-order centered in space finite difference derivative of a function f
# Assuming that f is a 2-D array, d=0 differentiates the first index (usually y),
# and d=1 differentiates the second index (usually x). d=2 for 1-D array.
# p is a periodic switch. p=0 for solid wall BCs, p=1 for periodic.
# delta is the space step.
# -2	-1 	 0	  1     2
# 1/12 -2/3  0	 2/3  -1/12
# f = f[0...dimy-1]
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
		
			df[2:dimy-2,:] = 2.0 * (f[3:dimy-1,:] - f[1:dimy-3,:]) / 3.0 + (f[0:dimy-4,:] - f[4:dimy,:]) / 12.0;
		
			# Southern boundary
			df[1,:] = 0.5 * (f[2,:] - f[0,:]);
			df[0,:] = f[1,:] - f[0,:];
			# Northern boundary
			df[dimy-2,:] = 0.5 * (f[dimy-1,:] - f[dimy-3,:]);
			df[dimy-1,:] = f[dimy-1,:] - f[dimy-2,:];
	
		elif d == 1:

			df[:,2:dimy-2] = 2.0 * (f[:,3:dimy-1] - f[:,1:dimy-3]) / 3.0 + (f[:,0:dimy-4] - f[:,4:dimy]) / 12.0;
		
			# Southern boundary
			df[:,1] = 0.5 * (f[:,2] - f[:,0]);
			df[:,0] = f[:,1] - f[:,0];
			# Northern boundary
			df[:,dimy-2] = 0.5 * (f[:,dimy-1] - f[:,dimy-3]);
			df[:,dimy-1] = f[:,dimy-1] - f[:,dimy-2];
		
		elif d == 2:

			df[2:dimy-2] = 2.0 * (f[3:dimy-1] - f[1:dimy-3]) / 3.0 + (f[0:dimy-4] - f[4:dimy]) / 12.0;
		
			# Southern boundary
			df[1] = 0.5 * (f[2] - f[0]);
			df[0] = f[1] - f[0];
			# Northern boundary
			df[dimy-2] = 0.5 * (f[dimy-1] - f[dimy-3]);
			df[dimy-1] = f[dimy-1] - f[dimy-2];

		else:
			print('error')

	elif p == 1:
		# Periodic option

		if d == 0:
		
			df[2:dimy-2,:] = 2.0 * (f[3:dimy-1,:] - f[1:dimy-3,:]) / 3.0 + (f[0:dimy-4,:] - f[4:dimy,:]) / 12.0;
		
			# Southern boundary
			df[1,:] = 2.0 * (f[2,:] - f[0,:]) / 3.0 + (f[dimy-1,:] - f[3,:]) / 12.0;
			df[0,:] = 2.0 * (f[1,:] - f[dimy-1,:]) / 3.0 + (f[dimy-2,:] - f[2,:]) / 12.0;
			# Northern boundary
			df[dimy-2,:] = 2.0 * (f[dimy-1,:] - f[dimy-3,:]) / 3.0 + (f[dimy-4,:] - f[0,:]) / 12.0;
			df[dimy-1,:] = 2.0 * (f[0,:] - f[dimy-2,:]) / 3.0 + (f[dimy-3,:] - f[1,:]) / 12.0;
	
		elif d == 1:

			df[:,2:dimy-2] = 2.0 * (f[:,3:dimy-1] - f[:,1:dimy-3]) / 3.0 + (f[:,0:dimy-4] - f[:,4:dimy]) / 12.0;
		
			# Southern boundary
			df[:,1] = 2.0 * (f[:,2] - f[:,0]) / 3.0 + (f[:,dimy-1] - f[:,3]) / 12.0;
			df[:,0] = 2.0 * (f[:,1] - f[:,dimy-1]) / 3.0 + (f[:,dimy-2] - f[:,2]) / 12.0;
			# Northern boundary
			df[:,dimy-2] = 2.0 * (f[:,dimy-1] - f[:,dimy-3]) / 3.0 + (f[:,dimy-4] - f[:,0]) / 12.0;
			df[:,dimy-1] = 2.0 * (f[:,0] - f[:,dimy-2]) / 3.0 + (f[:,dimy-3] - f[:,1]) / 12.0;
		
		elif d == 2:

			df[2:dimy-2] = 2.0 * (f[3:dimy-1] - f[1:dimy-3]) / 3.0 + (f[0:dimy-4] - f[4:dimy]) / 12.0;
		
			# Southern boundary
			df[1] = 2.0 * (f[2] - f[0]) / 3.0 + (f[dimy-1] - f[3]) / 12.0;
			df[0] = 2.0 * (f[1] - f[dimy-1]) / 3.0 + (f[dimy-2] - f[2]) / 12.0;
			# Northern boundary
			df[dimy-2] = 2.0 * (f[dimy-1] - f[dimy-3]) / 3.0 + (f[dimy-4] - f[0]) / 12.0;
			df[dimy-1] = 2.0 * (f[0] - f[dimy-2]) / 3.0 + (f[dimy-3] - f[1]) / 12.0;
			
			#print(str(df[0])+'='+str(f[1])+'+'+str(f[dimy-1]));
			#print(str(df[dimy-1])+'='+str(f[0])+'+'+str(f[dimy-2]));
		
		else:
			print('error')

	else:
		print('error')

	df = df / delta;

	return df

#====================================================

# diff
def diff(f,d,p,delta):
# Function for differentiating a vector
# f[y,x] is the function to be differentiated.
# d is the direction in which the differentiation is to be taken:
# d=0 for differentiation over the first index, d=1 for the second.
# d=2 for a 1D vector
# p is a periodic switch:
# p=1 calculates periodic derivative.
# Need to be careful with periodic derivatives, output depends on whether f[0]=f[dim-1] or =f[dim].
	
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

			df[1:dimy-1] = f[2:dimy] - f[0:dimy-2];

			df[0] = f[1] - f[dimy-2];
			df[dimy-1] = f[1] - f[dimy-2];
			
			#print(str(df[0])+'='+str(f[1])+'+'+str(f[dimy-1]));
			#print(str(df[dimy-1])+'='+str(f[0])+'+'+str(f[dimy-2]));
		
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
def error(u,v,h,dx_nd,dy_nd,dt_nd,U0_nd,H0_nd,Ro,gamma_nd,Re,f_nd,F1_nd,F2_nd,F3_nd,T_nd,ts,omega_nd,N):
# This function calculates the error of the 1L SW solutions, and is to be used in the main code RSW_visc_1L.py.

	SCHEME = diff;
	#SCHEME = diff_center_4th;	

	if Re == None:
		Ro_Re = 0;
	else:
		Ro_Re = Ro / Re;
	
	I = np.complex(0,1);
	ts = 10;
	# Now we calculate all the relevant x and y derivatives
	u_y = SCHEME(u[:,:,ts],0,0,dy_nd);
	u_yy = SCHEME(u_y[:,:],0,0,dy_nd);
	u_x = SCHEME(u[:,:,ts],1,1,dx_nd);
	u_xx = SCHEME(u_x[:,:],1,1,dx_nd);
	
	v_y = SCHEME(v[:,:,ts],0,0,dy_nd);
	v_yy = SCHEME(v_y[:,:],0,0,dy_nd);
	v_x = SCHEME(v[:,:,ts],1,1,dx_nd);
	v_xx = SCHEME(v_x[:,:],1,1,dx_nd);

	eta_x = SCHEME(h[:,:,ts],1,1,dx_nd);
	eta_y = SCHEME(h[:,:,ts],0,0,dy_nd);

	U0_y = SCHEME(U0_nd,2,0,dy_nd);
	H0_y = SCHEME(H0_nd,2,0,dy_nd);
	
	# t derivatives
	u_t = ddt(u,dt_nd);
	v_t = ddt(v,dt_nd);
	eta_t = ddt(h,dt_nd);
	
	e11 = np.zeros((N,N),dtype=complex);
	e12 = np.zeros((N,N),dtype=complex);
	e13 = np.zeros((N,N),dtype=complex);
	e14 = np.zeros((N,N),dtype=complex);
	e15 = np.zeros((N,N),dtype=complex);
	e16 = np.zeros((N,N),dtype=complex);

	for i in range(0,N):
		for j in range(0,N):
			e11[j,i] = Ro * (u_t[j,i,ts] + U0_nd[j] * u_x[j,i]);
			e12[j,i] = gamma_nd * u[j,i,ts];
			e13[j,i] = - Ro_Re * (u_xx[j,i] + u_yy[j,i]);
			e14[j,i] = (Ro * U0_y[j] - f_nd[j]) * v[j,i,ts];
			e15[j,i] = eta_x[j,i];
			e16[j,i] = - Ro * F1_nd[j,i] * np.exp(2. * np.pi * I * omega_nd * T_nd[ts]);
	error1 = e11 + e12 + e13 + e14 + e15 + e16;

	for i in range(0,N):
		for j in range(0,N):
			e11[j,i] = Ro * (v_t[j,i,ts] + U0_nd[j] * v_x[j,i]);
			e12[j,i] = gamma_nd * v[j,i,ts];
			e13[j,i] = - Ro_Re * (v_xx[j,i] + v_yy[j,i]);
			e14[j,i] = f_nd[j] * u[j,i,ts];
			e15[j,i] = eta_y[j,i];
			e16[j,i] = - Ro * F2_nd[j,i] * np.exp(2. * np.pi * I * omega_nd * T_nd[ts]);
	error2 = e11 + e12 + e13 + e14 + e15 + e16;
	
	PLOT = 0;
	if PLOT == 1:
		import matplotlib.pyplot as plt
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
			e13[j,i] = H0_y[j] * v[j,i,ts];
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
	
	e1 = np.sqrt((np.real(error1[3:N-4])**2).mean())
	e2 = np.sqrt((np.real(error2[3:N-4])**2).mean())
	e3 = np.sqrt((np.real(error3[3:N-4])**2).mean())

	return e1, e2, e3

#====================================================

# specError
def specError(utilde_nd,vtilde_nd,etatilde_nd,Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,a1,a2,a3,a4,b4,c1,c2,c3,c4,f_nd,Ro,K_nd,H0_nd,y_nd,dy_nd,N):
# An alternative error metric. Calculates the error associated with the three 1-D spectral equations for a given wavenumber K_nd[i].

	
	SCHEME = diff;
	#SCHEME = diff_center_4th;
	#SCHEME = diff_fwd_2nd;

	# Need relevant derivatives
	utilde_y = SCHEME(utilde_nd,2,0,dy_nd);
	utilde_yy = SCHEME(utilde_y,2,0,dy_nd);
	vtilde_y = SCHEME(vtilde_nd,2,0,dy_nd);
	vtilde_yy = SCHEME(vtilde_y,2,0,dy_nd);
	etatilde_y = SCHEME(etatilde_nd,2,0,dy_nd);

	# Some coefficients need to be multiplied by dy_nd.
	a2 = a2 * dy_nd**2;
	b4 = b4 * 2 * dy_nd;
	c3 = c3 * 2 * dy_nd;
	
	error1 = a1 * utilde_nd + a2 * utilde_yy + a3 * vtilde_nd + a4 * etatilde_nd - Ro * Ftilde1_nd;
	
	error2 = f_nd * utilde_nd + a1 * vtilde_nd + a2 * vtilde_yy + b4 * etatilde_y - Ro * Ftilde2_nd;

	error3 = c1 * utilde_nd + c2 * vtilde_nd + c3 * vtilde_y + c4 * etatilde_nd - Ftilde3_nd

	PLOT1 = False;
	if PLOT1:
		import matplotlib.pyplot as plt
		plt.subplot(221);
		plt.plot(abs(utilde_nd),y_nd);
		plt.title('u');
		plt.subplot(222);
		plt.plot(utilde_y,y_nd);
		plt.title('u_y')
		plt.subplot(223);
		plt.plot(utilde_yy,y_nd);
		plt.title('u_yy');
		plt.subplot(224);
		plt.plot(Ftilde1_nd);
		plt.show();
	
	PLOT2 = False;
	if PLOT2:
		plt.subplot(121);
		plt.plot(np.real(error2),y_nd);
		#plt.contourf(etatilde_nd);	
		plt.ylabel('y');
		plt.subplot(122);
		plt.plot(np.real(Ftilde2_nd),y_nd);
		plt.show();

	error1 = np.sqrt((np.real(error1[2:N-3])**2).mean());
	error2 = np.sqrt((np.real(error2[2:N-3])**2).mean());
	error3 = np.sqrt((np.real(error3[2:N-3])**2).mean());

	return error1,error2,error3;
	

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

#====================================================

# timeAverage
def timeAverage(u,T,Nt):
# Use to calculate the time average.
# Requires 3-D array input, with time in the third index (axis=2).

	shape = np.shape(u);

	# 
	u_new = np.zeros((shape[0],shape[1],shape[2]+1));
	u_new[:,:,0:Nt] = u;
	u_new[:,:,Nt] = u[:,:,0];

	dt = T[1] - T[0];
	u_tav = np.trapz(u_new,T,dt,axis=2);
	u_tav = u_tav / T[Nt];

	return u_tav

#====================================================

# arrayCorr
def arrayCorr(u,v):
# Takes as input a pair of 2D arrays (e.g. solution snapshots), 
# and calculates their correlation by rewriting them as 1D lists.

	Ny, Nx = np.shape(u);
	
	dim = Ny * Nx;

	u_list = np.zeros((dim));
	v_list = np.zeros((dim));

	# Rearrange square arrays into lists.
	for i in range(0,Nx):
		for j in range(0,Ny):
			u_list[i*Nx+j] = u[j,i];
			v_list[i*Nx+j] = v[j,i];

	# Correlation between two lists.
	corr = np.corrcoef(u_list,v_list)[0,1];
	
	return corr

#====================================================

# arrayCorrTime
def arrayCorrTime(u,v):
# Uses the above-defined function to calculate the average correlation between two time-dependent arrays.
	
	Nt = np.shape(u)[2];
	
	# Initialise the correlation.
	corr = 0;

	# Add the correlation between u and v at each time step.
	for ti in range(0,Nt):
		corr = corr + arrayCorr(u[:,:,ti],v[:,:,ti]);
	
	# Average.
	corr = corr / Nt;

	return corr

#====================================================




	






