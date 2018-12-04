# output.py
#=======================================================

# This file contains modules to be called by RSW_1L.py.
# It saves data in netcdf format using the netCDF4 module. 

#=======================================================

import numpy as np
import netCDF4 as nc

#=======================================================

# ncSave
def ncSaveSols(utilde_nd,vtilde_nd,etatilde_nd,u,v,h,x_nd,y_nd,K_nd,T_nd,PV_FULL,PV_PRIME,PV_bg,P,EEF,N,Nt):
# A function that saves the output of RSW_1L.py in netcdf format.
# Saves the solutions (physical and spectral by default) and depending on whether or not
# they were calculated, also saves the PV, footprints, and EEF.
# The last thing to be saved are the BG flow U0, forcing radius r0, kinematic viscosity nu, forcing period.

	# Initialise the nc file
	RSW1L = nc.Dataset('RSW1L.nc','w',format='NETCDF4');
		
	# Create dimensions
	x_dim = RSW1L.createDimension('x_dim',N+1);	
	y_dim = RSW1L.createDimension('y_dim',N);
	k_dim = RSW1L.createDimension('k_dim',N);
	t_dim = RSW1L.createDimension('t_dim',Nt);
	real_imag = RSW1L.createDimension('real_imag',2);

	# Initialise dimension variables...
	x = RSW1L.createVariable('x','f8',('x_dim',));
	y = RSW1L.createVariable('y','f8',('y_dim',));
	k = RSW1L.createVariable('k','f8',('k_dim',));
	t = RSW1L.createVariable('t','f8',('t_dim',));
	# ...and assign the data.
	x[:] = x_nd;
	y[:] = y_nd;
	k[:] = K_nd;
	t[:] = T_nd[0:Nt];

	# Initialise solution variables...
	u = RSW1L.createVariable('u','f8',('y_dim','x_dim','t_dim',));
	v = RSW1L.createVariable('v','f8',('y_dim','x_dim','t_dim',));
	eta = RSW1L.createVariable('eta','f8',('y_dim','x_dim','t_dim',));
	utilde = RSW1L.createVariable('utilde_real','f4',('k_dim','y_dim','real_imag',));
	vtilde = RSW1L.createVariable('vtilde_real','f4',('k_dim','y_dim','real_imag',));
	etatilde = RSW1L.createVariable('etatilde_real','f4',('k_dim','y_dim','real_imag',));	
	
	# ...and assign the data to the variables
	u[:,:,:] = u;
	v[:,:,:] = v;
	eta[:,:,:] = h;
	utilde[:,:,0] = np.real(utilde_nd); utilde[:,:,1] = np.imag(utilde_nd);
	vtilde[:,:,0] = np.real(vtilde_nd); vtilde[:,:,1] = np.imag(vtilde_nd);
	etatilde[:,:,0] = np.real(etatilde_nd); etatilde[:,:,1] = np.imag(etatilde_nd);	

	# Some variables (PV, footprint, EEF) are conditional on their existence in RSW_1L.py
	# Here we initialise them, and assign data to them.
	if PV_FULL != None:
		PV_full = RSW1L.createVariable('PV','f4',('t_dim','x_dim','y_dim',));
		PV_prime = RSW1L.createVariable('PV_prime','f4',('t_dim','x_dim','y_dim',));
		PV_BG = RSW1L.createVariable('PV_BG','f4',('y_dim',));
		#==
		PV_full[:,:,:] = PV_FULL;
		PV_prime[:,:,:] = PV_prime;
		PV_BG[:] = PV_bg;
	if P != None:
		P = RSW1L.createVariable('P','f4',('x_dim','y_dim',));
		#==
		P[:,:] = Pq;		
	if EEF != None:
		EEF = RSW1L.createVariable('EEF','f4');
		#==
		EEF[:] = EEFq;

	RSW1L.close();

#=======================================================

# ncSaveEigenmodes
def ncSaveEigenmodes(modes,val,zero_count,y_nd,k,N,dim,BC):
# A function that saves the output of EIG.py in netcdf format, with the u,v and eta modes under separate variable names.
# Saves the solutions (physical and spectral by default) and depending on whether or not
# they were calculated, also saves the PV, footprints, and EEF.
# The last thing to be saved are the BG flow U0, forcing radius r0, kinematic viscosity nu, forcing period.

	file_name = 'RSW1L_Eigenmodes_k' + str(int(k)) + '_N' + str(int(N)) + '.nc';

	# Initialise the nc file
	RSW1L_Eigenmodes = nc.Dataset(file_name,'w',format='NETCDF4');
		
	# Create dimensions
	y3_dim = RSW1L_Eigenmodes.createDimension('y3_dim',dim);
	omega_dim = RSW1L_Eigenmodes.createDimension('omega_dim',dim);
	real_imag = RSW1L_Eigenmodes.createDimension('real_imag',2);

	# Initialise dimension variables...
	y3 = RSW1L_Eigenmodes.createVariable('y3','f8',('y3_dim',)); # Here y3_dim is 3 times the domain, minus some endpoints
	omega = RSW1L_Eigenmodes.createVariable('omega','f8',('omega_dim','real_imag',));
	# ...and assign the data.
	if BC == 'NO-SLIP':		
		y3[0:N-2] = y_nd[1:N-1];
		y3[N-2:2*N-4] = y_nd[1:N-1];
		y3[2*N-4:3*N-4] = y_nd;
	if BC == 'FREE-SLIP':
		y3[0:N] = y_nd[0:N];
		y3[N:2*N-2] = y_nd[1:N-1];
		y3[2*N-2:3*N-2] = y_nd;		
	omega[:,0] = np.real(val); omega[:,1] = np.imag(val);
	
	# Initialise solution variables...
	vec = RSW1L_Eigenmodes.createVariable('vec','f8',('y3_dim','omega_dim','real_imag',));
	count = RSW1L_Eigenmodes.createVariable('count','i4',('y3_dim',));
	
	# ...and assign the data to the variables.
	vec[:,:,0] = np.real(modes); vec[:,:,1] = np.imag(modes);
	count[:] = zero_count;

	RSW1L_Eigenmodes.close();

#=======================================================

# ncSaveEigenmodes_sep
def ncSaveEigenmodes_sep(u_modes,v_modes,eta_modes,val,y_nd,k,N,dim):
# A function that saves the output of EIG.py in netcdf format, with the u,v and eta modes under separate variable names.
# Saves the solutions (physical and spectral by default) and depending on whether or not
# they were calculated, also saves the PV, footprints, and EEF.
# The last thing to be saved are the BG flow U0, forcing radius r0, kinematic viscosity nu, forcing period.

	file_name = 'RSW1L_Eigenmodes_k' + str(int(k)) + '_N' + str(int(N)) + '.nc';

	# Initialise the nc file
	RSW1L_Eigenmodes = nc.Dataset(file_name,'w',format='NETCDF4');
		
	# Create dimensions
	y_dim = RSW1L_Eigenmodes.createDimension('y_dim',N);
	omega_dim = RSW1L_Eigenmodes.createDimension('omega_dim',dim);
	real_imag = RSW1L_Eigenmodes.createDimension('real_imag',2);

	# Initialise dimension variables...
	y = RSW1L_Eigenmodes.createVariable('y','f8',('y_dim',));
	omega = RSW1L_Eigenmodes.createVariable('omega','f8',('omega_dim','real_imag',));
	# ...and assign the data.
	y[:] = y_nd;
	omega[:,0] = np.real(val); omega[:,1] = np.imag(val);
	
	# Initialise solution variables...
	u_vec = RSW1L_Eigenmodes.createVariable('u_vec','f8',('y_dim','omega_dim','real_imag',));
	v_vec = RSW1L_Eigenmodes.createVariable('v_vec','f8',('y_dim','omega_dim','real_imag',));
	eta_vec = RSW1L_Eigenmodes.createVariable('eta_vec','f8',('y_dim','omega_dim','real_imag',));
	
	# ...and assign the data to the variables.
	u_vec[:,:,0] = np.real(u_modes); u_vec[:,:,1] = np.imag(u_modes);
	v_vec[:,:,0] = np.real(v_modes); v_vec[:,:,1] = np.imag(v_modes);
	eta_vec[:,:,0] = np.real(eta_modes); eta_vec[:,:,1] = np.imag(eta_modes);

	RSW1L_Eigenmodes.close();

#=======================================================

# ncSaveEEF_y0_components
def ncSaveEEF_y0_components(EEF_array,y0_set,period_days,nn):
# A function that saves the output of EQUIV_EDDY_FLUXES.py in NETCDF format.

	file_name = 'EEF_y0_comp_' + str(int(period_days)) + '.nc';

	# Initialise the nc file
	EEF_y0_comp = nc.Dataset(file_name,'w',format='NETCDF4');
		
	# Create dimensions
	y0_dim = EEF_y0_comp.createDimension('y0_dim',nn);
	north_south_dim = EEF_y0_comp.createDimension('north_south_dim',2);

	# Initialise dimension variables...
	y0 = EEF_y0_comp.createVariable('y0','f8',('y0_dim',));
	north_south = EEF_y0_comp.createVariable('north_south','i1',('north_south_dim',))
	# ...and assign the data.
	y0[:] = y0_set;
	north_south[0] = 0; north_south[1] = 1;

	# Initialise solution variables...
	EEF = EEF_y0_comp.createVariable('EEF','f8',('y0_dim','north_south_dim',));
	uq = EEF_y0_comp.createVariable('uq','f8',('y0_dim','north_south_dim',));
	Uq = EEF_y0_comp.createVariable('Uq','f8',('y0_dim','north_south_dim',));
	uQ = EEF_y0_comp.createVariable('uQ','f8',('y0_dim','north_south_dim',));
	vq = EEF_y0_comp.createVariable('vq','f8',('y0_dim','north_south_dim',));
	vQ = EEF_y0_comp.createVariable('vQ','f8',('y0_dim','north_south_dim',));
	 
	# ...and assign the data to the variables.
	EEF[:,:] = EEF_array[:,0,:]; 
	uq[:,:] = EEF_array[:,1,:];
	Uq[:,:] = EEF_array[:,2,:];
	uQ[:,:] = EEF_array[:,3,:];
	vq[:,:] = EEF_array[:,4,:];
	vQ[:,:] = EEF_array[:,5,:];

	EEF_y0_comp.close();



