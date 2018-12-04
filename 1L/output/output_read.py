# output_read.py
#=======================================================

# This files read netcdf data, as produced by output.py. 

#=======================================================

import numpy as np
import netCDF4 as nc

#=======================================================

def ncReadEigenmodes(ncFile):

	I = np.complex(0,1);
	
	Eigenmodes = nc.Dataset(ncFile);	# Read the NETCDF file

	# Extract the eigenmodes and reconstruct from real and imag. parts.
	vec_tmp = Eigenmodes.variables['vec'][:,:,:];
	vec = vec_tmp[:,:,0] + I * vec_tmp[:,:,1];

	# Extract the eigenvalues
	val_tmp = Eigenmodes.variables['omega'][:,:];
	val = val_tmp[:,0] + I * val_tmp[:,1]; 

	# And extract the pseudo-wavenumber = count
	count = Eigenmodes.variables['count'][:];

	return val, vec, count


#=======================================================

def ncReadEEF_y0_components(ncFile):

	dataset = nc.Dataset(ncFile);	# Read the NETCDF file

	EEF = dataset.variables['EEF'][:,:];
	uq = dataset.variables['uq'][:,:];
	Uq = dataset.variables['Uq'][:,:];
	uQ = dataset.variables['uQ'][:,:];
	vq = dataset.variables['vq'][:,:];
	vQ = dataset.variables['vQ'][:,:];

	return EEF, uq, Uq,uQ, vq, vQ;

#=======================================================

def npyReadEEF_y0_components(npyFile):

	EEF_array = np.load(npyFile);	# Read the NETCDF file

	nn = np.shape(EEF_array)[0];

	EEF_north = EEF_array[:,0,0]; EEF_south = EEF_array[:,0,1];
	uq_north = EEF_array[:,1,0]; uq_south = EEF_array[:,1,1];
	Uq_north = EEF_array[:,2,0]; Uq_south = EEF_array[:,2,1];
	uQ_north = EEF_array[:,3,0]; uQ_south = EEF_array[:,3,1];
	vq_north = EEF_array[:,4,0]; vq_south = EEF_array[:,4,1];
	vQ_north = EEF_array[:,5,0]; vQ_south = EEF_array[:,5,1];

	EEF = EEF_north - EEF_south;
	uq = uq_north - uq_south;
	Uq = Uq_north - Uq_south;
	uQ = uQ_north - uQ_south;
	vq = vq_north - vq_south;
	vQ = vQ_north - vQ_south;

	return EEF, uq, Uq,uQ, vq, vQ;

#=======================================================

def npyReadEEF(npyFile):

	EEF_array = np.load(npyFile);	# Read the NETCDF file

	nn = np.shape(EEF_array)[0];

	EEF_north = EEF_array[:,0]; EEF_south = EEF_array[:,1];

	EEF = EEF_north - EEF_south;


	return EEF;


	

	
