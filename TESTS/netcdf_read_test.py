# netcdf_read_test.py
#==================================================================

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

#==================================================================

I = np.complex(0,1);

path = '/home/mike/Documents/GulfStream/RSW/DATA/1L/EIG/'
ncFile = path + 'RSW1L_Eigenmodes_k2_N128.nc';

Eigenmodes = nc.Dataset(ncFile);	# Read the NETCDF file

u_vec = Eigenmodes.variables['u_vec'][:];
print(np.shape(u_vec));
u_vec = u_vec[:,:,0] + I * u_vec[:,:,1];
print(np.shape(u_vec));

plt.subplot(131);
plt.plot(np.real(u_vec[:,0]));
plt.subplot(132);
plt.plot(np.imag(u_vec[:,0]));
plt.subplot(133);
plt.plot(np.abs(u_vec[:,0]));
plt.show();

