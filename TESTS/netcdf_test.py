# netcdf_test.py
#==================================================================

import netCDF4 as nc
import numpy as np

#==================================================================

N = 100;
L = 1000;

A = np.zeros((N,N));

x = np.linspace(-L/2,L/2,N);
y = np.linspace(-L/2,L/2,N);

for j  in range(0,N):
	for i in range(0,N):
		A[j,i] = (y[j]**2 - x[i]**2) / L**2;


rootgrp = nc.Dataset("test.nc", "w", format="NETCDF4");

testgrp = rootgrp.createGroup('test');

xdim = testgrp.createDimension('xdim',N);
ydim = testgrp.createDimension('ydim',N);

xvar = testgrp.createVariable('xdim','f4',('xdim',));
yvar = testgrp.createVariable('ydim','f4',('ydim',));
Avar = testgrp.createVariable('A','f4',('xdim','ydim',));

xvar[:] = x;
yvar[:] = y;
Avar[:] = A;








