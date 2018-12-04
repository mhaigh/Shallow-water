# eigDiagnostics.py
# File containing functions to be called by the master script EIG.py.
#====================================================

import numpy as np
import matplotlib.pyplot as plt

from diagnostics import diff
from diagnostics import extend

#====================================================

# The potential vorticity
# PV
def PV(u,v,h,H_nd,U_nd,f_nd,dx_nd,dy_nd,N):
# Takes input u, v, h which are the eigenvectors of the 2-L SW system, projected onto two dimensions, 
# i.e. u1 = u1_vec(y) * exp(i*k*x). Same code works for either layer.

	u_full = np.zeros((N,N));
	eta_full = np.zeros((N,N));
	for i in range(0,N):
		u_full[:,i] = u[:,i] + U_nd[:];
		eta_full[:,i] = eta[:,i] + H_nd[:];

	# Relative vorticity
	RV_full = diff(v,1,1,dx_nd) - diff(u_full,0,0,dy_nd);
	RV_prime = diff(v,1,1,dx_nd) - diff(u,0,0,dy_nd);
	RV_BG = - diff(U_nd,2,0,dy_nd);

	# Potential vorticity
	PV_BG = (f_nd + RV_BG) / H_nd; 
	PV_full = np.zeros((N,N));
	PV_prime = np.zeros((N,N));
	for i in range(0,N):
		PV_full[:,i] = (f_nd[:] + RV_full[:,i]) / eta_full[:,i];
		PV_prime[:,i] = PV_full[:,i] - PV_BG[:];

	return PV_full, PV_prime;

#====================================================

# footprint
#def footprint(u,v,)


#====================================================

# eigPlot
def eigPlot(u,v,eta,PV,x_nd,y_nd):
	
	u = extend(u);
	v = extend(v);
	eta = extend(eta);
	PV = extend(PV);

	plt.subplot(221)
	plt.contourf(x_nd,y_nd,u);
	plt.xticks((-1./2,0,1./2));
	plt.yticks((-1./2,0,1./2));	
	plt.xlabel('x');
	plt.ylabel('y');
	plt.colorbar();
	plt.subplot(222)
	plt.contourf(x_nd,y_nd,v);
	plt.xticks((-1./2,0,1./2));
	plt.yticks((-1./2,0,1./2));	
	plt.xlabel('x');
	plt.ylabel('y');
	plt.colorbar();
	plt.subplot(223)
	plt.contourf(x_nd,y_nd,eta);
	plt.xticks((-1./2,0,1./2));
	plt.yticks((-1./2,0,1./2));	
	plt.xlabel('x');
	plt.ylabel('y');
	plt.colorbar();
	plt.subplot(224)
	plt.contourf(x_nd,y_nd,PV);
	plt.xticks((-1./2,0,1./2));
	plt.yticks((-1./2,0,1./2));	
	plt.xlabel('x');
	plt.ylabel('y');
	plt.colorbar();
	plt.show();

	
	
