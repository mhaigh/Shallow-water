# EIG.py
#=======================================================

# This code finds the eigenmodes and eigenvalues for the 2-L shallow water system.

#====================================================

import numpy as np
import matplotlib.pyplot as plt

import diagnostics
import eigDiagnostics
import PV
import eigSolver

from inputFile_2L import *

# 2L eigenmode solver
#====================================================
#====================================================

# Numbers to note:
# Dimensional forcing periods and corresponding 0-D omega:
# 50 days --> omega = 0.888...
# 60 days --> omega = 0.740740...
# 70 days --> omega = 0.634920...
 
I = np.complex(0.0,1.0);

a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,d1,d3,d4,e4,f1,f2,f3,f4 = eigSolver.EIG_COEFFICIENTS(Ro,Re,K_nd,f_nd,U1_nd,U2_nd,H1_nd,H2_nd,rho1_nd,rho2_nd,gamma_nd,dy_nd,N)

k_start = 3;
k_end = k_start + 1;
for ii in range(k_start,k_end):
	# Run the solver for the current k-value.
	k = K_nd[ii];
	if BC == 'NO-SLIP':
		val, u1_vec, u2_vec, v1_vec, v2_vec, h1_vec, h2_vec = eigSolver.NO_SLIP_EIG(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,d1,d3,d4,e4,f1,f2,f3,f4,N,N2,ii,True);
#NO_SLIP_EIG(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,N,N2,ii,True);
	if BC == 'FREE-SLIP':
		val, u1_vec, u2_vec, v1_vec, v2_vec, h1_vec, h2_vec = eigSolver.FREE_SLIP_EIG(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,d1,d3,d4,e4,f1,f2,f3,f4,N,N2,ii,True);
	dim = np.size(val);

	# val is a set of dimensionless frequencies; it's easier to work in terms of dimensional periods.
	# Here we order dimensionalise val and define the index arrays that give orderings of val.
	freq = np.real(val);				# Frequencies
	period = 1. / freq;					# Dimensionless period
	growth = np.imag(val);				# Growth rates
	period_days = T_adv / (freq * 24 * 3600);		# Periods corresponding to eigenvalues (days).
	i_freq = np.argsort(freq);						# Indices of frequencies in ascending order.
	i_growth = np.argsort(growth);					# Indices of growth rates in ascending order.

	# We want to look at a eigenfunctions with eigenvalues within some frequency/period range, given by pmin, pmax.
	pmin = -100;		# Min and max period in days
	pmax = -40;
	# Some empty lists, to be added to and converted to numpy arrays.
	freq_set = [];
	freq_index = [];
	u1_set = [];	u2_set = [];
	v1_set = [];	v2_set = [];
	h1_set = [];	h2_set = [];
	for wi in range(0,dim):
		#plt.plot(u_vec[:,wi]);
		#plt.show();
		if pmin <= period_days[wi] <= pmax:
		#if pmin <= period_days[wi] <= pmax:
			freq_index.append(wi);
			freq_set.append(freq[wi]);
			u1_set.append(u1_vec[:,wi]);
			u2_set.append(u2_vec[:,wi]);
			v1_set.append(v1_vec[:,wi]);
			v2_set.append(v2_vec[:,wi]);
			h1_set.append(h1_vec[:,wi]);
			h2_set.append(h2_vec[:,wi]);
	u1_set = np.array(u1_set);
	u2_set = np.array(u2_set);
	v1_set = np.array(v1_set);
	v2_set = np.array(v2_set);
	h1_set = np.array(h1_set);
	h2_set = np.array(h2_set);
	# Note that using np.array reverses the indexing convention.

	# Now define u, v and eta in (x,y)-space.
	Nf = len(freq_set)
	u1 = np.zeros((N,N,Nf));
	u2 = np.zeros((N,N,Nf));
	v1 = np.zeros((N,N,Nf));
	v2 = np.zeros((N,N,Nf));
	h1 = np.zeros((N,N,Nf));
	h2 = np.zeros((N,N,Nf));
	for wi in range(0,Nf):
		for i in range(0,N):
			for j in range(0,N):
				u1[j,i,wi] = np.real(u1_set[wi,j] * np.exp(2 * np.pi * I * (k * x_nd[i])));
				u2[j,i,wi] = np.real(u1_set[wi,j] * np.exp(2 * np.pi * I * (k * x_nd[i])));				
				v1[j,i,wi] = np.real(v1_set[wi,j] * np.exp(2 * np.pi * I * (k * x_nd[i])));
				v2[j,i,wi] = np.real(u1_set[wi,j] * np.exp(2 * np.pi * I * (k * x_nd[i])));
				h1[j,i,wi] = np.real(h1_set[wi,j] * np.exp(2 * np.pi * I * (k * x_nd[i])));
				h2[j,i,wi] = np.real(u1_set[wi,j] * np.exp(2 * np.pi * I * (k * x_nd[i])));

		#PV_full, PV_prime = eigDiagnostics.PV(u[:,:,wi],v[:,:,wi],eta[:,:,wi],H0_nd,U0_nd,f_nd,dx_nd,dy_nd,N);
		#print(period_days[freq_index[wi]]);
	
		plt.contourf(u1[:,:,wi]);
		plt.show();
		
		eigDiagnostics.eigPlot(u[:,:,wi],v[:,:,wi],eta[:,:,wi],PV_prime,x_nd,y_nd);
	












	
