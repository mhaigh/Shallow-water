# EIG.py
#=======================================================

# This code finds the eigenmodes and eigenvalues for the 1-L shallow water system.

#====================================================

import sys

import numpy as np
import matplotlib.pyplot as plt
import itertools as it

from eig import eigDiagnostics, eigSolver
from core import diagnostics, PV, energy
from output.output import ncSaveEigenmodes

from inputFile import *

# 1L eigenmode solver
#====================================================
#====================================================

# Numbers to note:
# Dimensional forcing periods and corresponding 0-D omega:
# 50 days --> omega = 0.888...
# 60 days --> omega = 0.740740...
# 70 days --> omega = 0.634920...
 
I = np.complex(0.0,1.0);

# Define the coefficients required by the solver
a1,a2,a3,a4,b1,b4,c1,c2,c3,c4 = eigSolver.EIG_COEFFICIENTS2(Ro,Re,K_nd,f_nd,U0_nd,H0_nd,gamma_nd,dy_nd,N)
	
k_start = 3
k_end = 4
Nk = 8
#loop = it.chain(range(0,Nk+1),range(N-Nk-1,N));	##
loop = range(k_start,k_end)
loop = range(0,N)
for ii in loop:
	# Run the solver for the current k-value.
	k = K_nd[ii]	
	print('k = ' + str(k))
	if BC == 'NO-SLIP':
		val, u_vec, v_vec, h_vec = eigSolver.NO_SLIP_EIG(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,N,N2,ii,True)
	if BC == 'FREE-SLIP':
		val, vec = eigSolver.FREE_SLIP_EIG(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,N,N2,ii,False)
		#val, vec = eigSolver.FREE_SLIP_EIG2(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,N,N2,ii,False)
				
									
	#val = val / (2*np.pi*I*Ro)
	freq = np.real(val)
	period_days = T_adv / (freq * 24.0 * 3600.0)
	
	
	dim = np.size(val)
	
	# count = number of zero-crossings by the eigenvector 
	# i_count = set of indices ordering modes by their count
	#count, i_count = eigDiagnostics.orderEigenmodes(vec,val,x_nd,k,T_nd[ts],N,dim,BC);	
	#count, i_count = eigDiagnostics.orderEigenmodes2(vec,val,N,False);
	# orderEigenmodes2 works best.
	count = np.zeros((dim))
	i_count = np.ones((dim))
	# Order all relevant arrays
	#count = count[i_count];
	#vec = vec[:,i_count];
	#val = val[i_count];
	#period_days = period_days[i_count]

	p_sort = np.argsort(-np.abs(period_days))
	#print(period_days[p_sort]);
	
	# Before saving the modes, they need to be normalised by their energy.
	ENERGY = 1
	if ENERGY == 1:		
		E = np.zeros(dim)
		u_vec, v_vec, h_vec = eigDiagnostics.vec2vecs(vec,N,dim,BC)
		for wi in range(0,dim):
			#print(str(wi+1) + ' / ' + str(dim));	
			EE = energy.E_anomaly_EIG(u_vec[:,wi],v_vec[:,wi],h_vec[:,wi],H0_nd,U0_nd,Ro,y_nd,dy_nd)
			u_vec[:,wi], v_vec[:,wi], h_vec[:,wi] = u_vec[:,wi] / np.sqrt(EE), v_vec[:,wi] / np.sqrt(EE), h_vec[:,wi] / np.sqrt(EE)
		# Rebuild
		vec = eigDiagnostics.vecs2vec(u_vec,v_vec,h_vec,N,dim,BC)

	ncSaveEigenmodes(vec,val,count,y_nd,k,N,dim,BC)

#====================================================

if str(raw_input('continue? y or n: ')) == 'n':
	sys.exit();

#====================================================
 
u_vec, v_vec, h_vec = eigDiagnostics.vec2vecs(vec,N,dim,BC)

for ii in loop:

	k = K_nd[ii];

	#====================================================

	# By this point, eigenmodes have been calculated and ordered. The ordering algorithm may make mistakes,
	# however, so the set of modes can be manually updated with the remainder of the code.

	# 1. This first section runs through each mode, allowing to manually update the zero-crossings count.
	# To quit updating the vector's counts, type end.
	# We use the u vector as an example
	u = np.zeros((N,N),dtype=float);
	count_new = np.array(list(count));
	update_i = [];		# Used to store the set of wi indices that need updating.
	wiii = 0
	while wiii < dim:
		wii = dim-wiii-1;
		#wii = p_sort[wiii];
		print('i_count = ' + str(wii));
		u = eigDiagnostics.vec2field(u_vec[:,wii],val[wii],x_nd,k,N,T_nd[ts])
		print('count = ' + str(count[wii]));
		print('period = ' + str(period_days[wii]));
		plt.subplot(131);
		plt.contourf(u);
		plt.subplot(132);
		plt.plot(np.real(u_vec[:,wii]),y_nd);
		plt.ylim(-0.5,0.5);
		plt.subplot(133);
		plt.plot(np.imag(u_vec[:,wii]),y_nd)
		plt.ylim(-0.5,0.5);
		plt.show();

		# Can use these plots to check any errors, and update count accordingly;
		# comment out if not needed.

		count_new[wii], wii = eigDiagnostics.updateCount(count[wii],wii);	# Update the count, wii is set to high number if user wants to quit the algorithm.

		wiii += 1;

	#====================================================

	# 2. According to the updated count, the vectors are reordered.

	i_count_new = np.argsort(count_new);
	count = count_new[i_count_new];
	
	# Update the vectors & eigenvalues	
	vec = vec[:,i_count_new];
	val = val[i_count_new];

	u_vec, v_vec, h_vec = eigDiagnostics.vec2vecs(vec,N,dim,BC);		

	ncSaveEigenmodes(vec,val,count,y_nd,k,N,dim,BC);
	
	#====================================================
	
	# 3. The new vectors can be checked, again using u_vec as an example.

	u = np.zeros((N,N),dtype=float);
	for wi in range(0,0):
		for i in range(0,N):
			for j in range(0,N):
				u[j,i] = np.real(u_vec[j,wi] * np.exp(2 * np.pi * I * (k * x_nd[i])));
		print(count_new[wi]);
		u_abs = np.abs
		plt.subplot(121);
		plt.contourf(u);
		plt.subplot(122);
		plt.plot(np.abs(u_vec[:,wi]),y_nd);
		plt.ylim(-0.5,0.5);
		plt.show();

	#====================================================
	#====================================================

	# val is a set of dimensionless frequencies; it's easier to work in terms of dimensional periods.
	# Here we order dimensionalise val and define the index arrays that give orderings of val.
	freq = np.real(val);				# Frequencies
	period = 1. / freq;					# Dimensionless period
	growth = np.imag(val);				# Growth rates
	period_days = T_adv / (freq * 24. * 3600.);		# Periods corresponding to eigenvalues (days).
	i_freq = np.argsort(np.abs(freq));						# Indices of frequencies in ascending order.
	i_growth = np.argsort(growth);					# Indices of growth rates in ascending order.

	# We want to look at a eigenfunctions with eigenvalues within some frequency/period range, given by pmin, pmax.
	pmin = - 100;		# Min and max period in days
	pmax = - 40;
	# Some empty lists, to be added to and converted to numpy arrays.
	freq_set = [];
	freq_index = [];
	u_set = [];
	v_set = [];
	h_set = [];
	for wi in range(0,dim):
		#plt.plot(u_vec[:,wi]);
		#plt.show();
		if pmin <= period_days[wi] <= pmax:
		#if pmin <= period_days[wi] <= pmax:
			freq_index.append(wi);
			freq_set.append(freq[wi]);
			u_set.append(u_vec[:,wi]);
			v_set.append(v_vec[:,wi]);
			h_set.append(h_vec[:,wi]);
	u_set = np.array(u_set);
	v_set = np.array(v_set);
	h_set = np.array(h_set);
	# Note that using np.array reverses the indexing convention.

	# Now define u, v and h in (x,y)-space.
	Nf = len(freq_set);
	u = np.zeros((N,N,Nf));
	v = np.zeros((N,N,Nf));
	h = np.zeros((N,N,Nf));
	for wi in range(0,Nf):
		for i in range(0,N):
			for j in range(0,N):
				u[j,i,wi] = np.real(u_set[wi,j] * np.exp(2 * np.pi * I * (k * x_nd[i])));
				v[j,i,wi] = np.real(v_set[wi,j] * np.exp(2 * np.pi * I * (k * x_nd[i])));
				h[j,i,wi] = np.real(h_set[wi,j] * np.exp(2 * np.pi * I * (k * x_nd[i])));

		u_full = np.zeros((N,N));
		h_full = np.zeros((N,N));
		for i in range(0,N):
			u_full[:,i] = u[:,i,wi] + U0_nd[:];
			h_full[:,i] = h[:,i,wi] + H0_nd[:];
	

		plt.plot(y_nd,u_set[wi,:]);
		plt.show();


