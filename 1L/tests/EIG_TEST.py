# EIG_TEST.py
#=======================================================

# This code finds the eigenmodes and eigenvalues for the 1-L shallow water system.

#====================================================

import sys

import numpy as np
import matplotlib.pyplot as plt

import itertools as it
from diagnostics import diff
import eigDiagnostics
import eigSolver

import output_read 

from inputFile import *

# 1L eigenmode solver
#====================================================
#====================================================
 
I = np.complex(0.0,1.0);

path = '/home/mike/Documents/GulfStream/RSW/DATA/1L/EIG/128/nu='+str(int(nu))+'/';

# An arbitrary selection of modes and weights.
k_sel = [74,4,5];				# Wavenumbers.
mode_sel = [1,2,3];			# Modes at each wavenumber.
theta_sel = [1.0+3.0*I,2.0+1.0*I,3.0-2.0*I];		# Weights of each mode.

a1,a2,a3,a4,b4,c1,c2,c3,c4 = eigSolver.EIG_COEFFICIENTS(Ro,Re,K_nd,f_nd,U0_nd,H0_nd,gamma_nd,dy_nd,N);

# eigTest1
def eigTest1(k_sel,mode_sel,theta_sel,K_nd,N,dim,path):
# This function takes a selection of eigenmodes (reads the ones saved), constructs a flow solution with this eigenmode basis,
# and then performs the eigenmode decomposition on this basis to ensure that the original selection of eigenmodes is returned.

	# Initialise a solution array. An arbitrary set of modes will be added to this.
	solution = np.zeros((dim,N),dtype=complex);	

	for i in range(0,len(k_sel)):
		k = K_nd[k_sel[i]];
		ncFile = path + 'RSW1L_Eigenmodes_k' + str(int(k)) + '_N129.nc';
		print('Reading from ' + ncFile + '...');
		val, vec, count = output_read.ncReadEigenmodes(ncFile);
		solution[:,k_sel[i]] = theta_sel[i] * vec[:,mode_sel[i]];

		ii = k_sel[i];
		k = K_nd[ii];
		print('k = ' + str(k));

		val, vec, count = output_read.ncReadEigenmodes(ncFile);

		Phi = solution[:,ii]

		theta = np.linalg.solve(vec,Phi);

		print(theta[mode_sel[i]]);

		# This test shows that the correct mode and weights are returned.

# eigTest2
def eigTest2(k_sel,mode_sel,K_nd,dy_nd,N,dim,path,a1,a2,a3,a4,b4,c1,c2,c3,c4):
# This test makes sure that eigenmodes actually solve the governing equations.
	
	for i in range(0,len(k_sel)):

		ii = k_sel[i];
		k = K_nd[ii];
		mode = mode_sel[i];

		ncFile = path + 'RSW1L_Eigenmodes_k' + str(int(k)) + '_N129.nc';
		print('Reading from ' + ncFile + '...');
		val, vec, count = output_read.ncReadEigenmodes(ncFile);
		
		# Extract the eigenvectors.
		u_vec, v_vec, eta_vec = eigDiagnostics.vec2vecs(vec,N,dim,BC);

		# Now we have to check that this solves the governing equations by substituting it back in.
		u_vec = u_vec[:,mode]; v_vec = v_vec[:,mode]; eta_vec = eta_vec[:,mode]; val = val[mode];
		
		#val = val * (2 * np.pi * I * Ro);

		u_y = diff(u_vec,2,0,dy_nd) * dy_nd;
		u_yy = diff(u_vec,2,0,dy_nd) * dy_nd;
	
		e1 = - val * u_vec;
		e2 = a1[:,ii] * u_vec[:];
		e3 = a2 * u_yy;
		e4 = a3 * v_vec;
		e5 = a4[ii] * eta_vec;
		e = e1+e2+e3+e4+e5
	
		plt.subplot(321);
		plt.plot(e1);
		plt.subplot(322);
		plt.plot(abs(e2));
		plt.subplot(323);
		plt.plot(e3);
		plt.subplot(324);
		plt.plot(e4);
		plt.subplot(325);
		plt.plot(e5);
		plt.subplot(326);
		plt.plot(e[10:N-10]);

		plt.show();

if __name__ == '__main__':

	if len(sys.argv) < 2:
		print("USAGE: {} 1|2".format(sys.argv[0]))
		sys.exit()
	if "1" == sys.argv[1]:
		eigTest1(k_sel,mode_sel,theta_sel,K_nd,N,dim,path);
	elif "2" == sys.argv[1]:
		eigTest2(k_sel,mode_sel,K_nd,dy_nd,N,dim,path,a1,a2,a3,a4,b4,c1,c2,c3,c4);
	else:
		print("USAGE: {} 1|2".format(sys.argv[0]))
		sys.exit()

