# EIG_y0.py
#====================================================

# A code that calculates the eigenfunction decomposition for a given range of zonal wavenumbers k, for a differing y0.
# Looping through zonal wavenumbers in range [imin,imax], the structure of the code is as follows:
# 1. Calculate the set of eigenfunctions, order them according to the absolute value of the period (greatest period first).
# 2. Solve the forced 1L system N-n0 times, ONLY FOR THE CURRENT ZONAL WAVENUMBER. This requires using a variation of the original solver.
# ... For each solution, move the forcing location, y0, by 1 (or more) gridpoints, storing the solution for each y0.
# 3. Express each forced solution, a function of y only, in terms of the eigenfunctions, storing the decomposition weights each time.
# The aim of this is to observe a gradual dependency change of the forced solutions on the eigenfunctions, i.e. see how the weights change as y0 changes.

#====================================================

import numpy as np
import eigSolver
import matplotlib.pyplot as plt
import eigDiagnostics
import diagnostics
import solver
import solverAlt
import forcing_1L

from inputFile_1L import *

#====================================================

# The coefficients used in the eigenfunction solver, and the forced 1L solver.
a1,a2,a3,a4,b1,b4,c1,c2,c3,c4 = eigSolver.EIG_COEFFICIENTS(Ro,Re,K_nd,f_nd,U0_nd,H0_nd,gamma_nd,dy_nd,N)
a1,a2,a3,a4,b4,c1,c2,c3,c4 = solver.SOLVER_COEFFICIENTS(Ro,Re,K_nd,f_nd,U0_nd,H0_nd,omega_nd,gamma_nd,dy_nd,N);
if BC == 'FREE-SLIP':
	uBC_eig, etaBC_eig = eigSolver.BC_COEFFICIENTS(Ro,Re,f_nd,H0_nd,dy_nd,N);
	uBC, etaBC = solver.BC_COEFFICIENTS(Ro,Re,f_nd,H0_nd,dy_nd,N);

# The set of forcing locations, y0. Some choices:
opt = 2;
# Opt = 1: all gridpoints, ommiting a values at each boundary.
# Opt = 2: a much smaller set, only one or two y0 vals north/south of y0 = 0.
if opt == 1:
	a = 5;
	y0_set = y[a:N-1-a];
	y0_nd_set = y_nd[a:N-1-a];
	y0N = np.size(y0_set);
elif opt == 2:
	y0_set = np.array([-Ly/4, 0, Ly/4]);
	y0_nd_set = y0_set / Ly;
	y0N = np.size(y0_set);
else:
	print('Error: opt must be 1 or 2');
	
# Initialise the set of weights, one vector for each iteration through y0_set
theta = np.zeros((dim,y0N),dtype=complex);
dom_index = np.zeros((dim,y0N),dtype=int);
solution = np.zeros((dim,y0N),dtype=complex);

# Set the range of zonal wavenumbers
imin = 1; imax = 5;
for i in range(imin,imax):

	# 1. Calculate the set of eigenfunctions for wavenumber i.
	if BC == 'NO-SLIP':
		val, vec = eigSolver.NO_SLIP_EIG(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,N,N2,i,False);
	elif BC == 'FREE-SLIP':
		val, vec = eigSolver.FREE_SLIP_EIG(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,uBC_eig,etaBC_eig,N,N2,i,False);
	else:
		print('ERROR: choose BCs');

	# Order the eigenfunctions according to period, longest first.
	wi_period = np.argsort(np.abs(np.real(val))); 
	vec[:,:] = vec[:,wi_period];
	val = val[wi_period];

	# For reference, if needed.
	freq = np.real(val);
	per_days = T_adv / (freq * 24. * 3600.);

	# For the wavenumber K_nd[i]
	for yi in range(0,y0N):
		y0 = y0_set[yi];
		y0_nd = y0_nd_set[yi];

		# Run the solver
		#====================================================

		# Forcing
		F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing_1L.Forcing(x,y,K,y0,r0,N,FORCE,AmpF,g,f,f0,U,L,dx,dy);

		# Solver
		if BC == 'NO-SLIP':
			solution[:,yi] = solverAlt.NO_SLIP_SOLVER(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,N,N2,i);
		if BC == 'FREE-SLIP':
			solution[:,yi] = solverAlt.FREE_SLIP_SOLVER(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,uBC,etaBC,Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,N,N2,i);
		
		# Project onto vec & calculate weights
		#====================================================
	
		theta[:,yi] = eigSolver.eigDecomp(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,N,N2,i,BC,vec,solution[:,yi]);

		dom_index[:,yi] = np.argsort(-(np.abs(theta[:,yi]))**2);	# The indices of the modes, ordered by 'dominance'.

		#plt.plot(vec[0:N,dom_index[0]],y_nd);
		#plt.show();

	if opt == 2:
		print('South: ' + str(dom_index[0,0]) + ', ' + str(per_days[dom_index[0,0]]) + '; ' + str(dom_index[1,0]) + ', ' + str(per_days[dom_index[1,0]]));
		print('Center: ' + str(dom_index[0,1]) + ', ' + str(per_days[dom_index[0,1]]) + '; ' + str(dom_index[1,1]) + ', ' + str(per_days[dom_index[1,1]]));
		print('North: ' + str(dom_index[0,2]) + ', ' + str(per_days[dom_index[0,2]]) + '; ' + str(dom_index[1,2]) + ', ' + str(per_days[dom_index[1,2]]));

		uSouth0 = np.real(theta[dom_index[0,0],0] * vec[0:N,dom_index[0,0]]);
		uSouth1 = np.real(theta[dom_index[1,0],0] * vec[0:N,dom_index[1,0]]);
		uCenter0 = np.real(theta[dom_index[0,1],1] * vec[0:N,dom_index[0,1]]);
		uCenter1 = np.real(theta[dom_index[1,1],1] * vec[0:N,dom_index[1,1]]);
		uNorth0 = np.real(theta[dom_index[0,2],2] * vec[0:N,dom_index[0,2]]);
		uNorth1 = np.real(theta[dom_index[1,2],2] * vec[0:N,dom_index[1,2]]);

		plt.figure(1);
		plt.subplot(231);
		plt.plot(uSouth0,y_nd,label='dom1');
		plt.plot(uSouth1,y_nd,label='dom2');
		plt.legend();
		plt.ylabel('Dom Modes');
		plt.title('South')
		plt.subplot(232);
		plt.plot(uCenter0,y_nd);
		plt.plot(uCenter1,y_nd);
		plt.title('Center');
		plt.subplot(233);
		plt.plot(uNorth0,y_nd);
		plt.plot(uNorth1,y_nd);
		plt.title('North');
		plt.subplot(234);
		plt.plot(np.real(solution[0:N,0]),y_nd);
		plt.ylabel('Forced Sol');
		plt.subplot(235);
		plt.plot(np.real(solution[0:N,1]),y_nd);
		plt.subplot(236);
		plt.plot(np.real(solution[0:N,2]),y_nd);
		plt.show();
	

	#plt.plot(np.transpose(dom_index[0:2,:]));
	#plt.show();			
	#plt.plot(np.abs(theta[2,:]));
	#plt.show();
