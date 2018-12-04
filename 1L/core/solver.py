# solver.py
#=======================================================

# Contains a set of functions to be called by RSW_1L_visc.py. 
# They constitute the solvers (for two choices of BCs) as well as a function that defines all parameters of the linear solver.

#=======================================================

import numpy as np
from diagnostics import diff

#=======================================================

#=======================================================

# SOLVER_COEFFICIENTS
#=======================================================
def SOLVER_COEFFICIENTS(Ro,Re,K_nd,f_nd,U0_nd,H0_nd,omega_nd,gamma_nd,dy_nd,N):
# Here we also divide coeficients of derivatives by the relevant space-step: 
# 2*dy_nd for first-order derivatives, dy_nd**2 for second-order derivatives.

	I = np.complex(0,1);		# Define I = sqrt(-1)

	if Re == None:
		Ro_Re = 0;
	else:
		Ro_Re = Ro / Re;

	# Coefficients with no k- or y-dependence
	a2 = - Ro_Re / (dy_nd**2);	# Note: b3=a2, so instead of defining b3, we just use a2 in its place, saving time.
	b4 = 1. / (2. * dy_nd);

	# Coefficients with k-dependence only
	a4 = np.zeros(N,dtype=complex);		
	for i in range(0,N):
		a4[i] = 2. * np.pi * I * K_nd[i];

	# Coefficients with y-dependence only

	a3 = Ro * diff(U0_nd,2,0,dy_nd) - f_nd;		# For uniform U0_nd, the first term is zero	

	c2 = diff(H0_nd,2,0,dy_nd);					# For zero BG flow, H0_nd=Hflat=const, i.e. c2=0
	
	c3 = H0_nd / (2. * dy_nd);

	# a3, c2 and c3 have length N, so when used in defining u and v equations in the matrix below,
	# we must add 1 to each index in order to skip out the dead gridpoint.

	# Note that b1=f_nd so will not be defined, but we use f_nd directly.

	# Coefficients dependent on k and y
	delta = np.zeros((N,N));				# Appears in many of the coeffs, essentially a space-saver.
	a1 = np.zeros((N,N),dtype=complex);		# Note: b2=a1
	c1 = np.zeros((N,N),dtype=complex);
	for i in range(0,N):
		for j in range(0,N):
			delta[j,i] = 2. * np.pi * (omega_nd + U0_nd[j] * K_nd[i]);
			a1[j,i] = I * delta[j,i] * Ro + 4. * np.pi**2 * K_nd[i]**2 * Ro_Re + gamma_nd;
			c1[j,i] = 2. * np.pi * I * K_nd[i] * H0_nd[j];

	c4 = I * delta;

	return a1,a2,a3,a4,b4,c1,c2,c3,c4;

#=======================================================

# BC_COEFFICIENTS
#=======================================================
def BC_COEFFICIENTS(Ro,Re,f_nd,H0_nd,dy_nd,N):
# Some extra coefficients that are required by the free-slip solver in order to impose
# the du/dy=0 boundary conditions. These terms are not needed in the no-slip solver.

	if Re == None:
		Ro_Re = 0;
	else:
		Ro_Re = Ro / Re;

	uBC = 2 * Ro_Re / dy_nd**2;		# The extra term required for the u equation BCs.
	
	etaBC = np.zeros(2);
	if Re != None:
		etaBC[0] = f_nd[0] * H0_nd[0] * dy_nd * Re / (2 * Ro);		# The extra term required for the eta equation BCs.
		etaBC[1] = f_nd[N-1] * H0_nd[N-1] * dy_nd * Re / (2 * Ro);	

	return uBC, etaBC;

#=======================================================

# NO_SLIP_SOLVER 
#=======================================================
def NO_SLIP_SOLVER(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,N,N2):
# Called by RSW_1L.py if BC = 'NO-SLIP'.

	dim = 2 * N2 + N;
	#print(dim);

	A = np.zeros((dim,dim),dtype=complex);	# For the no-slip, no-normal flow BC.
	# eta has N gridpoints in y, whereas u and v have N2=N-2, after removing the two 'dead' gridpoints.
	# We primarily consider forcing away from the boundaries so that it's okay applying no-slip BCs.

	# Initialise the forcing.
	F = np.zeros((dim),dtype=complex);

	solution = np.zeros((dim,N),dtype=complex);

	# Start and end indices of u, v and eta.
	us = 0;  ue = N2-1;
	vs = N2;  ve = 2*N2-1;
	hs = 2*N2;  he = 2*N2+N-1; 

	for i in range(0,N):
		#print(i);

		# First the boundary terms. Some of these could be defined in the upcoming loop,
		# but for simplicity we define all boundary terms here.
	
		# u equation BCs
		# South
		A[us,us] = a1[1,i] - 2. * a2;	# u[1]
		A[us,us+1] = a2;				# u[2]
		A[us,vs] = a3[1];				# v[1]
		A[us,hs+1] = a4[i];				# eta[1] (eta[0] lies at i=2*N2 in A)
		# North
		A[ue,ue] = a1[N2,i] - 2. * a2;	# u[N2]=u[N-2]
		A[ue,ue-1] = a2;				# u[N2-1]=u[N-3]
		A[ue,ve] = a3[N2];				# v[N2]=v[N-2]
		A[ue,he-1] = a4[i];				# eta[N2] (eta[N2] lies at i=3*N2 in A)

		# v equation BCs
		# South
		A[vs,us] = f_nd[1];					# u[1]
		A[vs,vs] = a1[1,i] - 2. * a2		# v[1]
		A[vs,vs+1] = a2;					# v[2]
		A[vs,hs] = - b4;					# eta[0] 
		A[vs,hs+2] = b4;					# eta[2]
		# North
		A[ve,ue] = f_nd[N2];				# u[N-2] 
		A[ve,ve] = a1[N2,i] - 2. * a2;		# v[N-2]
		A[ve,ve-1] = a2;					# v[N-3]
		A[ve,he] = b4;						# eta[N-1]
		A[ve,he-2] = - b4;					# eta[N-3]

		# eta equation BCs (Here we have to define BCs at j=2*N2,2*N2+1,3*N2+1,3*N2)
		# South
		A[hs,vs] = 2. * c3[0]			# v[1] (factor of 2 because we use one-sided FD for v_y at the boundaries)
		A[hs,hs] = c4[0,i];				# eta[0]
		# South + 1
		A[hs+1,us] = c1[1,i];			# u[1]
		A[hs+1,vs] = c2[1];				# v[1]
		A[hs+1,vs+1] = c3[1];			# v[2] (factor of 1 here, back to using centered FD)
		A[hs+1,hs+1] = c4[1,i];			# eta[1]
		# North
		A[he,ve] = - 2. * c3[N2+1];		# v[N-2] (factor of 2 because we use one-sided FD for v_y at the boundaries)
		A[he,he] = c4[N-1,i];			# eta[N-1]
		# North - 1
		A[he-1,ue] = c1[N2,i];  		# u[N-2]
		A[he-1,ve] = c2[N2];			# v[N-2]
		A[he-1,ve-1] = - c3[N2];		# v[N-3] (factor of 1 here, back to using centered FD)
		A[he-1,he-1] = c4[N2,i];		# eta[N-2]

		# Now the inner values - two loops required: one for u,v, and one for eta
		for j in range(1,N2-1):
			# u equation
			A[j,j] = a1[j+1,i] - 2. * a2;			# u[j+1]
			A[j,j-1] = a2;							# u[j]
			A[j,j+1] = a2;							# u[j+2]
			A[j,vs+j] = a3[j+1];					# v[j+1]
			A[j,hs+j+1] = a4[i];					# eta[j+1] (must add 1 to each eta index to skip out eta[0])
			
			# v equation
			A[vs+j,j] = f_nd[j+1];					# u[j+1]
			A[vs+j,vs+j] = a1[j+1,i] - 2. * a2;		# v[j+1]
			A[vs+j,vs+j-1] = a2;					# v[j]
			A[vs+j,vs+j+1] = a2;					# v[j+2]
			A[vs+j,hs+j] = - b4;					# eta[j]
			A[vs+j,hs+j+2] = b4;					# eta[j+2]

		for j in range(2,N2):
			# eta equation
			A[hs+j,j-1] = c1[j,i];		# u[j] (the J=j-1 index of A(,J) corresponds to u[j], u without the deadpoints)
			A[hs+j,vs+j-1] = c2[j];		# v[j]
			A[hs+j,vs+j] = c3[j];			# v[j+1]
			A[hs+j,vs+j-2] = - c3[j];		# v[j-1]
			A[hs+j,hs+j] = c4[j,i];		# eta[j]
	
		for j in range(0,N2):	
			F[j] = Ftilde1_nd[j+1,i];		# Have to add 1 to y-index to skip the first 'dead' gridpoint.
			F[vs+j] = Ftilde2_nd[j+1,i];
		for j in range(0,N):
			F[hs+j] = Ftilde3_nd[j,i];	

		solution[:,i] = np.linalg.solve(A,F);

		#import matplotlib.pyplot as plt
		#plt.plot(solution[0:N,i]);
		#plt.show();

	return solution;

#=======================================================

# NO_SLIP_SOLVER 
#=======================================================
def NO_SLIP_SOLVER2(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,N,N2):
# Called by RSW_1L.py if BC = 'NO-SLIP'.
# At the boundaries, all perturbations are zero.

	dim = 3 * N2;
	#print(dim);

	A = np.zeros((dim,dim),dtype=complex);	# For the no-slip, no-normal flow BC.
	# eta has N gridpoints in y, whereas u and v have N2=N-2, after removing the two 'dead' gridpoints.
	# We primarily consider forcing away from the boundaries so that it's okay applying no-slip BCs.

	# Initialise the forcing.
	F = np.zeros((dim),dtype=complex);

	solution = np.zeros((dim,N),dtype=complex);

	for i in range(0,N):
		#print(i);

		# First the boundary terms. Some of these could be defined in the upcoming loop,
		# but for simplicity we define all boundary terms here.
	
		# u equation BCs
		# South
		A[0,0] = a1[1,i] - 2. * a2;			# u[1]
		A[0,1] = a2;						# u[2]
		A[0,N2] = a3[1];					# v[1]
		A[0,2*N2] = a4[i];					# eta[1] (eta[0] lies at i=2*N2 in A)
		# North
		A[N2-1,N2-1] = a1[N2,i] - 2. * a2;	# u[N2]=u[N-2]
		A[N2-1,N2-2] = a2;					# u[N2-1]=u[N-3]
		A[N2-1,2*N2-1] = a3[N2];			# v[N2]=v[N-2]
		A[N2-1,3*N2-1] = a4[i];				# eta[N2] (eta[N2] lies at i=3*N2 in A)

		# v equation BCs
		# South
		A[N2,0] = f_nd[1];					# u[1]
		A[N2,N2] = a1[1,i] - 2. * a2		# v[1]
		A[N2,N2+1] = a2;					# v[2]
		A[N2,2*N2+1] = b4;					# eta[2]
		# North
		A[2*N2-1,N2-1] = f_nd[N2];					# u[N-2] 
		A[2*N2-1,2*N2-1] = a1[N2,i] - 2. * a2;		# v[N-2]
		A[2*N2-1,2*N2-2] = a2;						# v[N-3]
		A[2*N2-1,3*N2-2] = - b4;					# eta[N-3]

		# eta equation BCs (Here we have to define BCs at j=2*N2,2*N2+1,3*N2+1,3*N2)
		# South
		A[2*N2,0] = c1[1,i];					# u[1]
		A[2*N2,N2] = c2[1];						# v[1]
		A[2*N2,N2+1] = c3[1];					# v[2] (factor of 1 here, back to using centered FD)
		A[2*N2,2*N2] = c4[1,i];					# eta[1]
		# North
		A[3*N2-1,N2-1] = c1[N2,i];  				# u[N-2]
		A[3*N2-1,2*N2-1] = c2[N2];				# v[N-2]
		A[3*N2-1,2*N2-2] = - c3[N2];				# v[N-3] (factor of 1 here, back to using centered FD)
		A[3*N2-1,3*N2-1] = c4[N2,i];				# eta[N-2]

		# Now the inner values - two loops required: one for u,v, and one for eta
		for j in range(1,N2-1):
			# u equation
			A[j,j] = a1[j+1,i] - 2. * a2;			# u[j+1]
			A[j,j-1] = a2;							# u[j]
			A[j,j+1] = a2;							# u[j+2]
			A[j,N2+j] = a3[j+1];					# v[j]
			A[j,2*N2+j] = a4[i];					# eta[j] (must add 1 to each eta index to skip out eta[0])
			
			# v equation
			A[N2+j,j] = f_nd[j+1];					# u[j+1]
			A[N2+j,N2+j] = a1[j+1,i] - 2. * a2;		# v[j+1]
			A[N2+j,N2+j-1] = a2;					# v[j]
			A[N2+j,N2+j+1] = a2;					# v[j+2]
			A[N2+j,2*N2+j-1] = - b4;					# eta[j]
			A[N2+j,2*N2+j+1] = b4;					# eta[j+2]

			# eta equation
			A[2*N2+j,j] = c1[j+1,i];			# u[j] (the J=j-1 index of A(,J) corresponds to u[j], u without the deadpoints)
			A[2*N2+j,N2+j] = c2[j+1];			# v[j]
			A[2*N2+j,N2+j+1] = c3[j+1];		# v[j+1]
			A[2*N2+j,N2+j-1] = - c3[j+1];		# v[j-1]
			A[2*N2+j,2*N2+j] = c4[j+1,i];		# eta[j]
	
		for j in range(0,N2):	
			F[j] = Ftilde1_nd[j+1,i];		# Have to add 1 to y-index to skip the first 'dead' gridpoint.
			F[N2+j] = Ftilde2_nd[j+1,i];
			F[2*N2+j] = Ftilde3_nd[j+1,i];	

		solution[:,i] = np.linalg.solve(A,F);

		#import matplotlib.pyplot as plt
		#plt.plot(solution[0:N,i]);
		#plt.show();

	return solution;

#=======================================================

### This is the solver used for the paper ###

# FREE_SLIP_SOLVER
#=======================================================
def FREE_SLIP_SOLVER(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,N,N2):
# Called by RSW_1L.py if BC = 'FREE-SLIP'.
# This is the better performing FREE-SLIP option.
# This solver assumes two things: 1. u[-1] = 0 and 2. one-sided fd approximation of v at the boundaries.
# This assumption can be more precisely motivated. u_y=v=v_yy=0.

	dim = N2 + 2 * N;
	#print(dim);
	
	A = np.zeros((dim,dim),dtype=complex);	# For the free-slip, no-normal flow BC.

	# Initialise the forcing.
	F = np.zeros((dim),dtype=complex);

	# Initialise the solution.
	solution = np.zeros((dim,N),dtype=complex);
	
	us = 0;  ue = N-1;
	vs = N;  ve = N+N2-1;
	hs = N+N2;  he = 2*N+N2-1;

	for i in range(0,N):
		#print(i);
				
		# First the boundary terms. Some of these could be defined in the upcoming loop,
		# but for simplicity we define all boundary terms here.
	
		# u equation BCs
		# South
		A[us,us] = a1[0,i] - 2 * a2;	# u[0]		# See documentation for reasoning behind BCs here
		A[us,us+1] = 2. * a2;			# u[1]
		A[us,hs] = a4[i];				# eta[0] 
		# North
		A[ue,ue] = a1[N-1,i]- 2 * a2;	# u[N-1]
		A[ue,ue-1] = 2. * a2;			# u[N-2]
		A[ue,he] = a4[i];				# eta[N-1]

		# v equation BCs
		# South
		A[vs,us+1] = f_nd[1];			# u[1]
		A[vs,vs] = a1[1,i] - 2. * a2;	# v[1]
		A[vs,vs+1] = a2;				# v[2]
		A[vs,hs] = - b4;				# eta[0] 	
		A[vs,hs+2] = b4;				# eta[2]
		# North
		A[ve,ue-1] = f_nd[N2];			# u[N-2] 
		A[ve,ve] = a1[N2,i] - 2. * a2;	# v[N-2]
		A[ve,ve-1] = a2;				# v[N-3]
		A[ve,he] = b4;					# eta[N-1]
		A[ve,he-2] = - b4;				# eta[N-3]

		# eta equation BCs (Here we have to define BCs at j=N+N2,N+N2+1,3*N-3,3*N-4)
		# South
		A[hs,us] = c1[0,i];			# u[0]		# Again, see documentation for BC reasoning
		A[hs,vs] = 2. * c3[0];		# v[1] (factor of 2 because we use one-sided FD for v_y at the boundaries. This comes from the the BC v_yy = 0.)
		A[hs,hs] = c4[0,i];			# eta[0]
		# South + 1
		A[hs+1,us+1] = c1[1,i];		# u[1]
		A[hs+1,vs] = c2[1];			# v[1]
		A[hs+1,vs+1] = c3[1];		# v[2] (factor of 1 here, back to using centered FD)
		A[hs+1,hs+1] = c4[1,i];		# eta[1]
		# North
		A[he,ue] = c1[N-1,i];		# u[N-1]
		A[he,ve] = - 2. * c3[N-1];	# v[N-2] (factor of 2 because we use one-sided FD for v_y at the boundaries)
		A[he,he] = c4[N-1,i];		# eta[N-1]
		# North - 1
		A[he-1,ue-1] = c1[N2,i];  	# u[N-2]
		A[he-1,ve] = c2[N2];		# v[N-2]
		A[he-1,ve-1] = - c3[N2];	# v[N-3] (factor of 1 here, back to using centered FD)
		A[he-1,he-1] = c4[N2,i];	# eta[N-2]

		# Now the inner values - two loops required: one for u,v, and one for eta
		for j in range(1,N-1):
			# u equation
			A[j,j] = a1[j,i] - 2. * a2;		# u[j+1]
			A[j,j-1] = a2;					# u[j]
			A[j,j+1] = a2;					# u[j+2]
			A[j,vs+j-1] = a3[j];			# v[j]
			A[j,hs+j] = a4[i];				# eta[j] 
			
		for j in range(1,N2-1):
			# v equation
			A[vs+j,j+1] = f_nd[j+1];			# u[j+1]
			A[vs+j,vs+j] = a1[j+1,i] - 2. * a2;	# v[j+1]
			A[vs+j,vs+j-1] = a2;				# v[j-1]
			A[vs+j,vs+j+1] = a2;				# v[j+1]
			A[vs+j,hs+j] = - b4;				# eta[j-1]
			A[vs+j,hs+j+2] = b4;				# eta[j+1]

		for j in range(2,N-2):
			# eta equation
			A[hs+j,j] = c1[j,i];		# u[j] 
			A[hs+j,vs+j-1] = c2[j];		# v[j]
			A[hs+j,vs+j] = c3[j];		# v[j+1]
			A[hs+j,vs+j-2] = - c3[j];	# v[j-1]
			A[hs+j,hs+j] = c4[j,i];		# eta[j]
	
		# Now define the forcing from the forcing input file.
		for j in range(0,N2):
			F[vs+j] = Ftilde2_nd[j+1,i];
		for j in range(0,N):
			F[j] = Ftilde1_nd[j,i];
			F[hs+j] = Ftilde3_nd[j,i];
			
		solution[:,i] = np.linalg.solve(A,F);
		
		#import matplotlib.pyplot as plt
		#plt.plot(solution[2*N:3*N,i]);
		#plt.show();
	
	return solution;

#=======================================================

# FREE_SLIP_SOLVER2
#=======================================================
def FREE_SLIP_SOLVER2(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,uBC,etaBC,Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,N,N2):
# Called by RSW_1L.py if BC = 'FREE-SLIP'.
# Imposes that v=v_yy=u_y=0.

	dim = N2 + 2 * N;
	#print(dim);

	A = np.zeros((dim,dim),dtype=complex);	# For the free-slip, no-normal flow BC.

	# Initialise the forcing.
	F = np.zeros((dim),dtype=complex);

	# Initialise the solution.
	solution = np.zeros((dim,N),dtype=complex);

	for i in range(0,N):
		#print(i);
				
		# First the boundary terms. Some of these could be defined in the upcoming loop,
		# but for simplicity we define all boundary terms here.
	
		# u equation BCs
		# South
		A[0,0] = a1[0,i];				# u[0]		# See documentation for reasoning behind BCs here
		A[0,N2+N] = a4[i];				# eta[0] 
		# North
		A[N-1,N-1] = a1[N-1,i];			# u[N-1]
		A[N-1,2*N+N2-1] = a4[i];		# eta[N-1]

		# v equation BCs
		# South
		A[N,1] = f_nd[1];					# u[1]
		A[N,N] = a1[1,i] - 2. * a2;			# v[1]
		A[N,N+1] = a2;						# v[2]
		A[N,N+N2] = - b4;					# eta[0] 	
		A[N,N+N2+2] = b4;					# eta[2]
		# North
		A[N+N2-1,N2] = f_nd[N2];					# u[N-2] 
		A[N+N2-1,N+N2-1] = a1[N2,i] - 2. * a2;		# v[N-2]
		A[N+N2-1,N+N2-2] = a2;						# v[N-3]
		A[N+N2-1,2*N+N2-1] = b4;					# eta[N-1]
		A[N+N2-1,2*N+N2-3] = - b4;					# eta[N-3]

		# eta equation BCs (Here we have to define BCs at j=N+N2,N+N2+1,3*N-3,3*N-4)
		# South
		A[N+N2,0] = c1[0,i];				# u[0]		# Again, see documentation for BC reasoning
		A[N+N2,N] = 2. * c3[0];				# v[1] (factor of 2 because we use one-sided FD for v_y at the boundaries)
		A[N+N2,N+N2] = c4[0,i];				# eta[0]
		A[N+N2+1,1] = c1[1,i];				# u[1]
		A[N+N2+1,N] = c2[1];				# v[1]
		A[N+N2+1,N+1] = c3[1];				# v[2] (factor of 1 here, back to using centered FD)
		A[N+N2+1,N+N2+1] = c4[1,i];			# eta[1]
		# North
		A[2*N+N2-1,N-1] = c1[N-1,i];			# u[N-1]
		A[2*N+N2-1,N+N2-1] = - 2. * c3[N-1];	# v[N-2] (factor of 2 because we use one-sided FD for v_y at the boundaries)
		A[2*N+N2-1,2*N+N2-1] = c4[N-1,i];		# eta[N-1]
		A[N+2*N2,N2] = c1[N2,i];  				# u[N-2]
		A[N+2*N2,N+N2-1] = c2[N2];				# v[N-2]
		A[N+2*N2,N+N2-2] = - c3[N2];			# v[N-3] (factor of 1 here, back to using centered FD)
		A[N+2*N2,N+2*N2] = c4[N2,i];			# eta[N-2]

		# Now the inner values - two loops required: one for u,v, and one for eta
		for j in range(1,N-1):
			# u equation
			A[j,j] = a1[j,i] - 2. * a2;			# u[j+1]
			A[j,j-1] = a2;						# u[j]
			A[j,j+1] = a2;						# u[j+2]
			A[j,N+j-1] = a3[j];					# v[j]
			A[j,N+N2+j] = a4[i];				# eta[j] 
			
		for j in range(1,N2-1):
			# v equation
			A[N+j,j+1] = f_nd[j+1];					# u[j+1]
			A[N+j,N+j] = a1[j+1,i] - 2. * a2;		# v[j+1]
			A[N+j,N+j-1] = a2;						# v[j-1]
			A[N+j,N+j+1] = a2;						# v[j+1]
			A[N+j,N+N2+j] = - b4;					# eta[j-1]
			A[N+j,N+N2+j+2] = b4;					# eta[j+1]

		for j in range(2,N-2):
			# eta equation
			A[N2+N+j,j] = c1[j,i];			# u[j] 
			A[N2+N+j,N+j-1] = c2[j];		# v[j]
			A[N2+N+j,N+j] = c3[j];			# v[j+1]
			A[N2+N+j,N+j-2] = - c3[j];		# v[j-1]
			A[N2+N+j,N+N2+j] = c4[j,i];		# eta[j]
	
		# Now define the forcing from the forcing input file.
		for j in range(0,N2):
			F[N+j] = Ftilde2_nd[j+1,i];
		for j in range(0,N):
			F[j] = Ftilde1_nd[j,i];
			F[N2+N+j] = Ftilde3_nd[j,i];
			
		solution[:,i] = np.linalg.solve(A,F);

		#import matplotlib.pyplot as plt
		#plt.plot(solution[0:N,i]);
		#plt.show();
	
	return solution;

#=======================================================

# FREE_SLIP_SOLVER
#=======================================================
def FREE_SLIP_SOLVER3(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,uBC,etaBC,Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,N,N2):
# Called by RSW_1L.py if BC = 'FREE-SLIP'.

	dim = N2 + 2 * N;
	#print(dim);

	A = np.zeros((dim,dim),dtype=complex);	# For the free-slip, no-normal flow BC.

	# Initialise the forcing.
	F = np.zeros((dim),dtype=complex);

	# Initialise the solution.
	solution = np.zeros((dim,N),dtype=complex);

	for i in range(0,N):
		#print(i);
				
		# First the boundary terms. Some of these could be defined in the upcoming loop,
		# but for simplicity we define all boundary terms here.
	
		# u equation BCs
		# South
		A[0,0] = a1[0,i] + uBC;			# u[0]		# See documentation for reasoning behind BCs here
		A[0,1] = - uBC;					# u[1]
		A[0,N2+N] = a4[i];				# eta[0] 
		# North
		A[N-1,N-1] = a1[N-1,i] + uBC;	# u[N-1]
		A[N-1,N-2] = - uBC;				# u[N-2]
		A[N-1,2*N+N2-1] = a4[i];		# eta[N-1]

		# v equation BCs
		# South
		A[N,1] = f_nd[1];					# u[1]
		A[N,N] = a1[1,i] - 2. * a2;			# v[1]
		A[N,N+1] = a2;						# v[2]
		A[N,N+N2] = - b4;					# eta[0] 	
		A[N,N+N2+2] = b4;					# eta[2]
		# North
		A[N+N2-1,N2] = f_nd[N2];					# u[N-2] 
		A[N+N2-1,N+N2-1] = a1[N2,i] - 2. * a2;		# v[N-2]
		A[N+N2-1,N+N2-2] = a2;						# v[N-3]
		A[N+N2-1,2*N+N2-1] = b4;					# eta[N-1]
		A[N+N2-1,2*N+N2-3] = - b4;					# eta[N-3]

		# eta equation BCs (Here we have to define BCs at j=N+N2,N+N2+1,3*N-3,3*N-4)
		# South
		A[N+N2,0] = c1[0,i] + etaBC[0];		# u[0]		# Again, see documentation for BC reasoning
		A[N+N2,N] = 2. * c3[0];				# v[1] (factor of 2 because we use one-sided FD for v_y at the boundaries)
		A[N+N2,N+N2] = c4[0,i];				# eta[0]
		A[N+N2+1,1] = c1[1,i];				# u[1]
		A[N+N2+1,N] = c2[1];				# v[1]
		A[N+N2+1,N+1] = c3[1];				# v[2] (factor of 1 here, back to using centered FD)
		A[N+N2+1,N+N2+1] = c4[1,i];			# eta[1]
		# North
		A[2*N+N2-1,N-1] = c1[N-1,i] + etaBC[1];	# u[N-1]
		A[2*N+N2-1,N+N2-1] = - 2. * c3[N-1];	# v[N-2] (factor of 2 because we use one-sided FD for v_y at the boundaries)
		A[2*N+N2-1,2*N+N2-1] = c4[N-1,i];		# eta[N-1]
		A[N+2*N2,N2] = c1[N2,i];  				# u[N-2]
		A[N+2*N2,N+N2-1] = c2[N2];				# v[N-2]
		A[N+2*N2,N+N2-2] = - c3[N2];			# v[N-3] (factor of 1 here, back to using centered FD)
		A[N+2*N2,N+2*N2] = c4[N2,i];			# eta[N-2]

		# Now the inner values - two loops required: one for u,v, and one for eta
		for j in range(1,N-1):
			# u equation
			A[j,j] = a1[j,i] - 2. * a2;			# u[j+1]
			A[j,j-1] = a2;						# u[j]
			A[j,j+1] = a2;						# u[j+2]
			A[j,N+j-1] = a3[j];					# v[j]
			A[j,N+N2+j] = a4[i];				# eta[j] 
			
		for j in range(1,N2-1):
			# v equation
			A[N+j,j+1] = f_nd[j+1];					# u[j+1]
			A[N+j,N+j] = a1[j+1,i] - 2. * a2;		# v[j+1]
			A[N+j,N+j-1] = a2;						# v[j-1]
			A[N+j,N+j+1] = a2;						# v[j+1]
			A[N+j,N+N2+j] = - b4;					# eta[j-1]
			A[N+j,N+N2+j+2] = b4;					# eta[j+1]

		for j in range(2,N-2):
			# eta equation
			A[N2+N+j,j] = c1[j,i];			# u[j] 
			A[N2+N+j,N+j-1] = c2[j];		# v[j]
			A[N2+N+j,N+j] = c3[j];			# v[j+1]
			A[N2+N+j,N+j-2] = - c3[j];		# v[j-1]
			A[N2+N+j,N+N2+j] = c4[j,i];		# eta[j]
	
		# Now define the forcing from the forcing input file.
		for j in range(0,N2):
			F[N+j] = Ftilde2_nd[j+1,i];
		for j in range(0,N):
			F[j] = Ftilde1_nd[j,i];
			F[N2+N+j] = Ftilde3_nd[j,i];
			
		solution[:,i] = np.linalg.solve(A,F);

		#import matplotlib.pyplot as plt
		#plt.plot(solution[0:N,i]);
		#plt.show();
	
	return solution;

#=======================================================

#=======================================================

# FREE_SLIP_SOLVER
#=======================================================
def FREE_SLIP_SOLVER4(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,N,N2):
# Called by RSW_1L.py if BC = 'FREE-SLIP'.
# The new test solver. Main assumption is that viscosity is zero on the boundaries.
# This results in the previously used one-sided fd approximation for v in the continuity equation.
# In the zonal mom. equation, we simply eliminate viscous contributions at the boundary.

	dim = N2 + 2 * N;
	#print(dim);

	A = np.zeros((dim,dim),dtype=complex);	# For the free-slip, no-normal flow BC.

	# Initialise the forcing.
	F = np.zeros((dim),dtype=complex);

	# Initialise the solution.
	solution = np.zeros((dim,N),dtype=complex);

	for i in range(0,N):
		#print(i);
				
		# First the boundary terms. Some of these could be defined in the upcoming loop,
		# but for simplicity we define all boundary terms here.
	
		# u equation BCs
		# South
		A[0,0] = a1[0,i];				# u[0]		# See documentation for reasoning behind BCs here
		A[0,N2+N] = a4[i];				# eta[0] 
		# North
		A[N-1,N-1] = a1[N-1,i];			# u[N-1]
		A[N-1,2*N+N2-1] = a4[i];		# eta[N-1]

		# v equation BCs
		# South
		A[N,1] = f_nd[1];					# u[1]
		A[N,N] = a1[1,i] - 2. * a2;			# v[1]
		A[N,N+1] = a2;						# v[2]
		A[N,N+N2] = - b4;					# eta[0] 	
		A[N,N+N2+2] = b4;					# eta[2]
		# North
		A[N+N2-1,N2] = f_nd[N2];					# u[N-2] 
		A[N+N2-1,N+N2-1] = a1[N2,i] - 2. * a2;		# v[N-2]
		A[N+N2-1,N+N2-2] = a2;						# v[N-3]
		A[N+N2-1,2*N+N2-1] = b4;					# eta[N-1]
		A[N+N2-1,2*N+N2-3] = - b4;					# eta[N-3]

		# eta equation BCs (Here we have to define BCs at j=N+N2,N+N2+1,3*N-3,3*N-4)
		# South
		A[N+N2,0] = c1[0,i];				# u[0]		# Again, see documentation for BC reasoning
		A[N+N2,N] = 2. * c3[0];				# v[1] (factor of 2 because we use one-sided FD for v_y at the boundaries)
		A[N+N2,N+N2] = c4[0,i];				# eta[0]
		A[N+N2+1,1] = c1[1,i];				# u[1]
		A[N+N2+1,N] = c2[1];				# v[1]
		A[N+N2+1,N+1] = c3[1];				# v[2] (factor of 1 here, back to using centered FD)
		A[N+N2+1,N+N2+1] = c4[1,i];			# eta[1]
		# North
		A[2*N+N2-1,N-1] = c1[N-1,i];			# u[N-1]
		A[2*N+N2-1,N+N2-1] = - 2. * c3[N-1];	# v[N-2] (factor of 2 because we use one-sided FD for v_y at the boundaries)
		A[2*N+N2-1,2*N+N2-1] = c4[N-1,i];		# eta[N-1]
		A[N+2*N2,N2] = c1[N2,i];  				# u[N-2]
		A[N+2*N2,N+N2-1] = c2[N2];				# v[N-2]
		A[N+2*N2,N+N2-2] = - c3[N2];			# v[N-3] (factor of 1 here, back to using centered FD)
		A[N+2*N2,N+2*N2] = c4[N2,i];			# eta[N-2]

		# Now the inner values - two loops required: one for u,v, and one for eta
		for j in range(1,N-1):
			# u equation
			A[j,j] = a1[j,i] - 2. * a2;			# u[j+1]
			A[j,j-1] = a2;						# u[j]
			A[j,j+1] = a2;						# u[j+2]
			A[j,N+j-1] = a3[j];					# v[j]
			A[j,N+N2+j] = a4[i];				# eta[j] 
			
		for j in range(1,N2-1):
			# v equation
			A[N+j,j+1] = f_nd[j+1];					# u[j+1]
			A[N+j,N+j] = a1[j+1,i] - 2. * a2;		# v[j+1]
			A[N+j,N+j-1] = a2;						# v[j-1]
			A[N+j,N+j+1] = a2;						# v[j+1]
			A[N+j,N+N2+j] = - b4;					# eta[j-1]
			A[N+j,N+N2+j+2] = b4;					# eta[j+1]

		for j in range(2,N-2):
			# eta equation
			A[N2+N+j,j] = c1[j,i];			# u[j] 
			A[N2+N+j,N+j-1] = c2[j];		# v[j]
			A[N2+N+j,N+j] = c3[j];			# v[j+1]
			A[N2+N+j,N+j-2] = - c3[j];		# v[j-1]
			A[N2+N+j,N+N2+j] = c4[j,i];		# eta[j]
	
		# Now define the forcing from the forcing input file.
		for j in range(0,N2):
			F[N+j] = Ftilde2_nd[j+1,i];
		for j in range(0,N):
			F[j] = Ftilde1_nd[j,i];
			F[N2+N+j] = Ftilde3_nd[j,i];
			
		solution[:,i] = np.linalg.solve(A,F);

		#import matplotlib.pyplot as plt
		#plt.plot(solution[0:N,i]);
		#plt.show();
	
	return solution;

#=======================================================

# extractSols
#=======================================================	
def extractSols(solution,N,N2,BC):

	dim = len(solution[:,0]);

	# Intialise the solutions
	utilde_nd = np.zeros((N,N),dtype=complex);
	vtilde_nd = np.zeros((N,N),dtype=complex);
	etatilde_nd = np.zeros((N,N),dtype=complex);
    
	if BC == 'NO-SLIP':
		if dim == 3**N2:
			for j in range(0,N2):
				utilde_nd[j+1,:] = solution[j,:];
				vtilde_nd[j+1,:] = solution[N2+j,:];
				etatilde_nd[j+1,:] = solution[2*N2+j,:];
		else:
			for j in range(0,N2):
				utilde_nd[j+1,:] = solution[j,:];
				vtilde_nd[j+1,:] = solution[N2+j,:];
			for j in range(0,N):		
				etatilde_nd[j,:] = solution[2*N2+j,:];
		
	elif BC == 'FREE-SLIP':
		for j in range(0,N):
			utilde_nd[j,:] = solution[j,:];
			etatilde_nd[j,:] = solution[N+N2+j,:];
		for j in range(0,N2):
			vtilde_nd[j+1,:] = solution[N+j,:];
		
	else:
		print('ERROR');

	return utilde_nd, vtilde_nd, etatilde_nd;

#=======================================================

# SPEC_TO_PHYS
def SPEC_TO_PHYS(utilde,vtilde,etatilde,T,dx,omega,N):
# Function takes the spectral-physical solutions produced by the solver and returns the time-dependent solutions in physical space.
	
	I = np.complex(0,1);

	Nt = np.size(T) - 1;

	u = np.zeros((N,N,Nt),dtype=complex);
	v = np.zeros((N,N,Nt),dtype=complex);
	eta = np.zeros((N,N,Nt),dtype=complex);	

	for ti in range(0,Nt):
		# Calculate the solutions in physical space at some instant t. Solutions are divided (later) by extra factor AmpF, so that they are normalised by the forcing amplitude.
		u[:,:,ti] = np.exp(2.*np.pi*I*omega*T[ti])*np.fft.ifft(utilde,axis=1) / dx 
		v[:,:,ti] = np.exp(2.*np.pi*I*omega*T[ti])*np.fft.ifft(vtilde,axis=1) / dx		
		eta[:,:,ti] = np.exp(2.*np.pi*I*omega*T[ti])*np.fft.ifft(etatilde,axis=1) / dx

	return u, v, eta;

#=======================================================

# SPEC_TO_PHYS_STOCH
def SPEC_TO_PHYS_STOCH(utilde,vtilde,etatilde,dx,N):
# Function takes the spectral-physical solutions produced by the solver and returns the time-dependent solutions in physical space.
	
	I = np.complex(0,1);

	u = np.fft.ifft(np.fft.ifft(utilde,axis=1),axis=2) / (dx);
	v = np.fft.ifft(np.fft.ifft(vtilde,axis=1),axis=2) / (dx);
	eta = np.fft.ifft(np.fft.ifft(etatilde,axis=1),axis=2) / (dx);


	return u, v, eta;






