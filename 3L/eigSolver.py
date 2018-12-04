# eigSolver.py
#=======================================================

# Contains a set of functions to be called by EIG.py or EIG_DECOMP.py. 

#=======================================================

import numpy as np
from diagnostics import diff

#=======================================================

# EIG_COEFFICIENTS
#=======================================================
def EIG_COEFFICIENTS(Ro,Re,K_nd,f_nd,U1_nd,U2_nd,H1_nd,H2_nd,rho1_nd,rho2_nd,gamma_nd,dy_nd,N):
# Here we also divide coeficients of derivatives by the relevant space-step: 
# 2*dy_nd for first-order derivatives, dy_nd**2 for second-order derivatives.
# Note: to speed up the algorithm, some coefficients aren't unnecessarily defined as they have a duplicate featuring in another equation.
# All coefficients are related to their solver.py counterparts: the same but divided by a factor of 2*pi*I*Ro in mom. eqs only,
# and 2*pi*I in cont. eqs. The time-derivative terms are also of course omitted.

	I = np.complex(0,1);

	# 1 - Define all k- and y-dependent coefficients
	a1 = np.zeros((N,N),dtype=complex);
	c1 = np.zeros((N,N),dtype=complex);
	c4 = np.zeros((N,N),dtype=complex);
	d1 = np.zeros((N,N),dtype=complex);
	f1 = np.zeros((N,N),dtype=complex);
	f4 = np.zeros((N,N),dtype=complex);
	for j in range(0,N):
		for i in range(0,N):
			a1[j,i] = U1_nd[j] * K_nd[i] - 2 * np.pi * I * K_nd[i]**2 / Re;
			c1[j,i] = H1_nd[j] * K_nd[i];
			c4[j,i] = U1_nd[j] * K_nd[i];
			d1[j,i] = U2_nd[j] * K_nd[i] - 2 * np.pi * I * K_nd[i]**2 / Re - I * gamma_nd / (2 * np.pi * Ro);
			f1[j,i] = H2_nd[j] * K_nd[i];
			f4[j,i] = U2_nd[j] * K_nd[i];
	# Edit from here.
	# 2 - Define all y-dependent coefficients
	a3 = I * (f_nd - Ro * diff(U1_nd,2,0,dy_nd)) / (2 * np.pi * Ro);
	b1 = - I * f_nd / (2 * np.pi * Ro);
	d3 = I * (f_nd - Ro * diff(U2_nd,2,0,dy_nd)) / (2 * np.pi * Ro);
	f2 = - I * diff(H2_nd,2,0,dy_nd) / (2 * np.pi);
	f3 = - I * H2_nd / (4 * np.pi * dy_nd);

	# 3 - Define all k-dependent coefficients
	a4 = K_nd / Ro;
	d4 = K_nd * rho1_nd / Ro;

	# 4 - Define all constant coefficients
	a2 = I / (2 * np.pi * Re * dy_nd**2);
	b4 = - I / (4 * np.pi * Ro * dy_nd); 
	c2 = - I * diff(H1_nd,2,0,dy_nd) / (2 * np.pi);
	c3 = - I * H1_nd / (4 * np.pi * dy_nd);
	e4 = - I * rho1_nd / (4 * np.pi * Ro * dy_nd);
	
	# Summary of coefficients not explicitly defined for sake of algorithm speed:
	# b2 = a1, b3 = a2, d2 = a2, d5 = a4, e1 = b1, e2 = d1, e3 = a2, e5 = b4.
	
	return a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,d1,d3,d4,e4,f1,f2,f3,f4;

#=======================================================

# NO_SLIP_EIG 
#=======================================================
def NO_SLIP_EIG(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,d1,d3,d4,e4,f1,f2,f3,f4,N,N2,i,VECS):
# Called if BC = 'NO-SLIP'.

	dim = 6 * N - 8; 	 # u1, u2 and v1, v2 have N-2 gridpoints, h1, h2 have N gridpoints
	#print(dim);

	A = np.zeros((dim,dim),dtype=complex);	# For the free-slip, no-normal flow BC.
	# eta has N gridpoints in y, whereas u and v have N2=N-2, after removing the two 'dead' gridpoints.
	# We define a so that the 6 equations are ordered as follows: u1, u2, v1, v2, h1, h2.

	# Boundary Conditions

	# u1 equation
	# South
	A[0,0] = a1[1,i] - 2 * a2;		# u1[1]
	A[0,1] = a2;					# u1[2]
	A[0,2*N2] = a3[1];				# v1[1]
	A[0,4*N2+1] = a4[i];			# h1[1]
	A[0,4*N2+N+1] = a4[i];			# h2[1]
	# North
	A[N2-1,N2-1] = a1[N2,i] - 2 * a2;	# u1[N-2]
	A[N2-1,N2-2] = a2;					# u1[N-3]
	A[N2-1,3*N2-1] = a3[N2];			# v1[N-3]
	A[N2-1,4*N2+N-2] = a4[i];			# h1[N-2]
	A[N2-1,4*N2+2*N-2] = a4[i];			# h2[N-2]

	# u2 equation
	# South
	A[N2,N2] = d1[1,i] - 2 * a2;	# u2[1]
	A[N2,N2+1] = a2;				# u2[2]
	A[N2,3*N2] = d3[1];				# v2[1]
	A[N2,4*N2+1] = d4[i];			# h1[1]
	A[N2,4*N2+N+1] = a4[i];			# h2[1]
	# North
	A[2*N2-1,2*N2-1] = d1[N2,i] - 2 * a2;	# u2[N-2]
	A[2*N2-1,2*N2-2] = a2;					# u2[N-3]
	A[2*N2-1,4*N2-1] = d3[N2];				# v2[N-2]
	A[2*N2-1,4*N2+N-2] = d4[i];				# h1[N-2]
	A[2*N2-1,4*N2+2*N-2] = a4[i];			# h2[N-2]
		
	# v1 equation
	# South
	A[2*N2,0] = b1[1];					# u1[1]
	A[2*N2,2*N2] = a1[1,i] - 2 * a2;	# v1[1]
	A[2*N2,2*N2+1] = a2;				# v1[2]
	A[2*N2,4*N2+2] = b4;				# h1[2]
	A[2*N2,4*N2] = - b4;				# h1[0]
	A[2*N2,4*N2+N+2] = b4;				# h2[2]
	A[2*N2,4*N2+N] = - b4;				# h2[0]
	# North
	A[3*N2-1,N2-1] = b1[N2];				# u1[N-2]
	A[3*N2-1,3*N2-1] = a1[N2,i] - 2 * a2;	# v1[N-2]
	A[3*N2-1,3*N2-2] = a2;					# v1[N-3]
	A[3*N2-1,4*N2+N-1] = b4;				# h1[N-1]
	A[3*N2-1,4*N2+N-3] = - b4;				# h1[N-3]
	A[3*N2-1,4*N2+2*N-1] = b4;				# h2[N-1]
	A[3*N2-1,4*N2+2*N-3] = - b4;			# h2[N-3]

	# v2 equation
	# South
	A[3*N2,N2] = b1[1];					# u2[1]
	A[3*N2,3*N2] = d1[1,i] - 2 * a2;	# v2[1]
	A[3*N2,3*N2+1] = a2;				# v2[2]
	A[3*N2,4*N2+2] = e4;				# h1[2]
	A[3*N2,4*N2] = - e4;				# h1[0]
	A[3*N2,4*N2+N+2] = b4;				# h2[2]
	A[3*N2,4*N2+N] = - b4;				# h2[0]
	# North
	A[4*N2-1,2*N2-1] = b1[N2];				# u2[N-2]
	A[4*N2-1,4*N2-1] = d1[N2,i] - 2 * a2;	# v2[N-2]
	A[4*N2-1,4*N2-2] = a2;					# v2[N-3]
	A[4*N2-1,4*N2+N-1] = e4;				# h1[N-1]
	A[4*N2-1,4*N2+N-3] = - e4;				# h1[N-3]
	A[4*N2-1,4*N2+2*N-1] = b4;				# h2[N-1]
	A[4*N2-1,4*N2+2*N-3] = - b4;			# h2[N-3]
	
	# h1 equation
	# South
	A[4*N2,2*N2] = 2 * c3[0];		# v1[1] (one-sided FD)
	A[4*N2,4*N2] = c4[0,i];			# h1[0]
	A[4*N2+1,0] = c1[1,i];			# u1[1]
	A[4*N2+1,2*N2] = c2[1];			# v1[1]
	A[4*N2+1,2*N2+1] = c3[1];		# v1[2]
	A[4*N2+1,4*N2+1] = c4[1,i];		# h1[1]

	# North
	A[4*N2+N-1,3*N2-1] = - 2 * c3[N-1];		# v1[N-2]
	A[4*N2+N-1,4*N2+N-1] = c4[N-1,i];		# h1[N-1]
	A[4*N2+N-2,N2-1] = c1[N2,i];			# u1[N-2]
	A[4*N2+N-2,3*N2-1] = c2[N2];			# v1[N-2]
	A[4*N2+N-2,3*N2-2] = - c3[N2];			# v1[N-3]
	A[4*N2+N-2,4*N2+N-2] = c4[N2,i];		# h1[N-2]

	# h2 equation
	# South
	A[4*N2+N,3*N2] = 2 * f3[0];			# v2[1] (one-sided FD)
	A[4*N2+N,4*N2+N] = f4[0,i];			# h2[0]
	A[4*N2+N+1,N2] = f1[1,i];			# u2[1]
	A[4*N2+N+1,3*N2] = f2[1];			# v2[1]
	A[4*N2+N+1,3*N2+1] = f3[1];			# v2[2]
	A[4*N2+N+1,4*N2+N+1] = f4[1,i];		# h2[1]
	# North
	A[4*N2+2*N-1,4*N2-1] = - 2 * f3[N-1];	# v2[N-1]
	A[4*N2+2*N-1,4*N2+2*N-1] = f4[N-1,i];	# h2[N-1]
	A[4*N2+2*N-2,2*N2-1] = f1[N2,i];		# u2[N-2]
	A[4*N2+2*N-2,4*N2-1] = f2[N2];			# v2[N-2]
	A[4*N2+2*N-2,4*N2-2] = f3[N2];			# v2[N-3]
	A[4*N2+2*N-2,4*N2+2*N-2] = f4[N2,i];	# h2[N-2]		

	# Inner domain values

	# A loop for the u and v equations
	for j in range(1,N-3):
		# u1 equation
		A[j,j] = a1[j+1,i] - 2 * a2;	# u1[j+1]
		A[j,j+1] = a2;					# u1[j+2]
		A[j,j-1] = a2;					# u1[j]
		A[j,2*N2+j] = a3[j+1];			# v1[j+1] 
		A[j,4*N2+j+1] = a4[i];			# h1[j+1]
		A[j,4*N2+N+j+1] = a4[i];		# h2[j+1]

		# u2 equation
		A[N2+j,N2+j] = d1[j+1,i] - 2 * a2;	# u2[j+1]
		A[N2+j,N2+j+1] = a2;				# u2[j+2]
		A[N2+j,N2+j-1] = a2;				# u2[j]
		A[N2+j,3*N2+j] = d3[j+1];			# v2[j+1]
		A[N2+j,4*N2+j+1] = d4[i];			# h1[j+1]
		A[N2+j,4*N2+N+j+1] = a4[i];			# h2[j+1]
	
		# v1 equation
		A[2*N2+j,j] = b1[j+1];					# u1[j+1]
		A[2*N2+j,2*N2+j] = a1[j+1,i] - 2 * a2;	# v1[j+1]
		A[2*N2+j,2*N2+j+1] = a2;				# v1[j+2]
		A[2*N2+j,2*N2+j-1] = a2;				# v1[j]
		A[2*N2+j,4*N2+j+2] = b4;				# h1[j+2]
		A[2*N2+j,4*N2+j] = - b4;				# h1[j] 
		A[2*N2+j,4*N2+N+j+2] = b4;				# h2[j+2]
		A[2*N2+j,4*N2+N+j] = - b4;				# h2[j]

		# v2 equation
		A[3*N2+j,N2+j] = b1[j+1];				# u2[j+1]
		A[3*N2+j,3*N2+j] = d1[j+1,i] - 2 * a2;	# v2[j+1]
		A[3*N2+j,3*N2+j+1] = a2;				# v2[j+2]
		A[3*N2+j,3*N2+j-1] = a2;				# v2[j]
		A[3*N2+j,4*N2+j+2] = e4;				# h1[j+2]
		A[3*N2+j,4*N2+j] = - e4;				# h1[j]
		A[3*N2+j,4*N2+N+j+2] = b4;				# h2[j+2]
		A[3*N2+j,4*N2+N+j] = - b4;				# h2[j]			
			
		# A loop for the h equations

	for j in range(2,N-2):
		# h1 equation
		A[4*N2+j,j-1] = c1[j,i];			# u1[j]
		A[4*N2+j,2*N2+j-1] = c2[j];			# v1[j]
		A[4*N2+j,2*N2+j] = c3[j];			# v1[j+1]
		A[4*N2+j,2*N2+j-2] = - c3[j];		# v1[j-1]
		A[4*N2+j,4*N2+j] = c4[j,i];			# h1[j]

		# h2 equation
		A[4*N2+N+j,N2+j-1] = f1[j,i];		# u2[j]
		A[4*N2+N+j,3*N2+j-1] = f2[j];		# v2[j]
		A[4*N2+N+j,3*N2+j] = f3[j];			# v2[j+1]
		A[4*N2+N+j,3*N2+j-2] = - f3[j];		# v2[j-1]
		A[4*N2+N+j,4*N2+N+j] = f4[j,i];		# h2[j]
			

	val, vec = np.linalg.eig(A);

	if VECS:
		u1_vec = np.zeros((N,dim),dtype=complex);
		u2_vec = np.zeros((N,dim),dtype=complex);
		v1_vec = np.zeros((N,dim),dtype=complex);
		v2_vec = np.zeros((N,dim),dtype=complex);
		h1_vec = np.zeros((N,dim),dtype=complex);
		h2_vec = np.zeros((N,dim),dtype=complex);
		for j in range(0,N2):
			u1_vec[j+1,:] = vec[j,:];
			u2_vec[j+1,:] = vec[N2+j,:];
			v1_vec[j+1,:] = vec[2*N2+j,:];
			v2_vec[j+1,:] = vec[3*N2+j,:];
		for j in range(0,N):
			h1_vec[j,:] = vec[4*N2+j,:];
			h2_vec[j,:] = vec[4*N2+N+j,:];	

		return val, u1_vec, u2_vec, v1_vec, v2_vec, h1_vec, h2_vec;

	else:
		return val, vec;

#=======================================================

# FREE_SLIP_EIG
#=======================================================
def FREE_SLIP_EIG(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,d1,d3,d4,e4,f1,f2,f3,f4,N,N2,i,VECS):
# Called if BC = 'FREE-SLIP'.

	dim = 6 * N - 4; 	 # u and eta have N gridpoints, v have N-2 gridpoints
	#print(dim);

	A = np.zeros((dim,dim),dtype=complex);	# For the free-slip, no-normal flow BC.
	# eta has N gridpoints in y, whereas u and v have N2=N-2, after removing the two 'dead' gridpoints.
	# We primarily consider forcing away from the boundaries so that it's okay applying no-slip BCs.
	# We define a so that the 6 equations are ordered as follows: u1, u2, v1, v2, eta0, eta1.

	solution = np.zeros((dim,N),dtype=complex);		

	for i in range(0,N):
		#print(i);

		# Boundary conditions

		# u1 equation
		# South
		A[0,0] = a1[0,i] + a2; 		# u1[0]
		A[0,1] = - 2. * a2;			# u1[1]
		A[0,2] = a2; 				# u1[2]
		A[0,2*N+2*N2] = a4[i];		# h1[0]
		A[0,3*N+2*N2] = a4[i];		# h2[0]
		# North
		A[N-1,N-1] = a1[N-1,i] + a2;	# u1[N-1]
		A[N-1,N-2] = - 2. * a2;			# u1[N-2]
		A[N-1,N-3] = a2;				# u1[N-3]
		A[N-1,3*N+2*N2-1] = a4[i];		# h1[N-1]
		A[N-1,4*N+2*N2-1] = a4[i];		# h2[N-1]
	
		# u2 equation
		# South
		A[N,N] = d1[0,i] + a2;		# u2[0]
		A[N,N+1] = - 2. * a2;		# u2[1]
		A[N,N+2] = a2;				# u2[2]
		A[N,2*N+2*N2] = d4[i];		# h1[0]
		A[N,3*N+2*N2] = a4[i];		# h2[0]
		# North
		A[2*N-1,2*N-1] = d1[N-1,i] + a2;	# u2[N-1]
		A[2*N-1,2*N-2] = - 2. * a2;			# u2[N-2]
		A[2*N-1,2*N-2] = a2;				# u2[N-3]
		A[2*N-1,3*N+2*N2-1] = d4[i];		# h1[N-1]
		A[2*N-1,4*N+2*N2-1] = a4[i];		# h2[N-1]
		
		# v1 equation
		# South
		A[2*N,1] = b1[1];					# u1[1]
		A[2*N,2*N] = a1[1,i] - 2. * a2;		# v1[1]
		A[2*N,2*N+1] = a2;					# v1[2]
		A[2*N,2*N+2*N2] = - b4;				# h1[0]
		A[2*N,2*N+2*N2+2] = b4;				# h1[2]
		A[2*N,3*N+2*N2] = - b4;				# h2[0]
		A[2*N,3*N+2*N2+2] = b4;				# h2[2]
		# North
		A[2*N+N2-1,N-2] = b1[N2];						# u1[N-2]
		A[2*N+N2-1,2*N+N2-1] = a1[N-2,i] - 2. * a2;		# v1[N-2]
		A[2*N+N2-1,2*N+N2-2] = a2;						# v1[N-3]
		A[2*N+N2-1,3*N+2*N2-1] = b4;					# h1[N-1]
		A[2*N+N2-1,3*N+2*N2-3] = - b4;					# h1[N-3]
		A[2*N+N2-1,4*N+2*N2-1] = b4;					# h2[N-1]
		A[2*N+N2-1,4*N+2*N2-3] = - b4;					# h2[N-3]

		# v2 equation
		# South
		A[2*N+N2,N+1] = b1[1];					# u2[1]
		A[2*N+N2,2*N+N2] = d1[1,i] - 2. * a2;	# v2[1]
		A[2*N+N2,2*N+N2+1] = a2;				# v2[2]
		A[2*N+N2,2*N+2*N2+2] = e4;				# h1[2]
		A[2*N+N2,2*N+2*N2] = - e4;				# h1[0]
		A[2*N+N2,3*N+2*N2+2] = b4;				# h2[2]
		A[2*N+N2,3*N+2*N2] = - b4;				# h2[0]
		# North
		A[2*N+2*N2-1,2*N-2] = b1[N2];					# u2[N-2]
		A[2*N+2*N2-1,2*N+2*N2-1] = d1[N2,i] - 2. * a2;	# v2[N-2]
		A[2*N+2*N2-1,2*N+2*N2-2] = a2;					# v2[N-3]
		A[2*N+2*N2-1,3*N+2*N2-1] = e4;					# h1[N-1]
		A[2*N+2*N2-1,3*N+2*N2-3] = - e4;				# h1[N-3]
		A[2*N+2*N2-1,4*N+2*N2-1] = b4;					# h2[N-1]
		A[2*N+2*N2-1,4*N+2*N2-3] = - b4;				# h2[N-3]

		# h1 equation
		# South
		A[2*N+2*N2,0] = c1[0,i];				# u1[0]
		A[2*N+2*N2,2*N] = 2. * c3[0];			# v1[1] (one-sided FD approx.)
		A[2*N+2*N2,2*N+2*N2] = c4[0,i];			# h1[0]
		A[2*N+2*N2+1,1] = c1[0,i];				# u1[1]
		A[2*N+2*N2+1,2*N] = c2[1];				# v1[1]
		A[2*N+2*N2+1,2*N+1] = c3[1];			# v1[2]
		A[2*N+2*N2+1,2*N+2*N2+1] = c4[1,i];		# h1[1]
		# North
		A[3*N+2*N2-1,N-1] = c1[N-1,i];				# u1[N-1]
		A[3*N+2*N2-1,2*N+N2-1] = - 2. * c3[N-1];	# v1[N-2] (one-sided FD approx.)
		A[3*N+2*N2-1,3*N+2*N2-1] = c4[N-1,i];		# h1[N-1]
		A[3*N+2*N2-2,N-2] = c1[N2,i];				# u1[N-2]
		A[3*N+2*N2-2,2*N+N2-1] = c2[N2];			# v1[N-2]
		A[3*N+2*N2-2,2*N+N2-2] = - c3[N2];			# v1[N-3]	
		A[3*N+2*N2-2,3*N+2*N2-2] = c4[N2,i];		# h1[N-2]

		# h2 equation
		# South
		A[3*N+2*N2,N] = f1[0,i];				# u2[0]
		A[3*N+2*N2,2*N+N2] = 2. * f3[0];		# v2[0] (one-sided FD approx.)
		A[3*N+2*N2,3*N+2*N2] = f4[0,i];			# h2[0]
		A[3*N+2*N2+1,N+1] = f1[1,i];			# u2[1]
		A[3*N+2*N2+1,2*N+N2] = f2[1];			# v2[1]
		A[3*N+2*N2+1,2*N+N2] = f3[1];			# v2[2]
		A[3*N+2*N2+1,3*N+2*N2+1] = f4[1,i];		# h2[1]
		# North
		A[4*N+2*N2-1,N-1] = f1[N-1,i];					# u2[N-1]
		A[4*N+2*N2-1,2*N+2*N2-1] = - 2. * f3[N-1];		# v2[N-1] (one-sided FD approx.)
		A[4*N+2*N2-1,4*N+2*N2-1] = f4[N-1,i];			# h2[N-1]
		A[4*N+2*N2-2,N-2] = f1[N2,i];					# u2[N-2]
		A[4*N+2*N2-2,2*N+2*N2-1] = f2[N2];				# v2[N-2]
		A[4*N+2*N2-2,2*N+2*N2-2] = - f3[N2];			# v2[N-3]
		A[4*N+2*N2-2,4*N+2*N2-2] = f4[N2,i];			# h2[N-2]

		# Inner domain values	
	
		# A loop for remaining values of the u equations.
		for j in range(1,N-1):
			# u1 equation
			A[j,j] = a1[j,i] - 2. * a2;		# u1[j]
			A[j,j+1] = a2;					# u1[j+1]
			A[j,j-1] = a2;					# u1[j-1]
			A[j,2*N+j-1] = a3[j];			# v1[j]
			A[j,2*N+2*N2+j] = a4[i];		# h1[j]
			A[j,3*N+2*N2+j] = a4[i];		# h2[j]
			# u2 equation
			A[N+j,N+j] = d1[j,i] - 2. * a2;		# u2[j]
			A[N+j,N+j+1] = a2;					# u2[j+1]
			A[N+j,N+j-1] = a2;					# u2[j-1]
			A[N+j,2*N+N2+j-1] = d3[j];			# v2[j]
			A[N+j,2*N+2*N2+j] = d4[i];			# h1[j]
			A[N+j,3*N+2*N2+j] = a4[i];			# h2[j]	

		# A loop for the remaining values of the v equations.
		for j in range(1,N-3):
			# v1 equation
			A[2*N+j,j+1] = b1[j+1];					# u1[j+1]
			A[2*N+j,2*N+j] = a1[j+1,i] - 2. * a2;	# v1[j+1]
			A[2*N+j,2*N+j+1] = a2;					# v1[j+2]
			A[2*N+j,2*N+j-1] = a2;					# v1[j]
			A[2*N+j,2*N+2*N2+j+2] = b4;				# h1[j+2]
			A[2*N+j,2*N+2*N2+j] = - b4;				# h1[j]
			A[2*N+j,3*N+2*N2+j+2] = b4;				# h2[j+2]
			A[2*N+j,3*N+2*N2+j] = - b4;				# h2[j]
			# v2 equation
			A[2*N+N2+j,N+j+1] = b1[j+1];				# u2[j+1]
			A[2*N+N2+j,2*N+N2+j] = d1[j+1,i] - 2. * a2;	# v2[j+1]
			A[2*N+N2+j,2*N+N2+j+1] = a2;				# v2[j+2]
			A[2*N+N2+j,2*N+N2+j-1] = a2;				# v2[j]
			A[2*N+N2+j,2*N+2*N2+j+2] = e4;				# h1[j+2]
			A[2*N+N2+j,2*N+2*N2+j] = - e4;				# h1[j]
			A[2*N+N2+j,3*N+2*N2+j+2] = b4;				# h2[j+2]
			A[2*N+N2+j,3*N+2*N2+j] = - b4;				# h2[j]

		# A loop for the remaining values of the h/eta equations.
		for j in range(2,N-2):
			# h1 equation
			A[2*N+2*N2+j,j] = c1[j,i];				# u1[j]
			A[2*N+2*N2+j,2*N+j-1] = c2[j];			# v1[j]
			A[2*N+2*N2+j,2*N+j] = c3[j];			# v1[j+1]
			A[2*N+2*N2+j,2*N+j-2] = - c3[j];		# v1[j-1]
			A[2*N+2*N2+j,2*N+2*N2+j] = c4[j,i];		# h1[j]
			# h2 equation
			A[3*N+2*N2+j,N+j] = f1[j,i];				# u2[j]
			A[3*N+2*N2+j,2*N+N2+j-1] = f2[j];			# v2[j]
			A[3*N+2*N2+j,2*N+N2+j] = f3[j];				# v2[j+1]
			A[3*N+2*N2+j,2*N+N2+j-2] = - f3[j];			# v2[j-1]
			A[3*N+2*N2+j,3*N+2*N2+j] = f4[j,i];			# h2[j]
		
	val, vec = np.linalg.eig(A);

	if VECS:
		u1_vec = np.zeros((N,dim),dtype=complex);
		u2_vec = np.zeros((N,dim),dtype=complex);
		v1_vec = np.zeros((N,dim),dtype=complex);
		v2_vec = np.zeros((N,dim),dtype=complex);
		h1_vec = np.zeros((N,dim),dtype=complex);
		h2_vec = np.zeros((N,dim),dtype=complex);
		for j in range(0,N2):
			v1_vec[j+1,:] = vec[2*N+j,:];
			v2_vec[j+1,:] = vec[2*N+N2+j,:];
		for j in range(0,N):
			u1_vec[j,:] = vec[j,:];
			u2_vec[j,:] = vec[N+j,:];
			h1_vec[j,:] = vec[2*N+2*N2+j,:];
			h2_vec[j,:] = vec[3*N+2*N2+j,:];	

		return val, u1_vec, u2_vec, v1_vec, v2_vec, h1_vec, h2_vec;

	else:
		return val, vec;


#=======================================================


