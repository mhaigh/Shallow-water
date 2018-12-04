# solverAlt.py
#=======================================================

# An alternative set of modules to the solver.py file.
# The modules here can be called for a specific zonal wavenumber index,
# instead of automatically running for all indices.

# Note: the coefficients module is, for now, not rewritten. 

#=======================================================

import numpy as np
from diagnostics import diff

#=======================================================

# NO_SLIP_SOLVER 
#=======================================================
def NO_SLIP_SOLVER(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,N,N2,i):
# Called by RSW_1L.py if BC = 'NO-SLIP'.

	dim = 2 * N2 + N;
	#print(dim);

	A = np.zeros((dim,dim),dtype=complex);	# For the no-slip, no-normal flow BC.
	# eta has N gridpoints in y, whereas u and v have N2=N-2, after removing the two 'dead' gridpoints.
	# We primarily consider forcing away from the boundaries so that it's okay applying no-slip BCs.

	# Initialise the forcing.
	F = np.zeros((dim),dtype=complex);

	solution = np.zeros((dim,N),dtype=complex);

	# First the boundary terms. Some of these could be defined in the upcoming loop,
	# but for simplicity we define all boundary terms here.
	
	# u equation BCs
	# South
	A[0,0] = a1[1,i] - 2. * a2;			# u[1]
	A[0,1] = a2;						# u[2]
	A[0,N2] = a3[1];					# v[1]
	A[0,2*N2+1] = a4[i];				# eta[1] (eta[0] lies at i=2*N2 in A)
	# North
	A[N2-1,N2-1] = a1[N2,i] - 2. * a2;	# u[N2]=u[N-2]
	A[N2-1,N2-2] = a2;					# u[N2-1]=u[N-3]
	A[N2-1,2*N2-1] = a3[N2];			# v[N2]=v[N-2]
	A[N2-1,3*N2] = a4[i];				# eta[N2] (eta[N2] lies at i=3*N2 in A)

	# v equation BCs
	# South
	A[N2,0] = f_nd[1];					# u[1]
	A[N2,N2] = a1[1,i] - 2. * a2		# v[1]
	A[N2,N2+1] = a2;					# v[2]
	A[N2,2*N2] = - b4;					# eta[0] 
	A[N2,2*N2+2] = b4;					# eta[2]
	# North
	A[2*N2-1,N2-1] = f_nd[N2];					# u[N-2] 
	A[2*N2-1,2*N2-1] = a1[N2,i] - 2. * a2;		# v[N-2]
	A[2*N2-1,2*N2-2] = a2;						# v[N-3]
	A[2*N2-1,2*N2+N-1] = b4;					# eta[N-1]
	A[2*N2-1,2*N2+N-3] = - b4;					# eta[N-3]
	
	# eta equation BCs (Here we have to define BCs at j=2*N2,2*N2+1,3*N2+1,3*N2)
	# South
	A[2*N2,N2] = 2. * c3[0]					# v[1] (factor of 2 because we use one-sided FD for v_y at the boundaries)
	A[2*N2,2*N2] = c4[0,i];					# eta[0]
	A[2*N2+1,0] = c1[1,i];					# u[1]
	A[2*N2+1,N2] = c2[1];					# v[1]
	A[2*N2+1,N2+1] = c3[1];					# v[2] (factor of 1 here, back to using centered FD)
	A[2*N2+1,2*N2+1] = c4[1,i];				# eta[1]
	# North
	A[2*N2+N-1,2*N2-1] = - 2. * c3[N2+1];	# v[N-2] (factor of 2 because we use one-sided FD for v_y at the boundaries)
	A[2*N2+N-1,3*N2+1] = c4[N-1,i];			# eta[N-1]
	A[3*N2,N2-1] = c1[N2,i];  				# u[N-2]
	A[3*N2,2*N2-1] = c2[N2];				# v[N-2]
	A[3*N2,2*N2-2] = - c3[N2];				# v[N-3] (factor of 1 here, back to using centered FD)
	A[3*N2,3*N2] = c4[N2,i];				# eta[N-2]

	# Now the inner values - two loops required: one for u,v, and one for eta
	for j in range(1,N2-1):
		# u equation
		A[j,j] = a1[j+1,i] - 2. * a2;			# u[j+1]
		A[j,j-1] = a2;							# u[j]
		A[j,j+1] = a2;							# u[j+2]
		A[j,N2+j] = a3[j+1];					# v[j]
		A[j,2*N2+j+1] = a4[i];					# eta[j] (must add 1 to each eta index to skip out eta[0])
			
		# v equation
		A[N2+j,j] = f_nd[j+1];					# u[j+1]
		A[N2+j,N2+j] = a1[j+1,i] - 2. * a2;		# v[j+1]
		A[N2+j,N2+j-1] = a2;					# v[j]
		A[N2+j,N2+j+1] = a2;					# v[j+2]
		A[N2+j,2*N2+j] = - b4;					# eta[j]
		A[N2+j,2*N2+j+2] = b4;					# eta[j+2]

	for j in range(2,N2):
		# eta equation
		A[2*N2+j,j-1] = c1[j,i];		# u[j] (the J=j-1 index of A(,J) corresponds to u[j], u without the deadpoints)
		A[2*N2+j,N2+j-1] = c2[j];		# v[j]
		A[2*N2+j,N2+j] = c3[j];			# v[j+1]
		A[2*N2+j,N2+j-2] = - c3[j];		# v[j-1]
		A[2*N2+j,2*N2+j] = c4[j,i];		# eta[j]
	
	for j in range(0,N2):	
		F[j] = Ftilde1_nd[j+1,i];		# Have to add 1 to y-index to skip the first 'dead' gridpoint.
		F[N2+j] = Ftilde2_nd[j+1,i];
	for j in range(0,N):
		F[2*N2+j] = Ftilde3_nd[j,i];	

	solution = np.linalg.solve(A,F);

	return solution;

#=======================================================

# FREE_SLIP_SOLVER
#=======================================================
def FREE_SLIP_SOLVER(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,uBC,etaBC,Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,N,N2,i):
# Called by RSW_1L.py if BC = 'FREE-SLIP'.

	dim = N2 + 2 * N;
	#print(dim);

	A = np.zeros((dim,dim),dtype=complex);	# For the free-slip, no-normal flow BC.

	# Initialise the forcing.
	F = np.zeros((dim),dtype=complex);

	# Initialise the solution.
	solution = np.zeros((dim,N),dtype=complex);
				
	# First the boundary terms. Some of these could be defined in the upcoming loop,
	# but for simplicity we define all boundary terms here.
	
	# u equation BCs
	# South
	A[0,0] = a1[0,i] - 2 * a2;			# u[0]		# Using one-sided approx. for 2nd-order derivative
	A[0,1] = a2;						# u[1]
	A[0,N2+N] = a4[i];					# eta[0] 
	# North
	A[N-1,N-1] = a1[N-1,i] - 2 * a2;	# u[N-1]
	A[N-1,N-2] = a2;					# u[N-2]
	A[N-1,2*N+N2-1] = a4[i];			# eta[N-1]

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
	A[N+N2,0] = c1[0,i];				# u[0]
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
			
	solution = np.linalg.solve(A,F);
	
	return solution;

#=======================================================




