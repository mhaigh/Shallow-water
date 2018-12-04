# eigSolver.py
#=======================================================

# Contains a set of functions for normal mode production/analysis.
# Includes solvers to be called by EIG.py

import numpy as np

from eigDiagnostics import diff

#=======================================================

# EIG_COEFFICIENTS
#=======================================================
def EIG_COEFFICIENTS(Ro,Re,K_nd,f_nd,U0_nd,H0_nd,gamma_nd,dy_nd,N):
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

	b1 = f_nd;

	c2 = Ro * diff(H0_nd,2,0,dy_nd);					# For zero BG flow, H0_nd=Hflat=const, i.e. c2=0
	
	c3 = Ro * H0_nd / (2. * dy_nd);

	# a3, c2 and c3 have length N, so when used in defining u and v equations in the matrix below,
	# we must add 1 to each index in order to skip out the dead gridpoint.


	# Coefficients dependent on k and y
	a1 = np.zeros((N,N),dtype=complex);		# Note: b2=a1
	c1 = np.zeros((N,N),dtype=complex);
	c4 = np.zeros((N,N),dtype=complex);
	for i in range(0,N):
		for j in range(0,N):
			a1[j,i] = 2. * np.pi * I * U0_nd[j] * K_nd[i] * Ro + 4. * np.pi**2 * K_nd[i]**2 * Ro_Re + gamma_nd;
			c1[j,i] = 2. * np.pi * I * K_nd[i] * H0_nd[j] * Ro;
			c4[j,i] = 2. * np.pi * I * U0_nd[j] * K_nd[i] * Ro;

	return a1,a2,a3,a4,b1,b4,c1,c2,c3,c4;

#=======================================================

# EIG_COEFFICIENTS2
#=======================================================
def EIG_COEFFICIENTS2(Ro,Re,K_nd,f_nd,U0_nd,H0_nd,gamma_nd,dy_nd,N):
# Here we also divide coeficients of derivatives by the relevant space-step: 
# 2*dy_nd for first-order derivatives, dy_nd**2 for second-order derivatives.

	I = np.complex(0.0,1.0);		# Define I = sqrt(-1)

	if Re == None:
		Re_inv = 0;
	else:
		Re_inv = 1. / Re;

	# Coefficients with no k- or y-dependence
	a2 = I * Re_inv / (2. * np.pi * dy_nd**2);	# Note: b3=a2, so instead of defining b3, we just use a2 in its place, saving time.
	b4 = - I / (4. * np.pi * Ro * dy_nd);

	# Coefficients with k-dependence only
	a4 = K_nd / Ro;

	# Coefficients with y-dependence only
	a3 = I * (f_nd - Ro * diff(U0_nd,2,0,dy_nd)) / (2. * np.pi * Ro);	
	b1 = - I * f_nd / (2. * np.pi * Ro);
	c2 = - I * diff(H0_nd,2,0,dy_nd) / (2. * np.pi);					
	c3 = - I * H0_nd / (4. * np.pi * dy_nd);

	# a3, c2 and c3 have length N, so when used in defining u and v equations in the matrix below,
	# we must add 1 to each index in order to skip out the dead gridpoint.

	# Coefficients dependent on k and y
	a1 = np.zeros((N,N),dtype=complex);		# Note: b2=a1
	c1 = np.zeros((N,N),dtype=complex);
	c4 = np.zeros((N,N),dtype=complex);
	for j in range(0,N):
		for i in range(0,N):
			a1[j,i] = U0_nd[j] * K_nd[i] - 2. * np.pi * I * K_nd[i]**2 * Re_inv - I * gamma_nd / (2. * np.pi * Ro);
			c1[j,i] = H0_nd[j] * K_nd[i];
			c4[j,i] = U0_nd[j] * K_nd[i];	

	return a1,a2,a3,a4,b1,b4,c1,c2,c3,c4;

#=======================================================

# BC_COEFFICIENTS
#=======================================================
def BC_COEFFICIENTS(Ro,Re,f_nd,H0_nd,dy_nd,N):
# Some extra coefficients that are required by the free-slip solver in order to impose
# the du/dy=0 boundary conditions. These terms are not needed in the no-slip solver.

	I = np.complex(0,1);

	uBC = I / (np.pi * Re * dy_nd**2);		# The extra term required for the u equation BCs.
	
	etaBC = np.zeros(2,dtype=complex);
	etaBC[0] = - I * f_nd[0] * H0_nd[0] * dy_nd * Re / (4 * np.pi * Ro);		# The extra term required for the eta equation BCs.
	etaBC[1] = - I * f_nd[N-1] * H0_nd[N-1] * dy_nd * Re / (4 * np.pi * Ro);	

	return uBC, etaBC

#=======================================================

# NO_SLIP_EIG
#=======================================================
def NO_SLIP_EIG(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,N,N2,i,VECS):
# Called by EIG.py if BC = 'NO-SLIP'.
# Also impose that eta_y = 0 on the boundaries, allowing to write 1st-order derivatives of v at the boundaries in a one-sided FD approx.

	dim = 2 * N2 + N;
	#print(dim);

	A = np.zeros((dim,dim),dtype=complex);	
				
	# First the boundary terms. Some of these could be defined in the upcoming loop,
	# but for simplicity we define all boundary terms here.

	# u equation BCs
	# South
	A[0,0] = a1[1,i] - 2. * a2;		# u[1]
	A[0,1] = a2;					# u[2]
	A[0,N2] = a3[1];				# v[1]
	A[0,2*N2+1] = a4[i];			# eta[1] 
	# North
	A[N2-1,N2-1] = a1[N2,i] - 2. * a2;	# u[N-2]
	A[N2-1,N2-2] = a2;					# u[N-3]
	A[N2-1,2*N2-1] = a3[N2]				# v[N-2] 
	A[N2-1,2*N2+N-2] = a4[i];			# eta[N-2]

	# v equation BCs
	# South
	A[N2,0] = b1[1];					# u[1]
	A[N2,N2] = a1[1,i] - 2. * a2;		# v[1]
	A[N2,N2+1] = a2;					# v[2]
	A[N2,2*N2] = - b4;					# eta[0]	
	A[N2,2*N2+2] = b4;					# eta[2]
	# North
	A[2*N2-1,N2-1] = b1[N2];					# u[N-2] 
	A[2*N2-1,2*N2-1] = a1[N2,i] - 2. * a2;		# v[N-2]
	A[2*N2-1,2*N2-2] = a2;						# v[N-3]
	A[2*N2-1,2*N2+N-1] = b4;					# eta[N-1]
	A[2*N2-1,2*N2+N-3] = - b4;					# eta[N-3]

	# eta equation BCs (Here we have to define BCs at j=2*N2,2*N2+1,2*N2+N-1,2*N2+N-2)
	# South
	A[2*N2,N2] = 2. * c3[0];			# v[1] (factor of 2 because we use one-sided FD for v_y at the boundaries)
	A[2*N2,2*N2] = c4[0,i];				# eta[0]
	A[2*N2+1,0] = c1[1,i];				# u[1]
	A[2*N2+1,N2] = c2[1];				# v[1]
	A[2*N2+1,N2+1] = c3[1];				# v[2]
	A[2*N2+1,2*N2+1] = c4[1,i];			# eta[1]
	# North
	A[2*N2+N-1,2*N2-1] = - 2. * c3[N-1];	# v[N-2]
	A[2*N2+N-1,2*N2+N-1] = c4[N-1,i];		# eta[N-1]
	A[2*N2+N-2,N2-1] = c1[N2,i]; 			# u[N-2]
	A[2*N2+N-2,2*N2-1] = c2[N2];			# v[N-2]
	A[2*N2+N-2,2*N2-2] = - c3[N2];			# v[N-3]
	A[2*N2+N-2,2*N2+N-2] = c4[N2,i];		# eta[N-2]

	# Now the inner values - two loops required: one for u,v, and one for eta
	for j in range(1,N2-1):
		# u equation
		A[j,j] = a1[j+1,i] - 2. * a2;		# u[j+1]
		A[j,j-1] = a2;						# u[j]
		A[j,j+1] = a2;						# u[j+2]
		A[j,N2+j] = a3[j+1];				# v[j+1]
		A[j,2*N2+j+1] = a4[i];				# eta[j+1] 
			
	for j in range(1,N2-1):
		# v equation
		A[N2+j,j] = b1[j+1];					# u[j+1]
		A[N2+j,N2+j] = a1[j+1,i] - 2. * a2;		# v[j+1]
		A[N2+j,N2+j-1] = a2;					# v[j]
		A[N2+j,N2+j+1] = a2;					# v[j+2]
		A[N2+j,2*N2+j] = - b4;					# eta[j]
		A[N2+j,2*N2+j+2] = b4;					# eta[j+2]

	for j in range(2,N-2):
		# eta equation
		A[2*N2+j,j-1] = c1[j,i];		# u[j] 
		A[2*N2+j,N2+j-1] = c2[j];		# v[j]
		A[2*N2+j,N2+j] = c3[j];			# v[j+1]
		A[2*N2+j,N2+j-2] = - c3[j];		# v[j-1]
		A[2*N2+j,2*N2+j] = c4[j,i];		# eta[j]
	
	val, vec = np.linalg.eig(A);

	if VECS:
		# Keep the set of eigenvalues as is; separate out the eigenvectors to be returned.
		u_vec = np.zeros((N,dim),dtype=complex);
		u_vec[1:N-1,:] = vec[0:N2,:];
		v_vec = np.zeros((N,dim),dtype=complex);
		v_vec[1:N-1,:] = vec[N:N+N2,:];
		h_vec = vec[2*N2:2*N2+N,:];	
		return val, u_vec, v_vec, h_vec;

	else:
		return val, vec;

#=======================================================

# FREE_SLIP_EIG
#=======================================================
def FREE_SLIP_EIG(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,N,N2,i,VECS):
				
# Called by RSW_1L.py if BC = 'FREE-SLIP'.
# This is the better performing FREE-SLIP option.
# This matrix should be formulated in an identical way to the one in '../core/solver'.

	dim = N2 + 2 * N;
	#print(dim);
	
	A = np.zeros((dim,dim),dtype=complex);	# For the free-slip, no-normal flow BC.

	us = 0;  ue = N-1;
	vs = N;  ve = N+N2-1;
	hs = N+N2;  he = 2*N+N2-1;

	# First the boundary terms. Some of these could be defined in the upcoming loop,
	# but for simplicity we define all boundary terms here.
	
	# u equation BCs
	# South
	A[us,us] = a1[0,i] - 2 * a2;	# u[0]		# See documentation for reasoning behind BCs here
	A[us,us+1] = 1. * a2;			# u[1]
	A[us,hs] = a4[i];				# eta[0] 
	# North
	A[ue,ue] = a1[N-1,i] - 2 * a2;	# u[N-1]
	A[ue,ue-1] = 1. * a2;			# u[N-2]
	A[ue,he] = a4[i];				# eta[N-1]

	# v equation BCs
	# South
	A[vs,us+1] = b1[1];				# u[1]
	A[vs,vs] = a1[1,i] - 2. * a2;	# v[1]
	A[vs,vs+1] = a2;				# v[2]
	A[vs,hs] = - b4;				# eta[0] 	
	A[vs,hs+2] = b4;				# eta[2]
	# North
	A[ve,ue-1] = b1[N2];			# u[N-2] 
	A[ve,ve] = a1[N2,i] - 2. * a2;	# v[N-2]
	A[ve,ve-1] = a2;				# v[N-3]
	A[ve,he] = b4;					# eta[N-1]
	A[ve,he-2] = - b4;				# eta[N-3]

	# eta equation BCs (Here we have to define BCs at j=N+N2,N+N2+1,3*N-3,3*N-4)
	# South
	A[hs,us] = c1[0,i];				# u[0]		# Again, see documentation for BC reasoning
	A[hs,vs] = 2. * c3[0];				# v[1] (factor of 2 because we use one-sided FD for v_y at the boundaries)
	A[hs,hs] = c4[0,i];				# eta[0]
	# South + 1
	A[hs+1,us+1] = c1[1,i];			# u[1]
	A[hs+1,vs] = c2[1];				# v[1]
	A[hs+1,vs+1] = c3[1];			# v[2] (factor of 1 here, back to using centered FD)
	A[hs+1,hs+1] = c4[1,i];			# eta[1]
	# North
	A[he,ue] = c1[N-1,i];			# u[N-1]
	A[he,ve] = - 2. * c3[N-1];			# v[N-2] (factor of 2 because we use one-sided FD for v_y at the boundaries)
	A[he,he] = c4[N-1,i];			# eta[N-1]
	# North - 1
	A[he-1,ue-1] = c1[N2,i];		# u[N-2]
	A[he-1,ve] = c2[N2];			# v[N-2]
	A[he-1,ve-1] = - c3[N2];		# v[N-3] (factor of 1 here, back to using centered FD)
	A[he-1,he-1] = c4[N2,i];		# eta[N-2]

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
		A[vs+j,j+1] = b1[j+1];				# u[j+1]
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
	
	val,vec = np.linalg.eig(A);

	if VECS:
		# Keep the set of eigenvalues as is; separate out the eigenvectors to be returned.
		u_vec = vec[us:ue+1,:];
		v_vec = np.zeros((N,dim),dtype=complex);
		v_vec[1:N-1,:] = vec[vs:ve+1,:];
		h_vec = vec[hs:he+1,:];	
		return val, u_vec, v_vec, h_vec;

 	else:
		return val, vec;
	
#=======================================================

# FREE_SLIP_EIG2
#=======================================================
def FREE_SLIP_EIG2(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,N,N2,i,VECS):
# Called by EIG.py if BC = 'FREE-SLIP'.
# If VECS = True, returns the eigenvector

	dim = N2 + 2 * N;
	#print(dim);

	A = np.zeros((dim,dim),dtype=complex);	# For the free-slip, no-normal flow BC.
				
	# First the boundary terms. Some of these could be defined in the upcoming loop,
	# but for simplicity we define all boundary terms here.

	us = 0;  ue = N-1;
	vs = N;  ve = N+N2-1;
	hs = N+N2;  he = 2*N+N2-1;

	# u equation BCs
	# South
	A[0,0] = a1[0,i] - 2 * a2;			# u[0] # Using BC u_y = 0.
	A[0,1] = a2;						# u[1]
	A[0,N2+N] = a4[i];					# eta[0] 
	# North
	A[N-1,N-1] = a1[N-1,i] - 2 * a2;	# u[N-1]
	A[N-1,N-2] = a2;					# u[N-2]
	A[N-1,2*N+N2-1] = a4[i];			# eta[N-1]

	# v equation BCs
	# South
	A[N,1] = b1[1];						# u[1]
	A[N,N] = a1[1,i] - 2. * a2;			# v[1]
	A[N,N+1] = a2;						# v[2]
	A[N,N+N2] = - b4;					# eta[0]	
	A[N,N+N2+2] = b4;					# eta[2]
	# North
	A[N+N2-1,N2] = b1[N2];						# u[N-2] 
	A[N+N2-1,N+N2-1] = a1[N2,i] - 2. * a2;		# v[N-2]
	A[N+N2-1,N+N2-2] = a2;						# v[N-3]
	A[N+N2-1,2*N+N2-1] = b4;					# eta[N-1]
	A[N+N2-1,2*N+N2-3] = - b4;					# eta[N-3]

	# eta equation BCs (Here we have to define BCs at j=N+N2,N+N2+1,3*N-3,3*N-4)
	# South
	A[N+N2,0] = c1[0,i];				# u[0] (See notes for arguments about BCs)
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
		A[N+j,j+1] = b1[j+1];					# u[j+1]
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
	
	val,vec = np.linalg.eig(A);

	if VECS:
		# Keep the set of eigenvalues as is; separate out the eigenvectors to be returned.
		u_vec = vec[0:N,:];
		v_vec = np.zeros((N,dim),dtype=complex);
		v_vec[1:N-1,:] = vec[N:N+N2,:];
		h_vec = vec[N+N2:2*N+N2,:];	
		return val, u_vec, v_vec, h_vec;

 	else:
		return val, vec;

#=======================================================

# FREE_SLIP_EIG3
#=======================================================
def FREE_SLIP_EIG3(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,uBC,etaBC,N,N2,i,VECS):
# Called by EIG.py if BC = 'FREE-SLIP'.
# If VECS = True, returns the eigenvector

	dim = N2 + 2 * N;
	#print(dim);

	A = np.zeros((dim,dim),dtype=complex);	# For the free-slip, no-normal flow BC.
				
	# First the boundary terms. Some of these could be defined in the upcoming loop,
	# but for simplicity we define all boundary terms here.

	# u equation BCs
	# South
	A[0,0] = a1[0,i] + uBC;			# u[0] # Using BC u_y = 0.
	A[0,1] = - uBC;					# u[1]
	A[0,N2+N] = a4[i];				# eta[0] 
	# North
	A[N-1,N-1] = a1[N-1,i] + uBC;	# u[N-1]
	A[N-1,N-2] = - uBC;				# u[N-2]
	A[N-1,2*N+N2-1] = a4[i];		# eta[N-1]

	# v equation BCs
	# South
	A[N,1] = b1[1];						# u[1]
	A[N,N] = a1[1,i] - 2. * a2;			# v[1]
	A[N,N+1] = a2;						# v[2]
	A[N,N+N2] = - b4;					# eta[0]	
	A[N,N+N2+2] = b4;					# eta[2]
	# North
	A[N+N2-1,N2] = b1[N2];						# u[N-2] 
	A[N+N2-1,N+N2-1] = a1[N2,i] - 2. * a2;		# v[N-2]
	A[N+N2-1,N+N2-2] = a2;						# v[N-3]
	A[N+N2-1,2*N+N2-1] = b4;					# eta[N-1]
	A[N+N2-1,2*N+N2-3] = - b4;					# eta[N-3]

	# eta equation BCs (Here we have to define BCs at j=N+N2,N+N2+1,3*N-3,3*N-4)
	# South
	A[N+N2,0] = c1[0,i] + etaBC[0];		# u[0] (See notes for arguments about BCs)
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
		A[N+j,j+1] = b1[j+1];					# u[j+1]
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
	
	val,vec = np.linalg.eig(A);

	if VECS:
		# Keep the set of eigenvalues as is; separate out the eigenvectors to be returned.
		u_vec = vec[0:N,:];
		v_vec = np.zeros((N,dim),dtype=complex);
		v_vec[1:N-1,:] = vec[N:N+N2,:];
		h_vec = vec[N+N2:2*N+N2,:];	
		return val, u_vec, v_vec, h_vec;

 	else:
		return val, vec;

#=======================================================

# eigDecomp
#=======================================================
def eigDecomp(VEC,Phi):
# A fucntion that calculates the weights for an eigenmode decomposition of a forced solution as calculated by RSW_1L.py
# The eigenmodes and forced solutions (in (k,y)-space) can be calculated anew (NEW) or from a numpy array file (FILE),
# or an eigenvector can be passed in the place of VEC. 
# Calculated is the decomposition for a single wavenumber value, i.e. to get the full decomposition, eigDecomp must be ran N times.
# The output depends on VEC. 

	theta = np.linalg.solve(VEC,Phi);

	return theta;


		

