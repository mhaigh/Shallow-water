# modes.py

# Look at eigenmodes for a given background flow, and see what wavenumbers are selected.

#====================================================================================================

import numpy as np
import matplotlib.pyplot as plt

from eig import eigSolver, eigDiagnostics
from core import diagnostics, solver, forcing, energy, PV
from output import output, output_read

from inputFile import *

#====================================================================================================

path = "/home/mike/Documents/GulfStream/RSW/DATA/1L/EIG/256/04/"
 

def EIG_PROJ_main(dim):
	'''
	This function projects the SW solution onto eigenmodes as outputted by EIG.py.
	The aim is to look at the number of modes required for certain levels of accuracy.
	To do this we can look at the error in the whole solution (x,y), or look at the
	error at specific wavenumbers (e.g. wavenumbers where there is a lot of weight).  
	'''

	# The 1L SW solution
	#====================================================

	I = np.complex(0.0,1.0)

	a1,a2,a3,a4,b4,c1,c2,c3,c4 = solver.SOLVER_COEFFICIENTS(Ro,Re,K_nd,f_nd,U0_nd,H0_nd,omega_nd,gamma_nd,dy_nd,N)
	# Define the solution in (k,y)-space - can be from FILE or a NEW run.
	solution = solver.FREE_SLIP_SOLVER(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,Ro*Ftilde1_nd,Ro*Ftilde2_nd,Ftilde3_nd,N,N2) / AmpF_nd

	utilde_nd, vtilde_nd, etatilde_nd = solver.extractSols(solution,N,N2,BC)
	u, v, h = solver.SPEC_TO_PHYS(utilde_nd,vtilde_nd,etatilde_nd,T_nd,dx_nd,omega_nd,N)
	u = np.real(u); v = np.real(v); h = np.real(h)

	print('solved')

	#====================================================

	# Perform decomposition	and build projection.
	#====================================================

	theta = np.zeros((dim,N),dtype=complex)
	theta_abs = np.zeros((dim,N))
	proj = np.zeros((dim,N),dtype=complex)
	projk = np.zeros((dim),dtype=complex)
	for i in range(0,N):

		print(i)

		k = K_nd[i]

		# Load modes
		path = '/home/mike/Documents/GulfStream/RSW/DATA/1L/EIG/256/04/'
		#path = ''
		ncFile = path + 'RSW1L_Eigenmodes_k' + str(int(k)) + '_N257.nc'	
		val, vec, count = output_read.ncReadEigenmodes(ncFile)

		Phi = solution[:,i]		# 1. Assign the solution corresponding to wavenumber k=K_nd[ii].

		theta_tmp = np.linalg.solve(vec,Phi) 				# 2.
		theta_abs_tmp = np.abs(theta_tmp)
		dom_index = np.argsort(-theta_abs_tmp)			# 3. The indices of the modes, ordered by 'dominance'.
		theta[:,i] = theta_tmp[dom_index]
		theta_abs[:,i] = np.abs(theta[:,i])
		vec = vec[:,dom_index]
		# Now loop over each mode (at wavenumber k)

		
		#np.save('mode1',vec[:,0]); np.save('mode2',vec[:,1])
		#np.save('mode3',vec[:,2]); np.save('mode4',vec[:,3])
		#theta0123 = theta[0:4,i]; np.save('theta0123',theta0123)

		for mi in range(0,100):
			proj[:,i] = proj[:,i] + theta[mi,i] * vec[:,mi]
			
	theta_abs_tot = np.sum(theta_abs,axis=0)

	utilde_proj = np.zeros((N,N),dtype=complex)
	vtilde_proj = np.zeros((N,N),dtype=complex)
	etatilde_proj = np.zeros((N,N),dtype=complex)

	# Separate projection into flow components.
	for j in range(0,N):
		utilde_proj[j,:] = proj[j,:]
		etatilde_proj[j,:] = proj[N+N2+j,:]
	for j in range(0,N2):
		vtilde_proj[j+1,:] = proj[N+j,:]


	u_proj, v_proj, eta_proj = solver.SPEC_TO_PHYS(utilde_proj,vtilde_proj,etatilde_proj,T_nd,dx_nd,omega_nd,N)
	u_proj = np.real(u_proj); v_proj = np.real(v_proj); eta_proj = np.real(eta_proj)

	return u, v, h, u_proj, v_proj, eta_proj, theta, theta_abs_tot, dom_index

#====================================================================================================

#u, v, h, u_proj, v_proj, eta_proj, theta, theta_abs_tot, dom_index = EIG_PROJ_main(dim)

#np.save('theta',theta); np.save('theta_abs_tot',theta_abs_tot); np.save('dom_index',dom_index)

theta = np.load('theta.npy'); theta_abs_tot = np.load('theta_abs_tot.npy'); dom_index = np.load('dom_index.npy')

plt.plot(K_nd,theta_abs_tot); plt.show()


k = - 23

thetak = theta[:,k]

# theta is already ordered by amplitude, need to order val or vec if used.

path = '/home/mike/Documents/GulfStream/RSW/DATA/1L/EIG/256/04/'
ncFile = path + 'RSW1L_Eigenmodes_k' + str(int(k)) + '_N257.nc'	
val, vec, count = output_read.ncReadEigenmodes(ncFile)

val = val[dom_index]
vec = vec[:,dom_index]

freq = np.real(val); growth = np.imag(val)

period = T_adv / freq # Reinstate dimension (units s)
period_days = period / (24. * 3600.)

print(period_days)
quit()
utilde_proj = np.zeros((N,N),dtype=complex)
vtilde_proj = np.zeros((N,N),dtype=complex)
etatilde_proj = np.zeros((N,N),dtype=complex)

for mode in range(0,5):
	# Separate projection into flow components.
	for j in range(0,N):
		utilde_proj[j,k] = vec[j,mode]
		etatilde_proj[j,k] = vec[N+N2+j,mode]
	for j in range(0,N2):
		vtilde_proj[j+1,k] = vec[N+j,mode]

	u_proj, v_proj, eta_proj = solver.SPEC_TO_PHYS(utilde_proj,vtilde_proj,etatilde_proj,T_nd,dx_nd,omega_nd,N)
	u_proj = np.real(u_proj)

	print(period_days[mode])

	plt.contourf(u_proj[:,:,ts]); plt.colorbar(); plt.show()


















