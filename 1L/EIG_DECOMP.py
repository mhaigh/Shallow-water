# EIG_DECOMP.py
#=======================================================

# This code decomposes a 1L SW solution, as produced by RSW_1L.py into into eigenmodes components,
# producing a set of weights corresponding the to set of eigenmodes found by runnning modules located in eigSolver.py
# First it defines the solution, which can either be defined from a file (FILE) or by running RSW_1L.py (NEW).

#====================================================

import sys

import numpy as np
#import matplotlib.pyplot as plt
import itertools as it

from eig import eigSolver, eigDiagnostics
from core import diagnostics, solver, forcing, energy, PV
from output import output, output_read

from inputFile import *

#====================================================

path = '/home/mike/Documents/GulfStream/RSW/DATA/1L/EIG/256/04/'
		
I = np.complex(0,1)

Nl = N//2 + 1 

for i in range(0,N):

	k = K_nd[i]
	ncFile = path + 'RSW1L_Eigenmodes_k' + str(int(k)) + '_N257.nc'	
	
	val, vec, count = output_read.ncReadEigenmodes(ncFile)

	u_vec, v_vec, h_vec = eigDiagnostics.vec2vecs(vec,N,dim,BC)	
	print(np.shape(u_vec))
	#print(count)

	#sys.exit()
	
	count, count2, ratio, i_count = eigDiagnostics.orderEigenmodes(u_vec,x_nd,k,N,dim)
	#count, i_count = eigDiagnostics.orderEigenmodes2(vec,val,N,False)
	
	count = count[i_count]
	count2 = count2[i_count]
	ratio = ratio[i_count]
	vec = vec[:,i_count]
	val = val[i_count]

	u_vec, v_vec, h_vec = eigDiagnostics.vec2vecs(vec,N,dim,BC)	

	# Each eigenmode is currently a unit vector, but we normalise so that each mode contains unit energy.
	#==
	
	# Extract the three components.


	#print(np.shape(u_vec))

	u = np.zeros((N,N))
	for wi in range(0,0):
		
		print(count[wi]); print(count2[wi]); print(ratio[wi]); print(T_adv/(np.real(val[wi])*24*3600))
		print('---')
		theta = 2*np.pi*k*x[0]
		cosx = np.cos(theta); sinx = np.sin(theta)
		for i in range(0,N):
			u[:,i] = np.real(u_vec[:,wi] * np.exp(2 * np.pi* I * k * x_nd[i]))
		theta = 2*np.pi*k*x_nd[0]
		cosx = np.cos(theta); sinx = np.sin(theta)
		ux = np.real(u_vec[:,wi]) * cosx - np.imag(u_vec[:,wi]) * sinx

		uxx = np.zeros((Nl))
		ft = np.abs(np.fft.fft(ux))
		uxx[0] = ft[0]
		uxx[1:Nl] = ft[1:Nl] + ft[N:N-Nl:-1]

		plt.subplot(131)
		plt.plot(uxx); plt.grid(); #plt.xlim(-5,5)
		plt.subplot(132)
		plt.plot(ux); 
		plt.subplot(133)
		plt.contourf(u); plt.show()
	sys.exit()




