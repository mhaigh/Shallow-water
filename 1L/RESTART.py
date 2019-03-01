# RESTART.py
#=======================================================

# This code works in the same way as the master file RSW_1L.py, but does not solve the SW model.
# Instead it takes saved solution (in .npy or .nc format), and then can perform all necessary diagnostics.
# Should ensure that the input file is defined in the same way as it was for originally producing the saved npy files.

#=======================================================

import sys

import numpy as np

from core import diagnostics, PV, forcing, solver, energy, corr
from output import plotting_bulk

from inputFile import *

#N = 513

#=======================================================

test = 'N' # U0, y0, N, h 
# Plot with differing U0, y0, N or h.

#samples = ['16']
samples = ['-08','08','32']

#samples = ['-1','0','1.5']
#samples = ['-1','1.5'] 				# sigma samples
#samples = ['1.5']

#samples = ['08']

ns = len(samples)

# Initialise u,v,eta,PV,P
u = np.zeros((N,N,ns),dtype=complex)
v = np.zeros((N,N,ns),dtype=complex)
h = np.zeros((N,N,ns),dtype=complex)
q = np.zeros((N,N,ns))
P = np.zeros((N,N+1,ns))
P_xav = np.zeros((N,ns))

K = np.zeros((N,N,ns))
M = np.zeros((N,N,ns))
N_ = np.zeros((N,N,ns))

#=======================================================

# First define all necessary terms to be plotted.
for si in range(0,ns):
	
	sample = samples[si]

	#=======================================================

	# For each U0 or sigma value, we need to first load the saved solutions
	if test == 'U0':
		path = '/media/mike/Seagate Expansion Drive/Documents/GulfStream/RSW/DATA/1L/PAPER1/UNIFORM/'
		U0_load = str(sample)
		u_tmp = np.load(path + 'u_U0='+U0_load+'.npy')[:,:,ts]
		v_tmp = np.load(path + 'v_U0='+U0_load+'.npy')[:,:,ts]
		h_tmp = np.load(path + 'eta_U0='+U0_load+'.npy')[:,:,ts]
		P[:,:,si] = np.load(path + 'P_U0='+U0_load+'.npy')
		P_xav[:,si] = np.trapz(P[:,:,si],x_nd,dx_nd,axis=1);

		# Last step: redefine U0 and H0 for each sample
		U0, H0 = BG_state.BG_uniform(float(sample)/100.,Hflat,f0,beta,g,y,N);
		U0_nd = U0 / U
		H0_nd = H0 / chi
	
	#=======================================================

	elif test == 'y0':
		path = '/media/mike/Seagate Expansion Drive/Documents/GulfStream/RSW/DATA/1L/PAPER1/GAUSSIAN/'
		print(AmpF_nd)
		if si == 1:
			u_tmp = np.load(path + 'u_y0='+sample+'.npy')[:,:,ts] / AmpF_nd
			v_tmp = np.load(path + 'v_y0='+sample+'.npy')[:,:,ts] / AmpF_nd
			h_tmp = np.load(path + 'eta_y0='+sample+'.npy')[:,:,ts] / AmpF_nd
		else:
			u_tmp = np.load(path + 'u_y0='+sample+'.npy')[:,:,ts]
			v_tmp = np.load(path + 'v_y0='+sample+'.npy')[:,:,ts]
			h_tmp = np.load(path + 'eta_y0='+sample+'.npy')[:,:,ts]
		P[:,:,si] = np.load(path + 'P_y0='+sample+'.npy')
		P_xav[:,si] = np.trapz(P[:,:,si],x_nd,dx_nd,axis=1);
	
	#=======================================================

	# For each U0 or sigma value, we need to first load the saved solutions
	elif test == 'N':

		if True:
			path = '/media/mike/Seagate Expansion Drive/Documents/GulfStream/RSW/DATA/1L/PAPER1/UNIFORM/'

			U0_load = str(sample)
			U0, H0 = BG_state.BG_uniform(float(sample)/100.,Hflat,f0,beta,g,y,N);

			u_tmp = np.load(path + 'u_U0='+U0_load+'.npy') #/ AmpF_nd
			v_tmp = np.load(path + 'v_U0='+U0_load+'.npy') #/ AmpF_nd

			u = np.real(u_tmp)
			v = np.real(v_tmp)
		
			K[:,:,si] = corr.K(u,v,T_nd)
			M[:,:,si] = corr.Mnorm(u,v,T_nd)
			N_[:,:,si] = corr.Nnorm(u,v,T_nd)


		else: 

			path = '/media/mike/Seagate Expansion Drive/Documents/GulfStream/RSW/DATA/1L/PAPER1/GAUSSIAN/'
		
			u_tmp = np.load(path + 'u_y0='+sample+'.npy')# / AmpF_nd
			v_tmp = np.load(path + 'v_y0='+sample+'.npy') #/ AmpF_nd
			h_tmp = np.load(path + 'eta_y0='+sample+'.npy')# / AmpF_nd
			P[:,:,si] = np.load(path + 'P_y0='+sample+'.npy')
			P_xav[:,si] = np.trapz(P[:,:,si],x_nd,dx_nd,axis=1);

			# Last step: redefine U0 and H0 for each sample
		
			U0, H0 = BG_state.BG_Gaussian(Umag,sigma,JET_POS,Hflat,f0,beta,g,y,Ly*L,N)


		U0_nd = U0 / U
		H0_nd = H0 / chi

	#=======================================================

	elif test == 'S':
		path = '/home/mike/Documents/GulfStream/RSW/DATA/1L/STOCH/'
		#path = '/media/mike/Seagate Expansion Drive/Documents/GulfStream/RSW/DATA/1L/PAPER1/STOCH/'
		U0_load = str(sample)
		u_tmp = np.load(path + 'u.npy')[:,:,ts]
		v_tmp = np.load(path + 'v.npy')[:,:,ts]
		h_tmp = np.load(path + 'h.npy')[:,:,ts]

		P[:,:,si] = np.load(path + 'P.npy')
		P_xav[:,si] = np.trapz(P[:,:,si],x_nd,dx_nd,axis=1);


		# Last step: redefine U0 and H0 for each sample
		U0, H0 = BG_state.BG_uniform(0.08,Hflat,f0,beta,g,y,N);
		U0_nd = U0 / U
		H0_nd = H0 / chi	

	#=======================================================	

	# For plotting SSH.
	elif test == "h":

		path = '/media/mike/Seagate Expansion Drive/Documents/GulfStream/RSW/DATA/1L/PAPER1/UNIFORM/'
		U0_load = str(sample)
		
		h[:,:,si] = np.load(path + 'eta_U0='+U0_load+'.npy')[:,:,ts]


	#=======================================================

	# Use this loop if we need full flow, otherwise ignore.
	if False:
		# Calculate full flows.
		h_full = np.zeros((N,N))
		u_full = np.zeros((N,N))
		for j in range(0,N):
			h_full[j,:] = np.real(h_tmp[j,:]) + H0_nd[j]
			u_full[j,:] = np.real(u_tmp[j,:]) + U0_nd[j]

		# Snapshot of PV anomaly
		q[:,:,si] = PV.PV_instant(np.real(u_tmp),np.real(v_tmp),np.real(h_tmp),u_full,h_full,H0_nd,U0_nd,N,Nt,dx_nd,dy_nd,f_nd,Ro)

		# Now we have snapshot of solution, snapshot of PV, and the footprint.
		ff = 1.
		u[:,:,si] = ff*u_tmp; v[:,:,si] = ff*v_tmp; h[:,:,si] = ff*h_tmp;
	
print('Plotting...')

#=======================================================

# Second, create all plots.

# Set the desired loop(s) to True or False as needed. 
# Most loops will call bulk plotting functions.

# Solutions
if False:
	fig, axes = plt.subplots(nrows=ns,ncols=3,figsize=(22,7*ns))

	for si in range(0,ns):
		sample = samples[si]
		if test == 'U0':
			string = r'$U_{0} = ' + str(float(sample)/100) + '$'	
		elif test == 'y0':
			if sample == '0':
				string = r'$y_{0}=0$'
			elif sample == '-1':
				string = r'$y_{0}=-\sigma$'
			else:
				string = r'$y_{0}=' + sample + '\sigma$'
		else:
			string = r'$U_{0} = 0.08$'
		U0_str = 'U0 = ' + str(U0)
		plotting_bulk.plotSolutions(np.real(u[:,:,si]),np.real(v[:,:,si]),np.real(h[:,:,si]),N,x_grid,y_grid,si,ns,string)
		plt.tight_layout(pad=0.3, w_pad=0.2, h_pad=1.0);
		plt.savefig('fig0.png')

# Solutions, h only
if False:

	h = np.real(h)

	fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(22,7))

	strings = [r'$U_{0} = ' + str(float(sample)/100) + '$' for sample in samples] 

	plotting_bulk.plotSSH(h,N,x_grid,y_grid,strings)
	plt.tight_layout(pad=0.3, w_pad=0.2, h_pad=1.0);
	plt.savefig('fig0.png')
	


# Phase & Amp
if False:
	fig, axes = plt.subplots(nrows=ns,ncols=3,figsize=(22,2*7*ns))

	for si in range(0,ns):
		sample = samples[si]
		if test == 'U0':
			string = r'$U_{0} = ' + str(float(sample)/100) + '$'
		elif test == 'y0':
			if sample == '0':
				string = r'$y_{0}=0$'
			elif sample == '-1':
				string = r'$y_{0}=-\sigma$'
			else:
				string = r'$y_{0}=' + sample + '\sigma$'
		else:
			string = r'$U_{0} = 0.08$'
		U0_str = 'U0 = ' + str(U0)
		plotting_bulk.plotSolutionsAmpPhase(u[:,:,si],v[:,:,si],h[:,:,si],N,x_grid,y_grid,si,ns,string,fig)
		plt.tight_layout(pad=0.3, w_pad=0.2, h_pad=1.0);
		plt.savefig('fig1.png');


# Footprints
if False:
	fig, axes = plt.subplots(nrows=ns,ncols=3,figsize=(22,7*ns))

	for si in range(0,ns):
		sample = samples[si]
		if test == 'U0':
			string = r'$U_{0} = ' + str(float(sample)/100) + '$'
		elif test == 'y0':
			if sample == '0':
				string = r'$y_{0}=0$'
			elif sample == '-1':
				string = r'$y_{0}=-\sigma$'
			else:
				string = r'$y_{0}=' + sample + '\sigma$'
		else:
			string = r'$U_{0} = 0.08$'

		plotting_bulk.fp_PV_plot(q[:,:,si],P[:,:,si],P_xav[:,si],N,x_grid,y_grid,y_nd,si,ns,string)
		plt.tight_layout(pad=0.3, w_pad=0.1, h_pad=0.6);
		plt.savefig('fig2.png');


# Footprint and Nyy
if False:
	
	u_tmp = np.real(u_tmp)
	v_tmp = np.real(v_tmp)

	print(np.shape(u_tmp))
	N_ = corr.N(u_tmp,v_tmp,T_nd)

	Nyy = diagnostics.diff(diagnostics.diff(N_,0,0,dy_nd),0,0,dy_nd)

	N_ = diagnostics.zonalMean(Nyy,x_nd)


	plt.subplot(121)
	plt.plot(N_); 
	plt.subplot(122)
	plt.plot(H0_nd); plt.show()

	N_ = 1.e5 * N_ / H0_nd



	plt.plot(N_); plt.show()

	dx_nd = x_nd[1] - x_nd[0]
	P = np.trapz(P,x_nd,dx_nd,axis=1) /  1.e3


	y_ticks = (0.,0.125,0.25,0.375,0.5)
	lim = [-.125,.125]
	y_ticks = [lim[0], (lim[1]+lim[0])/2, lim[1]]

	plotting_bulk.plotPandN(P,N_,y_nd,lim,y_ticks)
	
# Eddy shape parameters
if True:

	fig, axes = plt.subplots(nrows=ns,ncols=3,figsize=(22,7*ns))

	for si in range(0,ns):
		sample = samples[si]

		if False:
			if sample == '0':
				string = r'$y_{0}=0$'
			elif sample == '-1':
				string = r'$y_{0}=-\sigma$'
			else:
				string = r'$y_{0}=' + sample + '\sigma$'
		else:
			string = r'$U_{0} = ' + str(float(sample)/100) + '$'
	

		plotting_bulk.plotKMN(K[:,:,si],M[:,:,si],N_[:,:,si],x_grid,y_grid,N,si,ns,string)

		plt.tight_layout(pad=0.3, w_pad=0.1, h_pad=0.6);
		plt.savefig('fig2.png');
	


		



