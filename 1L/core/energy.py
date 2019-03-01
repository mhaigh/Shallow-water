# energy
#=========================================================

# A set of energy-related functions

#=========================================================

import numpy as np
import matplotlib.pyplot as plt

from diagnostics import diff, extend, timeAverage, timeDep

#=========================================================

# energy_BG
def energy_BG(U0_nd,H0_nd,Ro,y_nd,dy_nd,N):
# A function that calculates the total energy E = KE + PE of the background state - 
# a useful quantity for when calculating the forcing induced energy or energy of eigenmodes.

	# First kinetic energy, KE = 0.5 * U**2 * H (dimensionless, no v term)
	KE_BG = 0.5 * U0_nd**2 * H0_nd;					# KE in terms of y (uniform in x)
	KE_BG_tot = np.trapz(KE_BG,y_nd,dy_nd);		# Total KE in the 2D domain (x-length of domain is 1)
	
	# Next the potential energy, PE = 0.5 * H**2 / Ro
	PE_BG = 0.5 * H0_nd**2 / Ro;				# PE in terms of y
	PE_BG_tot = np.trapz(PE_BG,y_nd,dy_nd);		# Total PE in the 2D domain

	return KE_BG, KE_BG_tot, PE_BG, PE_BG_tot;

#=========================================================

# KE_from_spec
def KE_from_spec(u_tilde,v_tilde,eta_tilde,k_nd,x_nd,y_nd,Nt,N,output):
# A function that takes the spectral representation of the solution at only one wavenumber k, indexed by i,
# and calculates the kinetic energy (KE) at that wavenumber.
# Because of the linearity of the system, the KE outputted from this function can be summed over all wavenumbers
# to produce the total KE.
# All values are dimensionless - see documentation for scaling arguments.

# Options for output:
# 1. 'av': outputs only the KE temporal average (as a function of x and y);
# 2. 'av_tot': outputs the temporal and spatial average;
# 2. 'full': outputs the full KE, as a function of x, y and t.
 
	I = np.complex(0,1);
	
	# We want the KE at a discrete set of times over one forcing/solution/eigenmode period.
	# Note that this calculation can be made independent of this period/frequency;
	# instead we only need to sample 'times' from a interval of unit length, with Nt entries,
	# the frequency omega cancels out with the period T.

	u = np.zeros((N,N+1,Nt));
	v = np.zeros((N,N+1,Nt));
	eta = np.zeros((N,N+1,Nt));

	omega_t = np.linspace(0,1,Nt);

	for ti in range(0,Nt+1):
		for j in range(0,N):
			u[j,:,ti] = np.real(u_tilde[j] * np.exp(2 * np.pi * (k_nd * x_nd - omega_t[ti])));
			v[j,:,ti] = np.real(v_tilde[j] * np.exp(2 * np.pi * (k_nd * x_nd - omega_t[ti])));
			eta[j,:,ti] = np.real(eta_tilde[j] * np.exp(2 * np.pi * (k_nd * x_nd - omega_t[ti])));
	
	# 1. Temporally averaged KE
	if output == 'av':
		KE_av = 0.5 * (u[:,:,0]**2 + v[:,:,0]**2) * eta[:,:,0];
		for ti in range(1,Nt):
			KE_av = KE_av + 0.5 * (u[:,:,ti]**2 + v[:,:,ti]**2) * eta[:,:,ti];
		KE_av = KE_av / Nt;
		return KE_av;

	# 2. Temporal and spatial average of KE
	elif output == 'av_tot':
		dx_nd = x_nd[1] - x_nd[0];
		dy_nd = y_nd[1] - y_nd[0];
		KE = 0.5 * (u**2 + v**2) * eta;
		KE_av = np.trapz(np.trapz(KE,x_nd,dx_nd,1),y_nd,dy_nd,0);
		KE_av_tot = np.trapz(KE_av,T_nd,dt_nd,0);

		return KE_av_tot;
	
	# 3. Time-dependent KE
	elif output == 'full':
		KE = np.zeros((N,N,Nt));
		for ti in range(0,Nt):
			KE[:,:,ti] = 0.5 * (u[:,:,ti]**2 + v[:,:,ti]**2) * eta[:,:,ti];
		return KE;

	else:
		import sys
		sys.exit('Invalid output selection; must be "av", "av_tot" or "full".');

#=========================================================

# PE_from_spec
def PE_from_spec(eta_tilde,Ro,k_nd,x_nd,y_nd,Nt,N,output):
# A function that takes the spectral representation of the solution at only one wavenumber k, indexed by i,
# and calculates the potential energy (PE) at that wavenumber.
# Because of the linearity of the system, the PE outputted from this function can be summed over all wavenumbers
# to produce the total PE.
# All values are dimensionless - see documentation for scaling arguments.

# Options for output:
# 1. 'av': outputs only the PE temporal average (as a function of x and y);
# 2. 'av_tot': outputs the temporal and spatial average;
# 2. 'full': outputs the full PE, as a function of x, y and t.
 
	I = np.complex(0,1);
	
	# We want the KE at a discrete set of times over one forcing/solution/eigenmode period.
	# Note that this calculation can be made independent of this period/frequency;
	# instead we only need to sample 'times' from a interval of unit length, with Nt entries,
	# the frequency omega cancels out with the period T.

	eta = np.zeros((N,N,Nt));

	omega_t = np.linspace(0,1,Nt);

	for ti in range(0,Nt):
		for j in range(0,N):
			eta[j,:,ti] = np.real(eta_tilde[j] * np.exp(2 * np.pi * (k_nd * x_nd[0:N] - omega_t[ti])));
	
	# 1. Temporally averaged PE
	if output == 'av':
		PE_av = 0.5 * eta[:,:,0] / Ro;
		for ti in range(1,Nt):
			PE_av = PE_av + 0.5 * (u[:,:,ti]**2 + v[:,:,ti]**2) * eta[:,:,ti];
		PE_av = PE_av / Nt;
		return PE_av;

		# 1. Temporal and spatial average of PE
	elif output == 'av_tot':
		dx_nd = x_nd[1] - x_nd[0];
		dy_nd = y_nd[1] - y_nd[0];
		PE = 0.5 * eta[:,:,0]**2 / Ro;
		PE_av_tot = np.trapz(np.trapz(PE,x_nd[0:N],dx_nd,1),y_nd,dy_nd,0);
		for ti in range(1,Nt):
			PE = 0.5 * eta[:,:,ti]**2 / Ro;
			PE_av_tot = PE_av_tot + np.trapz(np.trapz(PE,x_nd[0:N],dx_nd,1),y_nd,dy_nd,0);
		PE_av_tot = PE_av_tot / Nt;
		return PE_av_tot;
	
	# 3. Time-dependent PE
	elif output == 'full':
		PE = np.zeros((N,N,Nt));
		for ti in range(0,Nt):
			PE[:,:,ti] = 0.5 * eta[:,:,ti]**2 / Ro;
		return PE;

	else:
		import sys
		sys.exit('Invalid output selection; must be "av", "av_tot" or "full".');

#=========================================================

# E_from_spec
def E_from_spec(u_tilde,v_tilde,eta_tilde,Ro,k_nd,x_nd,y_nd,T_nd,Nt,omega_nd,N,output):
# A function that takes the spectral representation of the solution at only one wavenumber k, indexed by i,
# and calculates the energy (both KE and PE) at that wavenumber.
# Because of the linearity of the system, the energy outputted from this function can be summed over all wavenumbers
# to produce the energy.
# All values are dimensionless - see documentation for scaling arguments.

# Options for output:
# 1. 'av': outputs only the temporal average (as a function of x and y);
# 2. 'av_tot': outputs the temporal and spatial average;
# 2. 'full': outputs the full energy, as a function of x, y and t.
 
	I = np.complex(0,1);
	
	# We want the KE at a discrete set of times over one forcing/solution/eigenmode period.
	# Note that this calculation can be made independent of this period/frequency;
	# instead we only need to sample 'times' from a interval of unit length, with Nt entries,
	# the frequency omega cancels out with the period T.

	u = np.zeros((N,N+1,Nt+1));
	v = np.zeros((N,N+1,Nt+1));
	eta = np.zeros((N,N+1,Nt+1));	

	dt_nd = T_nd[1] - T_nd[0];

	for ti in range(0,Nt+1):
		for j in range(0,N):
			u[j,:,ti] = np.real(u_tilde[j] * np.exp(2 * np.pi * I * (k_nd * x_nd - omega_nd * T_nd[ti])));
			v[j,:,ti] = np.real(v_tilde[j] * np.exp(2 * np.pi * I *(k_nd * x_nd - omega_nd * T_nd[ti])));
			eta[j,:,ti] = np.real(eta_tilde[j] * np.exp(2 * np.pi * I * (k_nd * x_nd - omega_nd * T_nd[ti])));
	
	# 1. Temporally averaged KE and PE
	if output == 'av':
		KE_av = 0.5 * (u[:,:,0]**2 + v[:,:,0]**2) * eta[:,:,0];
		PE_av = 0.5 * eta[:,:,0] / Ro;
		for ti in range(1,Nt):
			KE_av = KE_av + 0.5 * (u[:,:,ti]**2 + v[:,:,ti]**2) * eta[:,:,ti];
			PE_av = PE_av + 0.5 * (u[:,:,ti]**2 + v[:,:,ti]**2) * eta[:,:,ti];
		KE_av = KE_av / Nt;
		PE_av = PE_av / Nt;
		return KE_av, PE_av;

		# 1. Temporal and spatial average of KE and PE
	elif output == 'av_tot':
		dx_nd = x_nd[1] - x_nd[0];
		dy_nd = y_nd[1] - y_nd[0];

		KE = 0.5 * (u**2 + v**2) * eta;
		KE_av = np.trapz(np.trapz(KE,x_nd,dx_nd,1),y_nd,dy_nd,0);
		KE_av_tot = np.trapz(KE_av,T_nd,dt_nd,0) / T_nd[Nt];

		PE = 0.5 * eta**2 / Ro;
		PE_av = np.trapz(np.trapz(PE,x_nd,dx_nd,1),y_nd,dy_nd,0);
		plt.plot(PE_av);
		plt.show();
		PE_av_tot = np.trapz(PE_av,T_nd,dt_nd,0) / T_nd[Nt];

		return KE_av_tot, PE_av_tot;
	
	# 3. Time-dependent KE and PE
	elif output == 'full':
		KE = np.zeros((N,N,Nt));
		PE = np.zeros((N,N,Nt));
		for ti in range(0,Nt):
			KE[:,:,ti] = 0.5 * (u[:,:,ti]**2 + v[:,:,ti]**2) * eta[:,:,ti];
			PE[:,:,ti] = 0.5 * eta[:,:,ti]**2 / Ro;
		return KE, PE;

	else:
		import sys
		sys.exit('Invalid output selection; must be "av", "av_tot" or "full".');

#=========================================================

# E_anomaly_EIG
def E_anomaly_EIG(u_vec,v_vec,eta_vec,H0_nd,U0_nd,Ro,y_nd,dy_nd):
# This function calculates the energy anomaly as derived in the notes.

	E1 = 0.25 * H0_nd * (np.real(u_vec)**2 + np.imag(u_vec)**2 + np.real(v_vec)**2 + np.imag(v_vec)**2);
	E2 = 0.5 * U0_nd * (np.real(u_vec) * np.real(eta_vec) + np.imag(u_vec) * np.imag(eta_vec));
	E3 = 0.25 * (np.real(eta_vec)**2 + np.imag(eta_vec)**2) / Ro;

	E = E1 + E2 + E3;

	E = np.trapz(E,y_nd,dy_nd,0);

	return E;

#=========================================================
# KE
def KE(u,v,h,x_nd,y_nd,dx_nd,dy_nd,N):
# Outputs the KE

	KE = 0.5 * (u**2 + v**2) * h

	KE_tot = np.trapz(np.trapz(KE,x_nd[0:N],dx_nd,axis=1),y_nd,dy_nd,axis=0)

	return KE, KE_tot

#=========================================================

# PE
def PE(h_full,Ro,x_nd,y_nd,dx_nd,dy_nd,N):
# Outputs the KE at a moment in time, by taking as input as time-snapshot of the solution u,v,eta.

	PE = 0.5 * h_full**2 / Ro

	PE_tot = np.trapz(np.trapz(PE,x_nd[0:N],dx_nd,axis=1),y_nd,dy_nd,axis=0)

	return PE, PE_tot

#=========================================================

# Flux
def flux(E,u,v):
	'''Calculate fluxes of energy.'''

	uE = u * E
	vE = v * E

	return uE, vE

#=========================================================

def conv(uE,vE,T_nd,Nt,x_nd,dx_nd,y_nd,dy_nd):
	'''Time-mean convergence of energy fluxes.'''

	uE_av = timeAverage(uE,T_nd,Nt)
	vE_av = timeAverage(vE,T_nd,Nt)

	Econv = - diff(uE_av,1,1,dx_nd) - diff(vE_av,0,0,dy_nd)
	Econv = extend(Econv)
		
	Econv_xav = np.trapz(Econv,x_nd,dx_nd,axis=1) # / 1 not required.

	return Econv, Econv_xav
	
#=========================================================

def KEspectrum(u,v,K,y,T,Nt,N):
	'''Return kinetic energy (not column-averaged) spectrum, taking spectral solution as input'''

	from scipy.signal import welch
	
	dy = y[1] - y[0]
	dt = T[1] - T[0]

	N4 = N//4

	u4 = u[N4:N-N4,N4:N-N4,:]
	v4 = v[N4:N-N4,N4:N-N4,:]
	y4 = y[N4:N-N4]

	KEspec = 0.5 * (u4**2 + v4**2)
	freq, KEspec = welch(KEspec,axis=1)
	print(np.shape(KEspec))
 	
	KEspec = timeAverage(KEspec,T,Nt)
	KEspec = np.trapz(KEspec,y4,dy,axis=0)
	
	#freq, KEspec = welch(KEspec)


	plt.plot(KEspec); plt.show()
	np.save('KEouter', KEspec)

	return KEspec

#=========================================================

def budgetForcing(u,v,h,F1,F2,F3,Ro,N,T,omega,Nt):
	''' Calculate time-dependent and time-mean energy budget due to forcing.
	Make sure to use full u, v and h fields. Contribution from 
	background terms will be removed after time averaging.
	This will produce the same time-mean as the below function.'''

	F1t = timeDep(F1,T,omega,Nt)
	F2t = timeDep(F2,T,omega,Nt)
	F3t = timeDep(F3,T,omega,Nt)

	Ef = h * u * F1t + h * v * F2t + (0.5 * (u**2 + v**2) + h / Ro) * F3t

	Ef_av = timeAverage(Ef,T,Nt)

	return Ef, Ef_av


#=========================================================

def budgetForcing2(U0,H0,u,v,h,F1,F2,F3,Ro,N,T,omega,Nt):
	''' Calculate time-dependent and time-mean energy budget due to forcing.
	Make sure to use full u, v and h fields. Contribution from 
	background terms will be removed after time averaging.'''

	F1t = timeDep(F1,T,omega,Nt)
	F2t = timeDep(F2,T,omega,Nt)
	F3t = timeDep(F3,T,omega,Nt)

	H = np.zeros(u.shape)
	U = np.zeros(u.shape)
	for ti in range(0,Nt):
		for i in range(0,N):
			H[:,i,ti] = H0
			U[:,i,ti] = U0

	Ef = (H * u + h * U + h * u) * F1t + (H * v + h * v) * F2t + (0.5 * (2 * U * u + u**2 + v**2) + h / Ro) * F3t

	Ef_av = timeAverage(Ef,T,Nt)


	return Ef, Ef_av


#=========================================================

def budgetDissipation(u,v,h,Ro,Re,gamma,dx,dy,T,Nt):
	''' Calculate time-dependent and time-mean energy budget due to dissipation.
	Make sure to use full u, v and h fields. Contribution from 
	background terms will be removed after time averaging.'''

	uxx = np.zeros(u.shape)
	uyy = np.zeros(u.shape)
	vxx = np.zeros(u.shape)
	vyy = np.zeros(u.shape)

	for ti in range(0,Nt):

		uyy[:,:,ti] = diff(diff(u[:,:,ti],0,0,dy),0,0,dy)
		uxx[:,:,ti] = diff(diff(u[:,:,ti],1,1,dx),1,1,dx)

		vyy[:,:,ti] = diff(diff(v[:,:,ti],0,0,dy),0,0,dy)
		vxx[:,:,ti] = diff(diff(v[:,:,ti],1,1,dx),1,1,dx)

	
	Ed = h * (u * (uxx + uyy) + v * (vxx + vyy)) / Re - gamma * h * (u**2 + v**2) / Ro

	Ed_av = timeAverage(Ed,T,Nt)

	return Ed, Ed_av

#=========================================================

def budgetDissipation2(U0,H0,u,v,h,Ro,Re,gamma,dx,dy,T,Nt,N):
	''' Calculate time-dependent and time-mean energy budget due to dissipation.
	Make sure to use full u, v and h fields. Contribution from background terms
	will be removed after time averaging. This function makes sure background isn't
	subject to viscosity/drag.'''

	uxx = np.zeros(u.shape)
	uyy = np.zeros(u.shape)
	vxx = np.zeros(u.shape)
	vyy = np.zeros(u.shape)

	for ti in range(0,Nt):

		uyy[:,:,ti] = diff(diff(u[:,:,ti],0,0,dy),0,0,dy)
		uxx[:,:,ti] = diff(diff(u[:,:,ti],1,1,dx),1,1,dx)

		vyy[:,:,ti] = diff(diff(v[:,:,ti],0,0,dy),0,0,dy)
		vxx[:,:,ti] = diff(diff(v[:,:,ti],1,1,dx),1,1,dx)

	H = np.zeros(u.shape)
	U = np.zeros(u.shape)
	for ti in range(0,Nt):
		for i in range(0,N):
			H[:,i,ti] = H0
			U[:,i,ti] = U0

	Ed = (H * u + U * h + u * h) * ((uxx + uyy) / Re - gamma * u / Ro) + (H * v + v * h) * ((vxx + vyy) / Re - gamma * v / Ro)
	# This omits a couple of terms that average to zero.

	Ed_av = timeAverage(Ed,T,Nt)

	return Ed, Ed_av

#=========================================================

def budgetDissipation3(U0,H0,u,v,h,Ro,Re,gamma,dx,dy,T,Nt,N):
	''' Calculate time-dependent and time-mean energy budget due to dissipation.
	Make sure to use full u, v and h fields. Contribution from background terms
	will be removed after time averaging. This function makes sure background isn't
	subject to viscosity/drag.'''

# Contribution from dissipation terms, if we assume it only acts upon eddy flow, is h_full * u_full * u. 
# This separates this function from the previous two.

	uxx = np.zeros(u.shape)
	uyy = np.zeros(u.shape)
	vxx = np.zeros(u.shape)
	vyy = np.zeros(u.shape)

	for ti in range(0,Nt):

		uyy[:,:,ti] = diff(diff(u[:,:,ti],0,0,dy),0,0,dy)
		uxx[:,:,ti] = diff(diff(u[:,:,ti],1,1,dx),1,1,dx)

		vyy[:,:,ti] = diff(diff(v[:,:,ti],0,0,dy),0,0,dy)
		vxx[:,:,ti] = diff(diff(v[:,:,ti],1,1,dx),1,1,dx)

	H = np.zeros(u.shape)
	U = np.zeros(u.shape)
	for ti in range(0,Nt):
		for i in range(0,N):
			H[:,i,ti] = H0
			U[:,i,ti] = U0

	# Extra lines that look at contributions to dissipation of energy. 
	#Ed_drag = - (U + u) * (H + h) * gamma * u / Ro - v * (H + h) * gamma * v / Ro
	#Ed_diss = (U + u) * (H + h) * (uxx + uyy) / Re + v * (H + h) * (vxx + vyy) / Re

	#e1 = timeAverage(Ed_drag,T,Nt)	
	#e2 = timeAverage(Ed_diss,T,Nt)
	#plt.subplot(121)
	#plt.contourf(e1); plt.colorbar()
	#plt.subplot(122)
	#plt.contourf(e2); plt.colorbar()
	#plt.show()


	Ed = (U + u) * (H + h) * ((uxx + uyy) / Re - gamma * u / Ro) + v * (H + h) * ((vxx + vyy) / Re - gamma * v / Ro)
	# This omits a couple of terms that average to zero.

	Ed_av = timeAverage(Ed,T,Nt)

	return Ed, Ed_av



#========================================================= 

def budgetFlux(u,v,h,Ro,dx,dy,T,Nt):
	''' Calculate time-dependent and time-mean energy budget due to fluxes.
	Make sure to use full u, v and h fields. Contribution from 
	background terms will be removed after time averaging.'''
	
	E = 0.5 * h * (u**2 + v**2) + h**2 / Ro

	uEflux = np.zeros(u.shape)
	vEflux = np.zeros(u.shape)
	Eflux = np.zeros(u.shape)

	for ti in range(0,Nt):

		uEflux[:,:,ti] = u[:,:,ti]*E[:,:,ti]
		vEflux[:,:,ti] = v[:,:,ti]*E[:,:,ti]

		Eflux[:,:,ti] = - diff(uEflux[:,:,ti],1,1,dx) - diff(vEflux[:,:,ti],0,0,dy)
	
	Eflux_av = timeAverage(Eflux,T,Nt)
	uEflux_av = timeAverage(uEflux,T,Nt)
	vEflux_av = timeAverage(vEflux,T,Nt)

	return Eflux, Eflux_av, uEflux_av, vEflux_av

#=========================================================



