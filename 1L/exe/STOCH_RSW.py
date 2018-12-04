# STOCH_RSW.py

#=======================================================

# This code solves the single-layer shallow water equations (centered-in-space finite difference), with external forcing terms on each of the three equations.
# The equations are solved in a beta-plane zonally periodic channel, with no-normal flow BCs at the northern and southern boundaries.
# The model includes simple linear Rayleigh drag on the ocean's bottom and viscosity.
# Also included is a latitude-dependent zonal BG flow and corresponding geostrophic BG sea-surface height, around which the equations are linearised.
# The original governing equations are simplified by implementing the zonal Fourier transform and assuming time-periodicity.

# This means solving a system of the form:
# a1 * u + a2 * u_yy + a3 * v + a4 * eta = Ro * F1,
# b1 * u + b2 * v + b3 * v_yy + b4 * eta_y = Ro * F2,
# c1 * u + c2 * v + c3 * v_y + c4 * eta = F3,
# where (u,v) is the horizontal velocity vector, eta is the interface height, Fi are the forcings with correpsonding amplitude alphai
# and ai, bi, ci are k- and y-dependent coefficients.

#====================================================

import sys

import numpy as np

from core import diagnostics, PV, forcing, solver
import output

from inputFile import *

# RSW SOLVER STOCHASTIC 
#====================================================
#====================================================

u = np.zeros((N,N,Nt),dtype=complex);
v = np.zeros((N,N,Nt),dtype=complex);
h = np.zeros((N,N,Nt),dtype=complex);

S = np.load('time_series1.npy');

#Om = np.fft.fftfreq(Nt,dt_nd);
S_tilde = np.fft.fft(S);

for wi in range(1,Nt):
	print(wi);
	# Coefficients
	a1,a2,a3,a4,b4,c1,c2,c3,c4 = solver.SOLVER_COEFFICIENTS(Ro,Re,K_nd,f_nd,U0_nd,H0_nd,omega_nd,gamma_nd,dy_nd,N)
	# Solver
	if BC == 'NO-SLIP':
		solution = solver.NO_SLIP_SOLVER(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,S_tilde[wi]*Ro*Ftilde1_nd,S_tilde[wi]*Ro*Ftilde2_nd,S_tilde[wi]*Ftilde3_nd,N,N2);
	if BC == 'FREE-SLIP':
		solution = solver.FREE_SLIP_SOLVER(a1,a2,a3,a4,f_nd,b4,c1,c2,c3,c4,S_tilde[wi]*Ro*Ftilde1_nd,S_tilde[wi]*Ro*Ftilde2_nd,S_tilde[wi]*Ftilde3_nd,N,N2)

	u[:,:,wi], v[:,:,wi], h[:,:,wi] = solver.extractSols(solution,N,N2,BC);

u, v, h = solver.SPEC_TO_PHYS_STOCH(u,v,h,dx_nd,N);

# Normalise all solutions by the (non-dimensional) forcing amplitude. 
u = u / AmpF_nd;
v = v / AmpF_nd;
h = h / AmpF_nd;

mass = sum(sum(h[:,:,ts]))/N**2
print(mass)

plt.contourf(h[:,:,ts]);
plt.colorbar()
plt.show()

#np.save('u.npy',u);
#np.save('v.npy',v);
#np.save('h.npy',h);

sys.exit();


# Soltuion Plots
if plotSol:
	plotting.solutionPlots(x_nd,y_nd,u,v,h,ts,FORCE,BG,Fpos,N,x_grid,y_grid,False);
	plotting.solutionPlots_save(x_nd,y_nd,u,v,h,ts,FORCE,BG,Fpos,N,x_grid,y_grid,True);
	#plotting.solutionPlotsDim(x,y,u,v,eta,ts,L,FORCE,BG,Fpos,N);


