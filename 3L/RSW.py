# RSW_visc_2L
#=======================================================

# This code solves the two-layer shallow water equations (centered-in-space finite difference), with external forcing terms on each of the six equations.
# The equations are solved in a beta-plane zonally periodic channel, with no-normal flow BCs at the northern and southern boundaries.
# The model includes simple linear Rayleigh drag in the lower layer and viscosity in both layers.
# Also included is a latitude-dependent zonal BG flow in each layer and corresponding geostrophic BG sea-surface height, around which the equations are linearised.
# The original governing equations are simplified by implementing the zonal Fourier transform and assuming time-periodicity.

# This means solving a system of the form:
# a1 * u1 + a2 * u1_yy + a3 * v1 + a4 * eta0 = F1,
# b1 * u1 + b2 * v1 + b3 * v1_yy + b4 * eta0_y = F2,
# c1 * u1 + c2 * v1 + c3 * v1_y + c4 * eta0 = F3 
# d1 * u2 + d2 * u2_yy + d3 * v2 + d4 * eta0 + d5 * eta1 = F4;
# e1 * u2 + e2 * v2 + e3 * v2_yy + e4 * eta0_y + e5 * eta1_y = F5;
# f1 * u1 + f2 * v1 + f3 * v1_y + f4 * eta1 = F6; 
# where (u1,v1), (u2,v2) are the horizontal velocity vectors in each layer, eta0 is height of the uppermost interface (SSH) and eta1 is the height of the interface between the two layers.
# Fi are the forcings. The ai, bi, ci, di, ei, fi are k- and y-dependent coefficients.

#====================================================

import sys

import numpy as np
import matplotlib.pyplot as plt

from core import diagnostics, forcing, PV, solver
from output import plotting

from inputFile import *

# 2L SW Solver
#====================================================
#====================================================

def RSW_main():

	# Coefficients
	a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,c5,d1,d3,d4,d5,e4,e5,f1,f2,f3,f4 = solver.SOLVER_COEFFICIENTS(Ro,Re,K_nd,f_nd,U1_nd,U2_nd,H1_nd,H2_nd,rho1_nd,rho2_nd,omega_nd,gamma_nd,dy_nd,N);

	# Solver
	if BC == 'NO-SLIP':
		u1tilde_nd, u2tilde_nd, v1tilde_nd, v2tilde_nd, eta0tilde_nd, eta1tilde_nd = (solver.NO_SLIP_SOLVER(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,c5,d1,d3,d4,d5,e4,e5,f1,f2,f3,f4,Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,Ftilde4_nd,Ftilde5_nd,Ftilde6_nd,N,N2));
	if BC == 'FREE-SLIP':
		u1tilde_nd, u2tilde_nd, v1tilde_nd, v2tilde_nd, eta0tilde_nd, eta1tilde_nd = (solver.FREE_SLIP_SOLVER(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,c5,d1,d3,d4,d5,e4,e5,f1,f2,f3,f4,Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,Ftilde4_nd,Ftilde5_nd,Ftilde6_nd,N,N2));
	
	#===================================================
	u1_nd, u2_nd, v1_nd, v2_nd, eta0_nd, eta1_nd = solver.SPEC_TO_PHYS(u1tilde_nd,u2tilde_nd,v1tilde_nd,v2tilde_nd,eta0tilde_nd,eta1tilde_nd,T_nd,Nt,dx_nd,omega_nd,N);

	# Before taking real part, can define an error calculator to call here.

	u1_nd = np.real(u1_nd);
	u2_nd = np.real(u2_nd);
	v1_nd = np.real(v1_nd);
	v2_nd = np.real(v2_nd);
	eta0_nd = np.real(eta0_nd);
	eta1_nd = np.real(eta1_nd);

	# The interface thicknesses defined via the interface heights.
	h1_nd = eta0_nd - eta1_nd;
	h2_nd = eta1_nd;

	# For use in PV and footprint calculations: the 'full' zonal velocities and interface thicknesses.
	u1_full = np.zeros((N,N,Nt));
	u2_full = np.zeros((N,N,Nt));
	h1_full = np.zeros((N,N,Nt));
	h2_full = np.zeros((N,N,Nt));
	for j in range(0,N):
		u1_full[:,j,:] = u1_nd[:,j,:] + U1_nd[j];
		u2_full[:,j,:] = u2_nd[:,j,:] + U2_nd[j];
		h1_full[:,j,:] = h1_nd[:,j,:] + H1_nd[j];
		h2_full[:,j,:] = h2_nd[:,j,:] + H2_nd[j];

	# Call function calculate PV in each layer.
	#PV1_prime, PV1_full, PV1_BG = PV.vort(u1_nd,v1_nd,h1_nd,u1_full,h1_full,H1_nd,U1_nd,N,Nt,dx_nd,dy_nd,f_nd);
	#PV2_prime, PV2_full, PV2_BG = PV.vort(u2_nd,v2_nd,h2_nd,u2_full,h2_full,H2_nd,U2_nd,N,Nt,dx_nd,dy_nd,f_nd);

	# Calculate footprints using previously calculated PV. Most interseted in the upper layer.
	#P, P_xav = PV.footprint(u1_full,v1_nd,PV1_full,U1_nd,U1,x_nd,y_nd,dx_nd,dy_nd,AmpF_nd,FORCE1,r0,nu,BG1,Fpos,ts,period_days,N,Nt,GAUSS);

	# PLOTS
	#====================================================

	plotting.solutionPlots(x_nd,y_nd,x_grid,y_grid,u1_nd,u2_nd,v1_nd,v2_nd,h1_nd,h2_nd,ts,N,True);

		

#====================================================

if __name__ == '__main__':
	RSW_main();







