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
	a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,c5,d1,d3,d4,d5,e4,e5,f1,f2,f3,f4 = solver.SOLVER_COEFFICIENTS(Ro,Re,K,f,U1,U2,H1,H2,rho1_nd,rho2_nd,omega,gamma,dy,N)

	# Solver
	if BC == 'NO-SLIP':
		solution = solver.NO_SLIP_SOLVER(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,c5,d1,d3,d4,d5,e4,e5,f1,f2,f3,f4,Ro*Ftilde1,Ro*Ftilde2,Ftilde3,Ro*Ftilde4,Ro*Ftilde5,Ftilde6,N,N2)
	if BC == 'FREE-SLIP':
		solution = solver.FREE_SLIP_SOLVER4(a1,a2,a3,a4,b1,b4,c1,c2,c3,c4,c5,d1,d3,d4,d5,e4,e5,f1,f2,f3,f4,Ro*Ftilde1,Ro*Ftilde2,Ftilde3,Ro*Ftilde4,Ro*Ftilde5,Ftilde6,N,N2)
	
	#===================================================

	utilde, vtilde, htilde = solver.extractSols(solution,N,N2,BC)
	u, v, h = solver.SPEC_TO_PHYS(utilde,vtilde,htilde,T,Nt,dx,omega,N)

	# Before taking real part, can define an error calculator to call here.

	u = np.real(u)
	v = np.real(v)
	h = np.real(h)

	#u = u / AmpF_nd
	#v = v / AmpF_nd
	#h = h / AmpF_nd
	
	# For use in PV and footprint calculations: the 'full' zonal velocities and interface thicknesses.
	u_full = np.zeros((N,N,Nt,2))
	h_full = np.zeros((N,N,Nt,2))
	for j in range(0,N):
		u_full[j,:,:,0] = u[j,:,:,0] + U1[j]
		u_full[j,:,:,1] = u[j,:,:,1] + U2[j]
		h_full[j,:,:,0] = h[j,:,:,0] + H1[j]
		h_full[j,:,:,1] = h[j,:,:,1] + H2[j]

	# Call function calculate PV in each layer.
	q = np.zeros((N,N,Nt,2)); q_full = np.zeros((N,N,Nt,2)); Q = np.zeros((N,2))
	q[:,:,:,0], q_full[:,:,:,0], Q[:,0] = PV.vort(u[:,:,:,0],v[:,:,:,0],h[:,:,:,0],u_full[:,:,:,0],h_full[:,:,:,0],H1,U1,N,Nt,dx,dy,f)
	q[:,:,:,1], q_full[:,:,:,1], Q[:,1] = PV.vort(u[:,:,:,1],v[:,:,:,1],h[:,:,:,1],u_full[:,:,:,1],h_full[:,:,:,1],H2,U2,N,Nt,dx,dy,f)

	# Calculate footprints using previously calculated PV. Most interseted in the upper layer.
	P, P_xav = PV.footprint(u_full[:,:,:,0],v[:,:,:,0],q_full[:,:,:,0],x,y,dx,dy,T,Nt)

	# PLOTS
	#====================================================

	plt.contourf(u[:,:,0,1]); plt.colorbar(); plt.show()
	quit()
	plotting.solutionPlots(x,y,x_grid,y_grid,u,v,h,ts,N,False)
	plotting.footprintPlots(x,y,P,P_xav)

#====================================================

if __name__ == '__main__':
	RSW_main();







