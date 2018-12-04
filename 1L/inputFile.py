# inputFile.py
#=======================================================
#=======================================================
# File of input parameters for the 1L RSW plunger code
# Most parameters are given in dimensional values.
# The initalisation stage redefines them nondimensionally.

import sys

import numpy as np
from scipy.special import erf

from core.diagnostics import diff
from core import forcing, BG_state

import matplotlib.pyplot as plt

#=======================================================
#=======================================================

# Domain
#=======================================================

BC = 'FREE-SLIP'	# Two boundary condition choices at north and south boundaries: NO-SLIP or FREE-SLIP 

N = 257		# Number of gridpoints

Lx = 3840000.		# Zonal lengthscale (m)
Ly = 3840000.		# Meridional lengthscale (m)
Hflat = 4000.		# Motionless ocean depth (i.e. without BG flow SSH adjustment) (m)		

f0 = 0.83e-4      	# Base value of Coriolis parameter (s-1)
beta = 2.0e-11     	# Planetary vorticity gradient (m-1 s-1)

# Other physical parameters
#=======================================================

g = 9.81			# Acceleration due to gravity (m s-2)
gamma = 4.0e-8		# Linear friction (s-1)
nu = 100.			# Eddy viscosity (m2 s-1)

# Background flow
#=======================================================

# Keep the unused options commented out.

BG = 'UNIFORM'			# Options: UNIFORM, SHEAR, QUADRATIC, GAUSSIAN, LAPGAUSS, ZERO.

# Uniform options
Umag = 0.0688 #0.0688, -0.0233, 0.0213

# Gaussian jet options
#Umag = 0.8				# Jet max speed
#sigma = 0.02 * 3840000.0	# Jet width
#JET_POS = 'CENTER'

# Shear options
#Umag = 100.0;
#shear = 1.0;	# Shear
# Forcing
#=======================================================

FORCE = 'BALANCED'       	# 'BALANCED' for geostrophically balanced forcing, 
							# 'VORTICITY' for forcing on the momentum eqautions only,
							# 'BUOYANCY' for forcing on continuity equation only 'USER'.

FORCE_TYPE = 'CTS2'			# 'DCTS' is the original forcing, in which F3 has a discontinous derivative,
							# so that F1 and F2 are discontinous.

Fpos = 'CENTER'				# 4 choices for positioning of plunger, 'NORTH', 'CENTER', 'SOUTH' and 'USER' (define this manually below)
							
r0 = 90.0 * 1000.0	 		# Forcing radius
AmpF = 1.0e-7			# Forcing amplitude

# Be careful here to make sure that the plunger is not forcing boundary terms.

# Time parameters
#=======================================================

period_days = 60.						# Periodicity of plunger (days)
Nt = 200								# Number of time samples
ts = Nt-1 								# index at which the time-snapshot is taken

# Output
#=======================================================

outputPath = '/home/mike/Documents/GulfStream/RSW/DATA/1L/'

errorPhys = False     	# Print error of full solutions 
errorSpec = False		# Print error of spectral solutions

doEnergy = False				# Energy
doPV = True					# Calculate potential vorticity
doFootprints = True			# Calculate footprints, requires findPV = True.
doEEFs = True					# Calculate equivalent eddy fluxes, require findFootprints = True.
footprintComponents = False		# If true, calculates the footprint in terms of its components.
doMomentum = False
doThickness = False
doCorr = True

# Initialise all these variables as none; even if they are not calculated, they are still called by the ouput module.
PV_prime = None; PV_full = None; PV_BG = None; Pq = None; Pq_xav = None; EEFq = None

# Plots
#======================================================

plotForcing = False
plotBG = False
plotSol = True
plotPV = True
plotPV_av = False
plotFootprint = True
plotPhaseAmp = True

#======================================================================================================================================================================================================
#======================================================================================================================================================================================================

# Only edit from here to alter USER defined routines.
# Remainder initialises RSW.py

#=======================================================

# Initialisation

#=======================================================
#=======================================================

# Domain
#=======================================================

N2 = N-2			# Number of 'live' gridpoints for u and v, depending on BCs.	

# Define dim, the number of degrees of freedom in y
if BC == 'NO-SLIP':
	dim = 3 * N - 4
elif BC == 'FREE-SLIP':
	dim = 3 * N - 2
else:
	print('Error: must use valid boundary condition.')

y = np.linspace(-Ly/2,Ly/2,N)				# Array of all grid points in physical space.
x = np.linspace(-Lx/2,Lx/2+Lx/(N-1),N+1)	# Zonal grid points	set has an extra point, first and last points are duplicates of each other (periodicity).

# Remove dead points in the y-direction.
yd = np.zeros(N2)
for j in range(0,N2):
    yd[j] = y[j+1]

dy = y[1] - y[0]     # Distance between gridpoints (m)
dx = x[1] - x[0]

y_grid, x_grid = np.mgrid[slice(-Ly/2,Ly/2+dy,dy),slice(-Lx/2,Lx/2+2.*dx,dx)]

K = np.fft.fftfreq(N,Lx/N) 		 # Array of x-gridpoints in wavenumber space

f = f0 + beta * y      # Coriolis frequency (s-1)

# Background flow
#=======================================================

if BG == 'UNIFORM':
	U0, H0 = BG_state.BG_uniform(Umag,Hflat,f0,beta,g,y,N)
elif BG == 'SHEAR':
	U0, H0 = BG_state.BG_shear(Umag,shear,Hflat,f0,beta,g,y,Ly,N)
elif BG == 'GAUSSIAN':
	U0, H0 = BG_state.BG_Gaussian(Umag,sigma,JET_POS,Hflat,f0,beta,g,y,Ly,N)
elif BG == 'LAPGAUSS':
	U0, H0 = BG_state.BG_LapGauss(Umag,sigma,JET_POS,Hflat,f0,beta,g,y,Ly,N)
elif BG == 'QUAD':
	U0, H0 = BG_state.BG_quadratic(Umag,Hflat,f0,beta,g,y,Ly,N)
elif BG == 'ZERO':
	U0, H0 = BG_state.BG_zero(Hflat,N)
else:
	raise ValueError('Invalid BG flow option selected');

# Forcing
#=======================================================

if Fpos == 'NORTH':
	y0_index = int(3*N/4)
elif Fpos == 'CENTER':
	y0_index = int(N/2)
elif Fpos == 'SOUTH':
	y0_index = int(N/4)
elif Fpos == 'USER':
	y0_index = int(N/2) + int(1.5*N*sigma/Ly)# - int(N/4); # - sigma * 25./16.
y0 = y[y0_index]
print(y0/Ly);
# Note that the forcing itself is defined in terms of dimensionless parameters, so is defined at the end of initialisation. Need to do the same for the background flow.

# Time parameters
#=======================================================

period = 3600. * 24. * period_days;		# Periodicity of plunger (s)
omega = 1. / (period);          		# Frequency of plunger, once every 50 days (e-6) (s-1)
T = np.linspace(0,period,Nt+1);			# Array of time samples across one forcing period (s)
dt = T[1] - T[0];						# Size of the timestep (s)
t = T[ts];								# Time of the snapshot

#======================================================================================================================================================================================================
#======================================================================================================================================================================================================

# Non-dimensionalisation
#=======================================================
#=======================================================
# Here we denote all non-dimensional parameters by '_nd'.

#=======================================================

L = Lx
U = 1.0e-2
H = Hflat

chi = f0 * U * L / g;
T_adv = L / U;			# The advective timescale

# Defining dimensionless parameters
#=======================================================

Lx = Lx / L;		# In case Lx and Ly are chosen to be different, we still scale by the same length, Ly
Ly = Ly / L;				

y_nd = y / L;    
x_nd = x / L;
yd_nd = yd / L;
y0_nd = y0 / L;
r0_nd = r0 / L;

y_grid = y_grid / L;
x_grid = x_grid / L;

dy_nd = y_nd[1] - y_nd[0];
dx_nd = x_nd[1] - x_nd[0];

K_nd = K * L ;		# The same as: K_nd = np.fft.fftfreq(N2,dx_nd)

H0_nd = H0 / chi;	# The steady-state SSH scales the same way as eta.
U0_nd = U0 / U;

f0_nd = 1.0;					# =f0/f0      		 
bh = beta * L / f0;
f_nd = f / f0;					# The same as: f_nd = f0_nd + bh * y_nd      

gamma_nd = gamma / f0;			# Simply scaled by the base Coriolis frequency

omega_nd = omega * T_adv;      	# All time variables scale advectively, i.e. T_adv~L/U
t_nd = t / T_adv;
T_nd = T / T_adv;
dt_nd = dt / T_adv;

AmpF_nd = AmpF * g / (f0 * U**2);

# Note that gravity g and kinematic viscosity aren't scaled, but rather used to define some extra dimensionless parameters

# Important dimensionless numbers
#=======================================================

Ro = U / (f0 * L) 				# Rossby number: measures inertial forces relative to rotational ones.
if nu != 0:
	Re = L * U / nu				# Reynolds number: measures inertial forces relative to viscous ones.
else:
	Re = None					# Instead of defining it as infinity, define it as None.
Ld = np.sqrt(g * H0) / f	# Rossby def radius.

# Forcing
#=======================================================

if FORCE_TYPE == 'CTS':
	F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing.forcing_cts(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE,AmpF_nd,f_nd,f0_nd,dx_nd,dy_nd)
elif FORCE_TYPE == 'CTS2':
	F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing.forcing_cts2(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE,AmpF_nd,f_nd,f0_nd,bh,dx_nd,dy_nd)
elif FORCE_TYPE == 'DCTS':
	F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing.forcing_dcts(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE,AmpF_nd,f_nd,f0_nd,dx_nd,dy_nd)
elif FORCE_TYPE == 'ELLIPSE':
	F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing.forcing_ellipse(x_nd,y_nd,K_nd,y0_nd,r0_nd,N,FORCE,AmpF_nd,f_nd,f0_nd,bh,dx_nd,dy_nd,9.)
elif FORCE_TYPE == 'DELTA':
	F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing.forcing_delta(AmpF_nd,y0_index,dx_nd,N)
else:
	sys.exit('ERROR: Invalid forcing option selected.')

#=======================================================

# Print essential parameters to the terminal
print('SCALES: U = ' + str(U) + ', H = ' + str(H) + ', H scale = ' + str(chi) + ', T = ' + str(T_adv))
print('FORCING PERIOD = ' + str(period_days) + ' DAYS')
print(str(FORCE) + ' FORCING WITH ' + str(BG) + ' BG FLOW')
print('Ro = ' + str(Ro))
print('Re = ' + str(Re))
print('Ld = ' + str(Ld[0]))
print('N = ' + str(N))

#plt.plot(H0_nd); 
#plt.show()
#print(sigma/L)
#print(r0_nd)

#U0y =  diff(U0_nd,2,0,dy_nd)
#Q = (f_nd/ Ro - U0y) / H0_nd
#Qy = diff(Q,2,0,dy_nd)
#Qyy = diff(Qy,2,0,dy_nd)
#plt.plot(Qy); plt.show()
#from output.plotting import bgPV
#bgPV(U0,Qy,y_nd)
