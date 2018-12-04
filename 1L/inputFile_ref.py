# inputFile.py
#=======================================================
#=======================================================
# File of input parameters for the 1L RSW plunger code

import numpy as np
from scipy.special import erf

from diagnostics import diff
import matplotlib.pyplot as plt

#=======================================================
#=======================================================

FORCE = 'BALANCED';       	# 'BALANCED' for geostrophically balanced forcing, 
							# 'VORTICITY' for forcing on the momentum eqautions only,
							# 'BUOYANCY' for forcing on continuity equation only 'USER'.

FORCE_TYPE = 'DELTA';		# 'DCTS' is the original forcing, in which F3 has a discontinous derivative,
							# so that F1 and F2 are discontinous.
							# 'CTS' redefines the 'DCTS' forcing so that all forcing terms are continuous,
							# while still retaining the essential properties of the forcing. 

Fpos = 'CENTER';			# 4 choices for positioning of plunger, 'NORTH', 'CENTER' and 'SOUTH'
							

BG = 'UNIFORM';			# Options: UNIFORM, QUADRATIC, GAUSSIAN, NONE.

GAUSS = 'REF';			# If GAUSSIAN is selected, here are options for some predefined parameters.
							# Choices are REF,WIDE,SHARP,SHARPER,STRONG,WEAK

BC = 'FREE-SLIP';			# Two boundary condition choices at north and south boundaries: NO-SLIP or FREE-SLIP 

# Domain
#=======================================================

N = 128+1; 			# Number of gridpoints
					# For NO-SLIP: 44, 172, 684
					# For FREE-SLIP: 86, 342
N2 = N-2;			# Number of 'live' gridpoints for u and v, depending on BCs.	

# Define dim, the number of degrees of freedom in y
if BC == 'NO-SLIP':
	dim = 3 * N - 4;
elif BC == 'FREE-SLIP':
	dim = 3 * N - 2;
else:
	print('Error: must use valid boundary condition.');				

Lx = 3840000.		# Zonal lengthscale (m)
Ly = 3840000.		# Meridional lengthscale (m)
Hflat = 4000.		# Motionless ocean depth (i.e. without BG flow SSH adjustment) (m)		
L = Lx;

y = np.linspace(-Ly/2,Ly/2,N);				# Array of all grid points in physical space.
x = np.linspace(-Lx/2,Lx/2+Lx/(N-1),N+1);	# Zonal grid points	set has an extra point, first and last points are duplicates of each other (periodicity).

# Remove dead points in the y-direction.
yd = np.zeros(N2);
for j in range(0,N2):
    yd[j] = y[j+1];

dy = y[1] - y[0];     # Distance between gridpoints (m)
dx = x[1] - x[0];

y_grid, x_grid = np.mgrid[slice(-Ly/2,Ly/2+dy,dy),slice(-Lx/2,Lx/2+dx,dx)];

K = np.fft.fftfreq(N,Lx/N); 		 # Array of x-gridpoints in wavenumber space

# Rotation
#=======================================================

f0 = 0.83e-4;      		# Base value of Coriolis parameter (s-1)
beta = 2e-11;     		# Planetary vorticity gradient (m-1 s-1)
f = f0 + beta * y;      # Coriolis frequency (s-1)

# Other physical parameters
#=======================================================

g = 9.81;		# Acceleration due to gravity (m s-2)
gamma = 4.0e-8;	# Frictional coefficient (s-1)
nu = 100.0;		# Kinematic viscosity (m2 s-1)

# Background flow
#=======================================================

# Define a y-dependent zonal BG flow U. From U, the code prescribes the BG geostrophic SSH
# so that the BG steady state satisfies the geostrophic equations.

U0 = np.zeros(N);
H0 = np.zeros(N);

# Uniform zonal BG flow
if BG == 'UNIFORM':
	Umag = 0.16;
	for j in range(0,N):
		U0[j] = Umag; 			# (m s-1)
		H0[j] = - (U0[j] / g) * (f0 * y[j] + beta * y[j]**2 / 2) + Hflat;
	
# Quadratic BG flow
elif BG == 'QUADRATIC':
	Umag = 0.2;
	A = Umag * 4 / Ly**2;		# Maximum BG flow velocity Umag
	for j in range(0,N):
		U0[j] = A * (Ly**2 / 4 - y[j]**2);		# A quadratic eastward (positve amplitude) 'jet' with no flow on the boundaries
		H0[j] = A / g * (beta * y[j]**4 / 4 + f0 * y[j]**3 / 3 - Ly**2 * beta * y[j]**2 / 8 - Ly**2 * f0 * y[j] / 4) + Hflat;

# Gaussian BG flow
elif BG == 'GAUSSIAN':
	if GAUSS == 'REF':
		Umag = 0.4;
		sigma = 0.03 * Ly;			# Increasing sigma decreases the sharpness of the jet
	elif GAUSS == 'WIDE':
		Umag = 0.2;
		sigma = 0.4 * Ly;
	elif GAUSS == 'SHARP':
		Umag = 0.2;
		sigma = 0.2 * Ly;
	elif GAUSS == 'SHARPER':
		Umag = 0.2;
		sigma = 0.15 * Ly;
	elif GAUSS == 'STRONG':
		Umag = 0.3;
		sigma = 0.3 * Ly;
	elif GAUSS == 'WEAK':
		Umag = 0.1;
		sigma = 0.3 * Ly;
	# The rest of the parameters do not depend on the type of Gaussian flow we want
	l = Ly / 2;
	a = Umag / (np.exp(l**2 / sigma**2) - 1);	# Maximum BG flow velocity Umag
	for j in range(0,N):
		U0[j] = a * np.exp((l**2 - y[j]**2) / sigma**2) - a;		# -a ensures U0 is zero on the boundaries
		H0[j] = - a * (np.sqrt(np.pi) * f0 * sigma * np.exp(l**2 / sigma**2) * erf(y[j] / sigma) / 2 
					- beta * sigma**2 * np.exp((l**2 - y[j]**2) / sigma**2) / 2
					- f0 * y[j] - beta * y[j]**2 / 2) / g + Hflat; #erf(0);
		
elif BG == 'NONE':
	for j in range(0,N):
		Umag = 0;
		H0[j] = Hflat;

H0_y = diff(H0,2,0,dy);

# Calculate BG PV
Q = (f + diff(U0,2,0,dy)) / H0;

# Forcing
#=======================================================

# Instead of defining the forcing amplitude in the forcing module, we define it here as other codes require this value for normalisation
r0 = 2*90.0 * 1000.0;  
AmpF = 1.0e-7; 
if Fpos == 'NORTH':
	y0_index = int(3*N/4);
elif Fpos == 'CENTER':
	y0_index = int(N/2);
elif Fpos == 'SOUTH':
	y0_index = int(N/4);
elif Fpos == 'USER':
	y0_index = 3;
y0 = y[y0_index];

# Be careful here to make sure that the plunger is not forcing boundary terms.

# Time parameters
#=======================================================

period_days = 60.;						# Periodicity of plunger (days)
period = 3600. * 24. * period_days;		# Periodicity of plunger (s)
omega = 1. / (period);          		# Frequency of plunger, once every 50 days (e-6) (s-1)
Nt = 200;								# Number of time samples
T = np.linspace(0,period,Nt+1);			# Array of time samples across one forcing period (s)
dt = T[1] - T[0];						# Size of the timestep (s)
ts = Nt-1; 								# index at which the time-snapshot is taken
t = T[ts];								# Time of the snapshot

#=======================================================

# Non-dimensionalisation
#=======================================================
#=======================================================
# Here we denote all non-dimensional parameters by '_nd'.

# To find the characteristic velocity U and depthscale H, we find the spatial average of the 
# geostrophic BG state U0 and H0.
#=======================================================

U = 1.0e0;
H = Hflat;

chi = f0 * U * Ly / g;
T_adv = Lx / U;			# The advective timescale

# Defining dimensionless parameters
#=======================================================

Lx_nd = Lx / Ly;		# In case Lx and Ly are chosen to be different, we still scale by the same length, Ly
Ly_nd = Ly / Ly;				

y_nd = y / Ly;    
x_nd = x / Ly;
yd_nd = yd / Ly;
y0_nd = y0 / Ly;
r0_nd = r0 / Ly;

y_grid = y_grid / Ly;
x_grid = x_grid / Ly;

dy_nd = y_nd[1] - y_nd[0];
dx_nd = x_nd[1] - x_nd[0];

K_nd = K * Ly ;		# The same as: K_nd = np.fft.fftfreq(N2,dx_nd)

H0_nd = H0 / chi;	# The steady-state SSH scales the same way as eta.
H0_y_nd = H0_y * Ly / chi;
U0_nd = U0 / U;

f0_nd = 1.0;					# =f0/f0      		 
bh = beta * Ly / f0;
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

Ro = U / (f0 * L); 			# Rossby number: measures inertial forces relative to rotational ones.
if nu != 0:
	Re = L * U / nu;		# Reynolds number: measures inertial forces relative to viscous ones.
else:
	Re = None;				# Instead of defining it as infinity, define it as None.
Ld = np.sqrt(g * r0) / f0;	# Rossby def radius.

# Output
#=======================================================

outputPath = '/home/mike/Documents/GulfStream/RSW/DATA/1L/';

errorPhys = False;     	# Print error of full solutions 
errorSpec = False;		# Print error of spectral solutions

doEnergy = False;				# Energy
doPV = False;					# Calculate potential vorticity
doFootprints = True;			# Calculate footprints, requires findPV = True.
doEEFs = False;					# Calculate equivalent eddy fluxes, require findFootprints = True.
footprintComponents = True;		# If true, calculates the footprint in terms of its components.

# Initialise all these variables as none; even if they are not calculated, they are still called by the ouput module.
PV_prime = None; PV_full = None; PV_BG = None; Pq = None; Pq_xav = None; EEFq = None;

# Plots
#=======================================================

plotForcing = False;
plotBG = False;
plotSol = True;
plotPV = False;
plotPV_av = False;
plotFootprint = False;
plotPhaseAmp = False;

#=======================================================

# Print essential parameters to the terminal
print('SCALES: U = ' + str(U) + ', H = ' + str(H) + ', H scale = ' + str(chi) + ', T = ' + str(T_adv));
print('FORCING PERIOD = ' + str(period_days) + ' DAYS');
print(str(FORCE) + ' FORCING WITH ' + str(BG) + ' BG FLOW')
print('Ro = ' + str(Ro));
print('Re = ' + str(Re));
print('Ld = ' + str(Ld));
print('N = ' + str(N));

#=======================================================
