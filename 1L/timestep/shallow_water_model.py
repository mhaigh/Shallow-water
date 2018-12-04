# SHALLOW WATER MODEL
# ------------------------------------------------------------------


import numpy as np

# Possible initial conditions of the height field
UNIFORM_WESTERLY=1;
ZONAL_JET=2;
REANALYSIS=3;
GAUSSIAN_BLOB=4;
STEP=5;
CYCLONE_IN_WESTERLY=6;
SHARP_SHEAR=7;
EQUATORIAL_EASTERLY=8;
SINUSOIDAL=9;

# Possible orographies
FLAT=0;
SLOPE=1;
GAUSSIAN_MOUNTAIN=2;
EARTH_OROGRAPHY=3;
SEA_MOUNT=4;

# ------------------------------------------------------------------
# SECTION 1: Configuration
g    = 9.81;                # Acceleration due to gravity (m/s2)
f    = 1.0e-4;              # Coriolis parameter (s-1)
beta = 1.6e-11;             # Meridional gradient of f (s-1m-1)

dt_mins              = 1;   # Timestep (minutes)
output_interval_mins = 60;  # Time between outputs (minutes)
forecast_length_days = 4;   # Total simulation length (days)

orography = FLAT;
initial_conditions = GAUSSIAN_BLOB;
initially_geostrophic = False		   # Can be "true" or "false"
add_random_height_noise = False		 # Can be "true" or "false"

# If you change the number of gridpoints then orography=EARTH_OROGRAPHY
# or initial_conditions=REANALYSIS won't work
nx=254; # Number of zonal gridpoints
ny=50;  # Number of meridional gridpoints

dx=100.0e3; # Zonal grid spacing (m)
dy=dx;      # Meridional grid spacing

# Specify the range of heights to plot in metres
plot_height_range = [9500,10500];

# ------------------------------------------------------------------
# SECTION 2: Act on the configuration information
dt = dt_mins*60.0; # Timestep (s)
output_interval = output_interval_mins*60.0; # Time between outputs (s)
forecast_length = forecast_length_days*24.0*3600.0; # Forecast length (s)
nt = int(forecast_length // dt+1) # Number of timesteps
timesteps_between_outputs = output_interval//dt
noutput = int(nt//timesteps_between_outputs + 1) # Number of output frames

x = np.linspace(0,nx-1,nx)		# Zonal distance coordinate (m)
y = np.linspace(0,ny-1,ny)		# Meridional distance coordinate (m)

# Create the orography field "H"
if orography == FLAT:
    H = np.zeros((nx, ny));

# Create the initial height field 
mean_wind_speed = 20; # m/s
height = 10000-(mean_wind_speed*f/g) * (y-np.mean(y)); 

# Coriolis parameter as a matrix of values varying in y only
F = f+beta * (y-np.mean(y))
print(np.shape(F))
# Initialize the wind to rest
u = np.zeros((nx, ny));
v = np.zeros((nx, ny));

# We may need to add small-amplitude random noise in order to initialize 
# instability
if add_random_height_noise:
  height = height + 1.0 * np.randn(np.shape(height))*(dx/1.0e5)*(np.abs(F)/1e-4);

# Define h as the depth of the fluid (whereas "height" is the height of
# the upper surface)
h = height - H;

# Initialize the 3D arrays where the output data will be stored
u_save = np.zeros((nx, ny, noutput))
v_save = np.zeros((nx, ny, noutput))
h_save = np.zeros((nx, ny, noutput))
t_save = np.zeros((1, noutput));

# Index to stored data
i_save = 1;


# ------------------------------------------------------------------
# SECTION 3: Main loop
for n in range(0,nt):
	# Every fixed number of timesteps we store the fields
	if np.mod(n-1,timesteps_between_outputs) == 0:
		max_u = np.sqrt(np.max(u**2+v**2))
		u_save[:,:,i_save] = u
		v_save[:,:,i_save] = v
		h_save[:,:,i_save] = h
		t_save[i_save] = (n-1)*s*dt
		i_save = i_save+1


	# Compute the accelerations
	u_accel = F[1:ny-2]*v[1:nx-2,1:ny-2] - (g/(2*dx))*(H[2:nx-1,2:ny-1]-H[1:nx-2,2:ny-1]);
	v_accel = -F[2:ny-1]*u[2:nx-1,2:ny-1] - (g/(2*dy))*(H[2:nx-1,3:ny]-H[2:nx-1,1:ny-2]);

	# Call the Lax-Wendroff scheme to move forward one timestep
	unew, vnew, h_new = lax_wendroff(dx, dy, dt, g, u, v, h, u_accel, v_accel);

	# Update the wind and height fields, taking care to enforce boundary conditions 
	u = unew[[nx-1] + range(0,nx) +[0]],[[0] + range(0,ny) + [ny-1]]
	v = vnew[[nx-1] + range(0,nx) + [0]],[[0] + range(0,ny) + [ny-1]]
	v[:,[0, ny-1]] = 0
	h[:,1:ny-2] = h_new[[ny-1] + range(0,ny) + [0],:]

