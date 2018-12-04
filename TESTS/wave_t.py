from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.animation as animation

#=============================================================

# diff_center
def diff_center(f):
	
	N = len(f);
	df = np.zeros(N);

	for i in range(1,N-1):
		df[i] = f[i+1] - f[i-1];
	df[0] = f[1] - f[N-1];
	df[N-1] = f[0] - f[N-2];	

	return df

#=============================================================

I = np.complex(0.0,1.0);

N = 256;

g = 9.81;
H = 4000.0;
# Wave speed
c = np.sqrt(g*H);
nu = 1.e6;

# Space parameters
L = 3840.0 * 1000.0
x = np.linspace(-L/2,L/2,N+1);
x0 = 1. * 90.0 * 1000.0
K = np.fft.fftfreq(N,L/N);
dx = x[1] - x[0];

# Time parameters
period_days = 1.0e-2;
period = 1.*(L/(2.*c));# * 24. * period_days;		# Periodicity of plunger (s)
omega = 1. / (period);          		# Frequency of plunger, once every 50 days (e-6) (s-1)
Nt = 200;								# Number of time samples
T = np.linspace(0,period,Nt+1);			# Array of time samples across one forcing period (s)
dt = T[1] - T[0];						# Size of the timestep (s)
ts = 10; 								# index at which the time-snapshot is taken
t = T[ts];

courant = c * dt / dx;
print('Courant = ' + str(courant));
	
time_dep = np.cos(2. * np.pi * omega * T[0:Nt]);
# Plunger
A = 1.0e-7
F_dcts = np.zeros(N);
F_cts = np.zeros(N);
for i in range(0,N):
	if abs(x[i]) < x0:
		F_dcts[i] = A * np.cos((np.pi / 2) * x[i] / x0);
		F_cts[i] = 0.5 * A * (1 + np.cos(np.pi * x[i] / x0));

F_wave = A * np.sin(2 * np.pi * 2.0 * x[0:N] / L);
F_delta = np.zeros(N);
F_delta[int(N/2)]=1.


F = F_cts;

#=============================================================

NN = 40*Nt;

u0 = np.zeros(N);
u1 = np.zeros(N);

u = np.zeros((N,NN));
u_new = np.zeros(N);
u_old = u1;
u_oldold = u0;
for ti in range(2,NN):	
	print(ti);
	tii = ti % Nt
	u_xx = diff_center(diff_center(u_old)) / (2 * dx)**2;
	u_xxxx = diff_center(diff_center(u_xx)) / (2 * dx)**2;
	for i in range(0,N):
		u_new[i] = 2. * u_old[i] - u_oldold[i] + 2 * dt * (c**2 * u_xx[i] - nu * u_xxxx[i] + F[i] * time_dep[tii]);	

	u[:,ti] = u_new[:];
	u_oldold = u_old.copy();	
	u_old = u_new.copy();
	

print(u[:,NN-5]);

	

fig, ax = plt.subplots();
line, = ax.plot(x[0:N], np.zeros(N));

def animate(i):
	line.set_ydata(u[:,i]);  # update the data
	return line,

# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x[0:N], mask=True))
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(0, NN), init_func=init,
                              interval=5, blit=True)
plt.show()


























