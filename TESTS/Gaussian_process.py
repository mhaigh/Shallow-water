# Gaussian_process.py
#=================================================================

# A function that simulates a stochastic process S = (S[0], S[1], S[2]...)
# which S[i] are Gaussian distributed random variables and the process has
# exponentially distributed correllation, i.e. <S[i],S[i+n]>=exp(-n/tau),
# where tau>0 is some time-scale.

#=================================================================

import numpy as np
import matplotlib.pyplot as plt

#=================================================================

N = 10;
period_days = 60.
tau = 50.0

T = np.linspace(0,period_days,N+1);
dt = T[1] - T[0];

f = np.exp(-1./tau);
ff = np.sqrt(1-f**2);



S = np.zeros(N);

S[0] = 0 #np.random.normal(0,1.0)

for ti in range(1,N):
	g = np.random.normal(0,1.0);
	S[ti] = f * S[ti-1] + ff * g;


# Define a Plank taper window.
#================================

w = np.zeros(N)
eps = 0.1

for n in range(1,N-1):

	a = 2.*n / (N-1.) - 1.
	
	Zp = 2. * eps * (1. / (1. + a) + 1. / (1. - 2. * eps + a))
	Zm = 2. * eps * (1. / (1. - a) + 1. / (1. - 2. * eps - a))
	if (0 <= n < eps * (N-1)):
		w[n] = 1. / (np.exp(Zp) + 1.)
	elif (eps * (N-1) < n < (1-eps) * (N-1)):
		w[n] = 1.
	elif ((1-eps) * (N-1) < n <= N-1):
		w[n] = 1. / (np.exp(Zm) + 1.)

	
S = S * w

#========================================

np.save('time_series',S);

S_tilde = np.fft.fft(S);

fs = 24

plt.figure(1,figsize=[21,6]);
plt.plot(np.linspace(0,N,N),S,linewidth=1.3,color='k');
plt.xlim(0,N)
plt.xlabel('Time units',fontsize=fs)
plt.ylabel(r'$S$',fontsize=fs+4)
plt.tight_layout()
plt.grid()
plt.show();



