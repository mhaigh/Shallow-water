# forcing_test_RSW.py 
#====================================================

import numpy as np
import matplotlib.pyplot as plt

from diagnostics import diff

from inputFile_1L import *

#====================================================



I = np.complex(0,1);

# New forcing
F1_new = np.zeros((N,N+1));
F2_new = np.zeros((N,N+1));
F3_new = np.zeros((N,N+1));
count = 0;
mass = 0;
for i in range(0,N+1):
	for j in range(0,N):
		r = np.sqrt(x[i]**2 + (y[j]-y0)**2);
		if r < r0:
			count = count + 1;
			if r == 0:
				F1_new[j,i] = 0;
				F2_new[j,i] = 0;						
			else:	
				F1_new[j,i] = AmpF * np.pi * g * (y[j]-y0) / (2 * r0 * f[j] * r) * np.sin((np.pi / 2) * r / r0);
				F2_new[j,i] = - AmpF * np.pi * g * x[i] / (2 * r0 * f[j] * r) * np.sin((np.pi / 2) * r / r0);
			F3_new[j,i] = 0.5 * AmpF * (1 + np.cos((np.pi) * r / r0));
			mass = mass + F3_new[j,i];
mass = mass / (N*(N+1) - count);
for i in range(0,N+1):
	for j in range(0,N):
		F3_new[j,i] = F3_new[j,i] - mass;


# Original forcing
F1_orig = np.zeros((N,N+1));
F2_orig = np.zeros((N,N+1));
F3_orig = np.zeros((N,N+1));
count = 0;
mass = 0;
for i in range(0,N+1):
	for j in range(0,N):
		r = np.sqrt(x[i]**2 + (y[j]-y0)**2);
		if r < r0:
			count = count + 1;
			if r == 0:
				F1_orig[j,i] = 0;
				F2_orig[j,i] = 0;						
			else:	
				F1_orig[j,i] = AmpF * np.pi * g * (y[j]-y0) / (2 * r0 * f[j] * r) * np.sin((np.pi / 2) * r / r0);
				F2_orig[j,i] = - AmpF * np.pi * g * x[i] / (2 * r0 * f[j] * r) * np.sin((np.pi / 2) * r / r0);
			F3_orig[j,i] = AmpF * np.cos((np.pi / 2) * r / r0);
			mass = mass + F3_orig[j,i];
mass = mass / (N*(N+1) - count);
for i in range(0,N+1):
	for j in range(0,N):
		F3_orig[j,i] = F3_orig[j,i] - mass;


i_sample = N/2;

F3_orig_y = diff(F3_orig[:,i_sample],2,0,dy_nd);
F3_orig_yy = diff(F3_orig_y,2,0,dy_nd);

F3_new_y = diff(F3_new[:,i_sample],2,0,dy_nd);
F3_new_yy = diff(F3_new_y,2,0,dy_nd);

plt.subplot(321);
#plt.contourf(F3_orig);
plt.plot(F3_orig[:,i_sample]);
plt.subplot(322);
#plt.contourf(F3_new);
plt.plot(F3_new[:,i_sample]);
plt.subplot(323);
plt.plot(F3_orig_y);
plt.subplot(324);
plt.plot(F3_new_y);
plt.subplot(325);
plt.plot(F3_orig_yy);
plt.subplot(326);
plt.plot(F3_new_yy);
plt.show();



 
