
import sys

import numpy as np
import matplotlib.pyplot as plt


#================================================================

# These plots show that the integral of P is the dominant contributor to the EEF.
# Although the lenghtscale l has well defined extrema, it doesn't vary far from a nonzero average. 
# Division of the EEF by the lengthscale l yields a line qualitatively and quantitatively similar to the EEF.

#================================================================

EEF01 = np.load('EEF_PV.npy');
l = np.load('EEF_l.npy');
P_xav = np.transpose(np.load('P_xav.npy'));

P1 = np.transpose(np.load('P1.npy'));
P2 = np.transpose(np.load('P2.npy'));

# After transposing, first index in P_xav corresponds to y, and second to U0

EEF = EEF01[:,0] - EEF01[:,1];

#================================================================


# Define the background flow array manually.
nn = len(l);
Umin = -0.1;
Umax = 0.1;

U0 = np.linspace(Umin,Umax,nn);

# Define the meridional coordinate
N = 257;
y = np.linspace(-0.5,0.5,N);

#================================================================

# At what index of y, does the maximum of P_xav occur. Is it always the same? 
i_max = np.zeros(nn);
i_min = np.zeros(nn);
for ni in range(0,nn):
	a = np.argsort(P_xav[:,ni])
	i_max[ni], i_min[ni] = a[N-1], a[0];
# With i_max, i_min, we can plot the maxima of P_xav against U0.
maxi = int(i_max[0]);
mini = int(i_min[0]);

if False:
	
	fig, ax1 = plt.subplots()
	ax1.plot(U0,P_xav[maxi,:]/max(P_xav[maxi,:]), 'b-')
	ax1.set_xlabel('U0')
	ax1.set_ylabel('P_xav', color='b')
	ax1.tick_params('y', colors='b')
	
	ax2 = ax1.twinx()
	ax2.plot(U0,EEF/max(EEF), 'r-')
	ax2.set_ylabel('EEF/l', color='r')
	ax2.tick_params('y', colors='r')
	
	fig.tight_layout();
	plt.show();

ratio = P1[maxi,:] / P2[maxi,:];
plt.subplot(121);
plt.plot(U0,P1[maxi,:],label='P1');
plt.plot(U0,P2[maxi,:],label='P2');
plt.subplot(122);
plt.plot(U0,ratio);
plt.show();

#================================================================

# EEF contributions from the P1 and P2.

#================================================================

if False:
	fig, ax1 = plt.subplots()
	ax1.plot(U0,l, 'b-')
	ax1.set_xlabel('U0')
	# Make the y-axis label, ticks and tick labels match the line color.
	ax1.set_ylabel('l', color='b')
	ax1.tick_params('y', colors='b')
	
	ax2 = ax1.twinx()
	ax2.plot(U0,EEF, 'r-')
	ax2.set_ylabel('EEF', color='r')
	ax2.tick_params('y', colors='r')
	
	fig.tight_layout();
	plt.show();
	
if True:
	plt.subplot(131);
	plt.contourf(U0,y,P1);
	plt.plot(-0.023*np.ones(N),y,'k--');
	plt.plot(0.068*np.ones(N),y,'k--');
	plt.colorbar();
	plt.xlabel('U0');
	plt.ylabel('y');

	plt.subplot(132);
	plt.contourf(U0,y,P2);
	plt.plot(-0.023*np.ones(N),y,'k--');
	plt.plot(0.068*np.ones(N),y,'k--');
	plt.colorbar();
	plt.xlabel('U0');
	plt.ylabel('y');

	plt.subplot(133);
	plt.contourf(U0,y,P_xav);
	plt.plot(-0.023*np.ones(N),y,'k--');
	plt.plot(0.068*np.ones(N),y,'k--');
	plt.colorbar();
	plt.xlabel('U0');
	plt.ylabel('y');
	plt.show();


if True:
	fig, ax1 = plt.subplots();
	ax1.plot(U0,P1[mini,:],label='P1');
	ax1.plot(U0,P2[maxi,:],label='P2');
	ax1.plot(U0,P_xav[maxi,:],label='P_xav');
	plt.legend();
	
	ax2 = ax1.twinx();
	ax2.plot(U0,EEF,label='EEF');
	plt.legend();
	plt.show();





