	
import numpy as np
import matplotlib.pyplot as plt

#======================================

N = 513;
L = 3840000;

sigma = 0.02 * L;
Umag = 0.8;

y = np.linspace(-L/2,L/2,N);

U0 = np.zeros(N);
l = L / 2;
a = Umag / (np.exp(l**2 / (2. * sigma**2)) - 1.);	# Maximum BG flow velocity Umag
for j in range(0,N):
	gauss = np.exp((l**2 - y[j]**2) / (2. * sigma**2));
	lap = (y[j]**2 / (2 * sigma**2) - 1);
	U0[j] = (a * lap * gauss - a);		# -a ensures U0 is zero on the boundaries


print(max(U0));

plt.plot(U0);
plt.show();
