import numpy as np
import matplotlib.pyplot as plt

N = 129;
L = N;

K = np.linspace(-L/2,L/2,(N));
L = np.linspace(-L/2,L/2,(N))

U = 0.08;

beta = 2.0e-11;

w = np.zeros((N,N));
px = np.zeros((N,N));
py = np.zeros((N,N))

for i in range(0,N):
	for j in range(0,N):
		if K[i] == 0 or L[j] == 0:
			w[j,i] = 0
		else:
			w[j,i] = U * K[i] - beta * K[i] / (K[i]**2 + L[j]**2);
			px[j,i] = w[j,i] / K[i]
			py[j,i] = w[j,i] / L[j]

lim = 10.0e-10
lim = np.max(w)
v = np.linspace(-lim,lim,11)
plt.contourf(K,L,w,v,cmap=plt.cm.jet);
plt.colorbar(ticks=v)
plt.show();

plt.contourf(px)
plt.show()

plt.plot(K,px[64,:]);
plt.show();

