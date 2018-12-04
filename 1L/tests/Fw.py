
# fw.py

#=================================================================================

import sys

import numpy as np
import matplotlib.pyplot as plt

from diagnostics import diff

from inputFile import *


#=================================================================================

# Already have forcing from input File.

Ks = np.fft.fftshift(K_nd)
F = F1_nd
Ftilde = np.fft.fftshift(np.fft.fft2(F))

om = 1. / (30. * 3600. * 24.)
beta = 2.0e-11


skip = 00
U = np.zeros((N,N))
W = 0
for j in range(1,N-skip):
	for i in range(1,N-skip):
		U[j,i] = om / K[i] - beta / (K[i]**2 + K[j]**2)
		W += U[j,i] * np.abs(Ftilde[j,i])

W = W / np.sum(np.abs(Ftilde[1:N,1:N]))
print(W)


plt.contourf(np.fft.fftshift(U))
plt.colorbar()
plt.show()

sys.exit()
plt.contourf(F)
plt.show()
plt.contourf(Ks,Ks,np.abs(Ftilde))
plt.colorbar()
plt.grid()
plt.show()

