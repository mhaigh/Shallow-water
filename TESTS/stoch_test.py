# stoch_test.py
#=======================================================
#=======================================================

# Test file for producing a stochastic process and presenting it in frequency space

#=======================================================

import numpy as np
import matplotlib.pyplot as plt

#=======================================================


S = np.load('time_series.npy');
wn = len(S) - 2;
S = S[1:wn];
print(wn);
T = np.linspace(0,wn,wn+1);
Om = np.fft.fftfreq(wn,1);
print(Om);
print(T);
S_tilde = np.fft.fft(S);

plt.subplot(121);
plt.plot(S);
plt.subplot(122);
plt.plot(S_tilde);
plt.show();
