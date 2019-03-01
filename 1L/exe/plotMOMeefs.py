import numpy as np 
import matplotlib.pyplot as plt

#====================================================================================================

path = "/home/mike/Documents/GulfStream/RSW/DATA/1L/Paper2/efficiency/"

# Read data 
m50 = np.load(path + "Mu_array50.npy")

Nu, N = np.shape(m50)

m = np.zeros((3,np.shape(m50)[0],np.shape(m50)[1]))
mnorm = np.zeros((3,np.shape(m50)[0],np.shape(m50)[1]))
E = np.zeros((3,np.shape(m50)[0]))

m[0,] = m50 
m[1,] = np.load(path + "Mu_array60.npy")
m[2,] = np.load(path + "Mu_array70.npy")
# shape(m50) = ((321,257)) = ((Nu,N)),
# shape(m) = ((3,321,257)).

mnorm[0,] = np.load(path + "Munorm_array50.npy") 
mnorm[1,] = np.load(path + "Munorm_array60.npy")
mnorm[2,] = np.load(path + "Munorm_array70.npy")

E[0,] = np.load(path + "E_array50.npy")
E[1,] = np.load(path + "E_array60.npy")
E[2,] = np.load(path + "E_array70.npy")

# Define U0 set
U0 = np.linspace(-0.3,0.5,Nu)

#====================================================================================================

# What can we do with this data?

# Simple scalar measure of redistribution could me max(m50). This always lies at the latitude of the forcing (for uniform BG flow)

m_max = np.max(m,axis=2)
mnorm_max = np.max(mnorm,axis=2)


fs = 12

plt.figure(figsize=(21,6))

plt.subplot(131)
plt.plot(U0,m_max[0,],label="50 days",linewidth=1.2); plt.plot(U0,m_max[1,],label="60 days",linewidth=1.2); plt.plot(U0,m_max[2,],label="70 days",linewidth=1.2)
plt.xlim((-0.3,0.5)); 
plt.xlabel('U0',fontsize=fs+4)
plt.xticks(fontsize=fs)
plt.text(-.25,.9*np.max(m_max),r'$\mathcal{M}_{0}$',fontsize=fs+8)
plt.yticks(fontsize=fs)
plt.grid(); plt.legend()

plt.subplot(132)
plt.plot(U0,mnorm_max[0,],label="50 days",linewidth=1.2); plt.plot(U0,mnorm_max[1,],label="60 days",linewidth=1.2); plt.plot(U0,mnorm_max[2,],label="70 days",linewidth=1.2)
plt.xlim((-0.3,0.5)); 
plt.xlabel('U0',fontsize=fs+4)
plt.xticks(fontsize=fs)
plt.text(-.25,1.*np.max(mnorm_max),r'$\hat{\mathcal{M}}_{0}$',fontsize=fs+8)
#plt.text(-.25,1.*np.max(mnorm_max),r'$max\left(\langle \hat{\mathcal{M}}\rangle\right)$',fontsize=fs+8)
plt.yticks(fontsize=fs)
plt.grid(); plt.legend()

plt.subplot(133)
plt.plot(U0,E[0,],label="50 days",linewidth=1.2); plt.plot(U0,E[1,],label="60 days",linewidth=1.2); plt.plot(U0,E[2,],label="70 days",linewidth=1.2)
plt.xlim((-0.3,0.5)); 
plt.xlabel('U0',fontsize=fs+4)
plt.xticks(fontsize=fs)
plt.text(-.25,.95*np.max(E),r'$\mathcal{K}$',fontsize=fs+8)
plt.yticks(fontsize=fs)
plt.grid(); plt.legend()

plt.tight_layout()
plt.show()

