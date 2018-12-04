# plot2dEEFs.py

#========================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from core import BG_state

from inputFile import *

#========================================================


path = '/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/med_res/G/'
#path = ''
file_umag = 'EEF_umag.npy'; file_sigma = 'EEF_sigma.npy'

EEF = np.load(path + file_umag)
EEFu = 1e7 * (EEF[:,:,0] - EEF[:,:,1])

EEF = np.load(path + file_sigma)
EEFs = 1e7 * (EEF[:,:,0] - EEF[:,:,1])

Ns = 161

sigma_set = np.linspace(0.015,0.045,Ns) * 3840000.0
ds = sigma_set[1] - sigma_set[0]

umag_set = np.linspace(0.4,1.2,Ns) 
du = umag_set[1] - umag_set[0]

y0_min = y[0] + L/3					# We want to keep the forcing at least one gridpoint away from the boundary
y0_max = y[N-1] - L/3
y0_set = [];							# Initialise an empty set of forcing latitudes
y0_index_set = [];
for j in range(0,N):
	if y0_min <= y[j] <= y0_max:
		y0_set.append(y[j]);			# Build the set of forcing locations, all at least 1 gridpoint away from the boundary.	
		y0_index_set.append(j);
y0_set = np.array(y0_set) / L;
y0_index_set = np.array(y0_index_set);
nn = np.shape(y0_set)[0]

sigma_mesh, y0_mesh = np.mgrid[slice(sigma_set.min()/1000.0,(sigma_set.max())/1000.,ds/1000.),slice(y0_set.min(),y0_set.max()+dy_nd,dy_nd)];
umag_mesh, y0_mesh = np.mgrid[slice(umag_set.min(),(umag_set.max()+du),du),slice(y0_set.min(),y0_set.max()+dy_nd,dy_nd)];
#slice(sigma_set.min()/1000.0,(sigma_set.max()+ds)/1000.,ds/1000.)

#========================================================


Umag_ref = 0.8
sigma_ref = 0.02 * 3840000.

Qyu = np.zeros((Ns,nn))
Qys = np.zeros((Ns,nn))
for ui in range(0,Ns):
	
	# Umag
	sigma = sigma_set[ui]
	U0, H0 = BG_state.BG_Gaussian(Umag_ref,sigma,JET_POS,Hflat,f0,beta,g,y,L,N)
	U0 = U0 / U; H0 = H0 / chi
	U0 = U0[y0_index_set]; H0 = H0[y0_index_set]
	U0y =  diff(U0,2,0,dy_nd)
	Q = (f_nd[y0_index_set] / Ro - U0y) / H0

	Qys[ui,:] = diff(Q,2,0,dy_nd)

	# Sigma
	Umag = umag_set[ui]
	U0, H0 = BG_state.BG_Gaussian(Umag,sigma_ref,JET_POS,Hflat,f0,beta,g,y,L,N)
	U0 = U0 / U; H0 = H0 / chi
	U0 = U0[y0_index_set]; H0 = H0[y0_index_set]
	U0y =  diff(U0,2,0,dy_nd)
	Q = (f_nd[y0_index_set] / Ro - U0y) / H0

	Qyu[ui,:] = diff(Q,2,0,dy_nd)

	

#========================================================


fs = 16
cm = 'bwr'
Elim = 2.#.8*np.max(np.abs(EEF))


grsp = gs.GridSpec(1,2,width_ratios=[1,1.25])


fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(14,5.9))

plt.subplot(grsp[0,0])	
CS = plt.contour(y0_set,umag_set,Qyu,1,colors='k')
plt.clabel(CS, fontsize=9, inline=1)
plt.pcolor(y0_mesh,umag_mesh,EEFu,cmap=cm,vmin=-Elim,vmax=Elim)

plt.grid()
plt.xlabel('y0',fontsize=fs)
plt.ylabel('Max jet speed (m/s)',fontsize=fs)
plt.xticks(fontsize=fs-2)
plt.yticks(fontsize=fs-2)

plt.subplot(grsp[0,1])	
CS = plt.contour(y0_set,sigma_set/1000,Qys,1,colors='k')
plt.clabel(CS, fontsize=9, inline=1)
plt.pcolor(y0_mesh,sigma_mesh,EEFs,cmap=cm,vmin=-Elim,vmax=Elim)
plt.colorbar()

plt.grid()
plt.xlabel('y0',fontsize=fs)
plt.ylabel('Jet width (km)',fontsize=fs)
plt.xticks(fontsize=fs-2)
plt.yticks(fontsize=fs-2)

#plt.tight_layout(pad=0.5, w_pad=0.4, h_pad=1.0);
plt.show()





















