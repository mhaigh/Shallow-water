# ENERGY.py

#==================================================================================

# Calculate energy statistics from saved numpy file, output of RSW.py

#==================================================================================

import sys

import numpy as np
import matplotlib.pyplot as plt

from core import energy, diagnostics

from inputFile import *

#==================================================================================

# BG flow
bg = '-08'

# Read data.
path = '/home/mike/Documents/GulfStream/RSW/DATA/1L/ENERGY/' + bg + '/'
#path = ''

#ufile = 'u16.npy'; vfile = 'v16.npy'; hfile = 'h16.npy'
ufile = 'u.npy'; vfile = 'v.npy'; hfile = 'h.npy'

u = np.load(path + ufile)
v = np.load(path + vfile)
h = np.load(path + hfile)

# Flow has been normalised by forcing amplitude, can rescale it here. 
scale = 1. / AmpF_nd
u /= scale; v /= scale; h /= scale

# Define background state. Done.

# Calculate full flow quantities.
u_full = diagnostics.fullFlow(u,U0_nd)
h_full = diagnostics.fullFlow(h,H0_nd)

#==================================================================================


# Background energy
KE_BG, KE_BG_tot, PE_BG, PE_BG_tot = energy.energy_BG(U0_nd,H0_nd,Ro,y_nd,dy_nd,N)

# Total energy
KE, KE_tot = energy.KE(u_full,v,h_full,x_nd,y_nd,dx_nd,dy_nd,N)
PE, PE_tot = energy.PE(h_full,Ro,x_nd,y_nd,dx_nd,dy_nd,N)

KE_eddy = KE_tot - KE_BG_tot


Ed, Ed_av = energy.budgetDissipation3(U0_nd,H0_nd,u,v,h,Ro,Re,gamma_nd,dx_nd,dy_nd,T_nd,Nt,N)
Ef, Ef_av = energy.budgetForcing(u_full,v,h_full,F1_nd,F2_nd,F3_nd,Ro,N,T_nd,omega_nd,Nt)

Ed_tot = np.sum(Ed_av)
Ef_tot = np.sum(Ef_av)

print(Ed_tot)
print(Ef_tot)
print(Ed_tot / Ef_tot)

print('Max dissipation = ' + str(np.max(Ed_av)))
print('Max/min diss. = ' + str(np.max(Ed_av) / np.min(Ed_av)))

plt.figure(figsize=(22,7))

plt.subplot(131)
plt.contourf(Ef_av); plt.grid(); plt.colorbar()

plt.subplot(132)
plt.contourf(Ed_av); plt.grid(); plt.colorbar(); 

plt.subplot(133)
plt.contourf(Ef_av + Ed_av); plt.grid(); plt.colorbar();

plt.show()










