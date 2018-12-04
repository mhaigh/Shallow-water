#
# Move this file up in order to run.
#

import numpy as np
import matplotlib.pyplot as plt


from inputFile import *

#==================================================================

path = '/media/mike/Seagate Expansion Drive/Documents/GulfStream/RSW/DATA/1L/PAPER1/UNIFORM/'

u = np.load(path + 'u_U0='+U0_load+'.npy')[:,:,ts]
v = np.load(path + 'v_U0='+U0_load+'.npy')[:,:,ts]
h = np.load(path + 'eta_U0='+U0_load+'.npy')[:,:,ts]
P[:,:,si] = np.load(path + 'P_U0='+U0_load+'.npy')
P_xav[:,si] = np.trapz(P[:,:,si],x_nd,dx_nd,axis=1);

	
plt.pcolor(x_grid, y_grid, h, cmap=cmap,vmin=-lm,vmax=lm)
plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
plt.xlabel('x')
plt.ylabel('y')
plt.xticks([-0.5,0,0.5])
plt.yticks([-0.5,0,0.5])
plt.text(0.4,0.4,r'$\eta^{\prime}$',fontsize=fs+12);
plt.grid();
plt.colorbar()
