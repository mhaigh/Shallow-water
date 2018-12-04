#
# Move this file up in order to run.
#

import numpy as np
import matplotlib.pyplot as plt


from inputFile import *

#==================================================================

path = '/media/mike/Seagate Expansion Drive/Documents/GulfStream/RSW/DATA/1L/PAPER1/UNIFORM/'

U0_load = '08'
#u = np.load(path + 'u_U0='+U0_load+'.npy')[:,:,ts]
#v = np.load(path + 'v_U0='+U0_load+'.npy')[:,:,ts]
h = np.load(path + 'eta_U0='+U0_load+'.npy')[:,:,ts-40]

h=np.real(h)

lm = 0.5 

h = h / np.max(h)

plt.pcolor(x_grid, y_grid, h, cmap='jet',vmin=-lm,vmax=lm)
plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
plt.xlabel('x',fontsize=18)
plt.ylabel('y',fontsize=18)
plt.xticks([-0.5,-0.25,0,0.25,0.5],fontsize=14)
plt.yticks([-0.5,-0.25,0,0.25,0.5],fontsize=14)
plt.text(0.4,0.4,r'$\eta^{\prime}$',fontsize=26);
plt.grid();
plt.colorbar()
plt.tight_layout()
plt.show()
