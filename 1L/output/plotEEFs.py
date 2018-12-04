# plotEEFs
#========================================

import numpy as np
import matplotlib.pyplot as plt

from output_read import npyReadEEF

#========================================

# This module creates three plots of the EEF.
# In each panel r0, Re and T are varied

BG = 'U0=-08'

nn = 497
EEF = np.zeros((nn,7))

# Control
EEF[:,0] = npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_y0.npy')
# r0 
EEF[:,1] = npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_y0_r60.npy')
EEF[:,2] = npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_y0_r120.npy')
# Re
EEF[:,0] = npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_y0_k50.npy')
EEF[:,0] = npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_y0_k200.npy')
# T
EEF[:,0] = npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_y0_om50.npy')
EEF[:,0] = npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_y0_om70.npy')


plt.plot(EEF)
plt.show()


