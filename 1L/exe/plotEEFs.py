
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np


from inputFile import *
from output import output_read


vs = 'y0';

BG = 'U0=Gaussian'
#BG = 'U0=Gaussian_wide'
#BG = 'U0=16'
#BG = 'U0=0'
#BG = 'U0=-08'
#BG = 'vsU0'

opt = 'w'

#=====================================================================================================================

# Read EEF arrays

if vs == 'y0':

	path = '/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/med_res/'+BG+'/PV/'

	EEF_r = np.load(path+'EEF_PV_r.npy')
	EEF_k = np.load(path+'EEF_PV_k.npy')
	EEF_om = np.load(path+'EEF_PV_om.npy')

	lr0 = 'r0 = 60 km';
	lr1 = 'r0 = 90 km';
	lr2 = 'r0 = 120 km';

	lk0 = r'Re = 2Re$_{0}$';
	lk1 = r'Re = Re$_{0}$';
	lk2 = r'Re = Re$_{0}$/2';


	lw0 = 'T = 50 days';
	lw1 = 'T = 60 days';
	lw2 = 'T = 70 days';

	NN = np.shape(EEF_r0)[1]
	N_skip = (N - NN) / 2
	y_forced = y_nd[N_skip:N-N_skip]

elif vs == 'y02':
	# Benchmark
	EEF_1 = output_read.npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/med_res/'+BG+'/PV/EEF_PV_y0.npy')
	
	# r
	#EEF_r0, uq_0, Uq_0, uQ_0, vq_0, vQ_0 = output_read.npyReadEEF_y0_components('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_y0_r60.npy')
	#EEF_r2, uq_2, Uq_2, uQ_2, vq_2, vQ_2 = output_read.npyReadEEF_y0_components('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_y0_r120.npy')
	#lr0 = 'r0 = 60 km';
	#lr1 = 'r0 = 90 km';
	#lr2 = 'r0 = 120 km';
	# Note that varying r0 means that the lengths of the EEF array will differ
	
	#NN = len(EEF_r0)
	#N_skip = (N - NN) / 2
	#y_forced_0 = y_nd[N_skip:N-N_skip]	

	#NN = len(EEF_r2)
	#N_skip = (N - NN) / 2
	#y_forced_2 = y_nd[N_skip:N-N_skip]
		
	# k
	#EEF_k0 = output_read.npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_y0_k50.npy');
	#EEF_k2, uq_2, Uq_2, uQ_2, vq_2, vQ_2 = output_read.npyReadEEF_y0_components('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_y0_k200.npy');
	#lk0 = r'Re = 2Re$_{0}$';
	#lk1 = r'Re = Re$_{0}$';
	#lk2 = r'Re = Re$_{0}/2$';

	# w
	EEF_w0 = output_read.npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/med_res/'+BG+'/PV/EEF_PV_y0_om50.npy');
	EEF_w2 = output_read.npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/med_res/'+BG+'/PV/EEF_PV_y0_om70.npy');
	lw0 = 'T = 50 days';
	lw1 = 'T = 60 days';
	lw2 = 'T = 70 days';

	NN = len(EEF_1);

elif vs == 'U0':
	
	EEF_r0 = output_read.npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_U0_r60.npy');
	EEF_r1 = output_read.npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_U0.npy');
	EEF_r2 = output_read.npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_U0_r120.npy');
	lr0 = 'r0 = 60 km';
	lr1 = 'r0 = 90 km';
	lr2 = 'r0 = 120 km';

	EEF_k0 = output_read.npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_U0_k50.npy');
	EEF_k1 = output_read.npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_U0.npy');
	EEF_k2 = output_read.npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_U0_k200.npy');
	lk0 = r'Re = 2Re$_{0}$';
	lk1 = r'Re = Re$_{0}$';
	lk2 = r'Re = Re$_{0}$/2';

	EEF_w0 = output_read.npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_U0_om50.npy');
	EEF_w1 = output_read.npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_U0.npy');
	EEF_w2 = output_read.npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_U0_om70.npy');
	lw0 = 'T = 50 days';
	lw1 = 'T = 60 days';
	lw2 = 'T = 70 days';

	EEF_y0 = output_read.npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_U0_south.npy');
	EEF_y1 = output_read.npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_U0.npy');
	EEF_y2 = output_read.npyReadEEF('/home/mike/Documents/GulfStream/RSW/DATA/1L/EEFs/high_res/'+BG+'/PV/EEF_PV_U0_north.npy');
	ly0 = 'SOUTH';
	ly1 = 'CENTER';
	ly2 = 'NORTH';

	NN = len(EEF_y0);


#====================================================================================================================



if False:
	for j in range(1,100):
		if j%2 == 0:	
			EEF_50[j] = 0.5 * (EEF_50[j+1] + EEF_50[j-1]);
			EEF_60[j] = 0.5 * (EEF_60[j+1] + EEF_60[j-1]);
			EEF_70[j] = 0.5 * (EEF_70[j+1] + EEF_70[j-1]);
			EEF_50[NN-j] = 0.5 * (EEF_50[NN-j+1] + EEF_50[NN-j-1]);
			EEF_60[NN-j] = 0.5 * (EEF_60[NN-j+1] + EEF_60[NN-j-1]);
			EEF_70[NN-j] = 0.5 * (EEF_70[NN-j+1] + EEF_70[NN-j-1]);

#=====================================================================================================================
          
w50 = T_adv / (3600. * 24. * 50.);
w60 = T_adv / (3600. * 24. * 60.);
w70 = T_adv / (3600. * 24. * 70.);

#=====================================================================================================================

if vs == 'y0':
	N_skip = (N - NN) / 2; # Should always be even
	y_forced = y_nd[N_skip:N-N_skip];
elif vs == 'U0':
	U_range = np.linspace(-0.3,0.5,NN);

#Rd2=1./f[N_skip:N-N_skip];
#plt.plot(y_forced,EEF_0/Rd2);
#plt.plot(y_forced,EEF_0/Rd2);
#plt.show();


# Rescale all EEFs to correct value
#==================================
if vs == 'U0':
	# w
	EEF_w0 = 10.**4 * EEF_w0 / w50;
	EEF_w1 = 10.**4 * EEF_w1 / w60;
	EEF_w2 = 10.**4 * EEF_w2 / w70;
	# r
	EEF_r0 = 10.**4 * EEF_r0 / w60;
	EEF_r1 = 10.**4 * EEF_r1 / w60;
	EEF_r2 = 10.**4 * EEF_r2 / w60;
	# k
	EEF_k0 = 10.**4 * EEF_k0 / w60;
	EEF_k1 = 10.**4 * EEF_k1 / w60;
	EEF_k2 = 10.**4 * EEF_k2 / w60;
	# y
	EEF_y0 = 10.**4 * EEF_y0 / w60;
	EEF_y1 = 10.**4 * EEF_y1 / w60;
	EEF_y2 = 10.**4 * EEF_y2 / w60;
#===================================

#==================================

#===================================


if vs == 'y0':

	plt.plot(y_forced,10**8*EEF_w0,label=lw0,linewidth=1.3);
	plt.plot(y_forced,10**8*EEF_1,label=lw1,linewidth=1.3);
	plt.plot(y_forced,10**8*EEF_w2,label=lw2,linewidth=1.3);
	#if BG == 'U0=Gaussian':
	#	plt.yticks((-6,-4,-2,0,2,4,6,8,10,12),fontsize=0);
	#elif BG == 'U0=0':
	#	plt.yticks((0,2,4,6,8,10,12),fontsize=0);
	#elif BG == 'U0=16':
	#	plt.yticks((0,2,4,6,8,10,12),fontsize=0);
	plt.legend(prop={'size': 18},loc=3)
	plt.xlim(-0.5,0.5);
	plt.xticks((-0.5,-0.25,0.0,0.25,0.5));
	plt.xlabel('Forcing latitude, y0',fontsize=18);
	plt.grid()
	plt.tight_layout(pad=0.3, w_pad=0.2, h_pad=1.0);
	plt.show()



elif vs == 'y02':

	lw = 1.3

	plt.plot(y_forced,10**8*EEF_w0,label=lw0,linewidth=1.3);
	plt.plot(y_forced,10**8*EEF_1,label=lw1,linewidth=1.3);
	plt.plot(y_forced,10**8*EEF_w2,label=lw2,linewidth=1.3);
	#if BG == 'U0=Gaussian':
	#	plt.yticks((-6,-4,-2,0,2,4,6,8,10,12),fontsize=0);
	#elif BG == 'U0=0':
	#	plt.yticks((0,2,4,6,8,10,12),fontsize=0);
	#elif BG == 'U0=16':
	#	plt.yticks((0,2,4,6,8,10,12),fontsize=0);
	plt.legend(prop={'size': 18},loc=3)
	plt.xlim(-0.5,0.5);
	plt.xticks((-0.5,-0.25,0.0,0.25,0.5));
	plt.xlabel('Forcing latitude, y0',fontsize=18);
	plt.grid()
	plt.tight_layout(pad=0.3, w_pad=0.2, h_pad=1.0);
	plt.show()

# y0







