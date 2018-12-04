#PLOT.py
#=======================================================

# This module plots sols and footprint from saved data.

#=======================================================

import sys

import numpy as np
import diagnostics
import output
import plotting
import PV

from inputFile import *

#=======================================================

# UNIFORM


U0_name = 'U0=08';
U0_str = r'$U_{0}=0.08$';
#U0_name = 'y0=15sigma';
#U0_str = r'$y_{0}=1.5\sigma$';

#path = '/home/mike/Documents/GulfStream/RSW/DATA/1L/Paper1/';
path = '/media/mike/Seagate Expansion Drive/Documents/GulfStream/RSW/DATA/1L/PAPER1/UNIFORM/';
#path = '/media/mike/Seagate Expansion Drive/Documents/GulfStream/RSW/DATA/1L/PAPER1/GAUSSIAN/' + U0_name + '/';


u = np.load(path+'u_'+U0_name+'.npy');
v = np.load(path+'v_'+U0_name+'.npy');
h = np.load(path+'eta_'+U0_name+'.npy');
#P = np.load(path+'P_'+U0_name+'.npy');

#u = np.load(path+'u_complex_' + U0_name + '.npy');
#v = np.load(path+'v_complex_' + U0_name + '.npy');
#h = np.load(path+'h_complex_' + U0_name + '.npy');

#plotting.solutionPlotsPhase(x_grid,y_grid,u,v,h,ts,FORCE,BG,Fpos,U0_name,U0_str,N);
#plotting.solutionPlotsAmp(x_grid,y_grid,u,v,h,ts,FORCE,BG,Fpos,U0_name,U0_str,N);

#sys.exit();

#=======================================================

# GAUSSIAN

#U0_name = 'y0=0';
#U0_str = r'$y_{0}=-\sigma$';
#U0_str = r'$y_{0}=0$';

#path = '/home/mike/Documents/GulfStream/RSW/DATA/1L/PAPER1/GAUSSIAN/'
#u = np.load(path + '/' + U0_name + '/u_' + U0_name + '.npy');
#v = np.load(path + '/' + U0_name + '/v_' + U0_name + '.npy');
#h = np.load(path + '/' + U0_name + '/eta_' + U0_name + '.npy');
#P = np.load(path + '/' + U0_name + '/P_' + U0_name + '.npy');
#u = np.load('/home/mike/Documents/GulfStream/RSW/PYTHON/1L/u.npy');
#v = np.load('/home/mike/Documents/GulfStream/RSW/PYTHON/1L/v.npy');
#h = np.load('/home/mike/Documents/GulfStream/RSW/PYTHON/1L/h.npy');
#P = np.load('/home/mike/Documents/GulfStream/RSW/PYTHON/1L/P.npy');

#=======================================================


#ss = 0;
#if ss == 1:
#	u = u/0.01;
#	v = v/0.01;
#	h = h/0.01;
#	#P = P * 1.0e-8;
#	np.save(path+'u_U0='+U0_name+'.npy',u);
#	np.save(path+'v_U0='+U0_name+'.npy',v);
#	np.save(path+'eta_U0='+U0_name+'.npy',h);
#	#np.save(path+'P_U0='+U0_name+'.npy',P);

if False:
	h_full = np.zeros((N,N,Nt));
	u_full = np.zeros((N,N,Nt));
	for j in range(0,N):
		h_full[j,:,:] = h[j,:,:] + H0_nd[j];
		u_full[j,:,:] = u[j,:,:] + U0_nd[j];
	PV_prime, PV_full, PV_BG = PV.potentialVorticity(u,v,h,u_full,h_full,H0_nd,U0_nd,N,Nt,dx_nd,dy_nd,f_nd,Ro);
#P_xav = np.trapz(P,x_nd,dx_nd,axis=1);

#=======================================================

#plotting.pvPlots_save(PV_full,PV_prime,x_nd,y_nd,ts,FORCE,BG,Fpos,N,U0_str,x_grid,y_grid,U0_name);
plotting.solutionPlots_save(x_nd,y_nd,u,v,h,ts,FORCE,BG,Fpos,N,U0_str,x_grid,y_grid,True);
#plotting.footprintPlots_save(P,P_xav,x_nd,y_nd,ts,FORCE,BG,Fpos,N,U0_str,x_grid,y_grid,U0_name);


#EEF_array = PV.EEF(P_xav,y_nd,y0_nd,y0_index,dy_nd,N);
#EEF_north = EEF_array[0]; EEF_south = EEF_array[1];
#print(EEF_north, EEF_south);
#print(EEF_north-EEF_south);




#=======================================================


