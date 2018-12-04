# Other
#========================================

import numpy as np
import matplotlib.pyplot as plt
import forcing_1L

import diagnostics

from inputFile_1L import *

#========================================
option = -2;

if option == -2:

	e = [[0.00124837567643, 0.000366720099694,0.00010394024873,3.29592544853e-05,1.19889451831e-05], [3.88268775732e-05,2.09050012386e-05,1.0968198529e-05,5.61471142092e-06,2.90614746129e-06], [4.21546218187,1.4634855819,0.460154365787,0.15656485614,0.0598864150904]];

	eq = 0;	

	N_set = [64,128,256,512,1024];
	Nlen = len(N_set);
	scale = np.zeros(Nlen);
	d = np.zeros(Nlen);
	for i in range(0,Nlen):
		scale[i] = 1.0/float(N_set[i])**2;	
		d[i] = e[eq][i] - scale[i];
	
	plt.plot(N_set,scale,'k--',linewidth=2.0);
	plt.plot(N_set,e[eq],color='r',linewidth=2.0);
	plt.xscale('log',basex=2);
	plt.yscale('log',basex=2);
	plt.show();

	plt.plot(d);
	plt.show();


if option == -1:

	ts = 10;	
	
	u_nd = np.load('u_nd.npy');
	v_nd = np.load('v_nd.npy');
	eta_nd = np.load('eta_nd.npy');

	if FORCE_TYPE == 'CTS':
		F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing_1L.forcing_cts(x,y,K,y0,r0,N,FORCE,AmpF,g,f,f0,U,L,dx,dy);
	elif FORCE_TYPE == 'DCTS':
		F1_nd, F2_nd, F3_nd, Ftilde1_nd, Ftilde2_nd, Ftilde3_nd = forcing_1L.forcing_dcts(x,y,K,y0,r0,N,FORCE,AmpF,g,f,f0,U,L,dx,dy);
	else:
		sys.exit('ERROR: Invalid forcing option selected.');

	e1, e2, e3 = diagnostics.error(u_nd,v_nd,eta_nd,dx_nd,dy_nd,dt_nd,U0_nd,H0_nd,Ro,gamma_nd,Re,f_nd,F1_nd,F2_nd,F3_nd,T_nd,ts,omega_nd,N);
	e = np.sqrt((e1**2 + e2**2 + e3**2) / 3.0);
	print 'Error = ' + str(e) + '. Error split = ' + str(e1) + ', ' + str(e2) + ', ' + str(e3);

	


if option == 0:
	cos = np.cos(2*np.pi*x_nd/(x_nd[N]-x_nd[0]));
	cos_y = diagnostics.diff(cos,2,1,dy_nd) / (2*np.pi);
	cos_yy = diagnostics.diff(cos_y,2,1,dy_nd)  / (2*np.pi);

	dim = len(cos);
	print(cos[0],cos[dim-1]);
	plt.subplot(131);
	plt.plot(cos);
	plt.grid();
	plt.subplot(132);
	plt.plot(cos_y);
	plt.grid();
	plt.subplot(133);
	plt.plot(cos_yy);
	plt.grid();
	plt.show();

# Plots a selection of Gaussian profiles
if option == 1:
	if BG == 'UNIFORM':
		a = 2;

	if BG == 'GAUSSIAN':

		U01 = np.zeros(N);
		U02 = np.zeros(N);
		U03 = np.zeros(N);
		U04 = np.zeros(N);
		U05 = np.zeros(N);
		U06 = np.zeros(N);

		H01 = np.zeros(N);
		H02 = np.zeros(N);
		H03 = np.zeros(N);
		H04 = np.zeros(N);
		H05 = np.zeros(N);
		H06 = np.zeros(N);

		# U01 - reference Gaussian jet BG flow
		Umag = 0.2;
		sigma = 0.3 * Ly;			# Increasing sigma decreases the sharpness of the jet
		l = Ly / 2;
		a = Umag / (np.exp(l**2 / sigma**2) - 1);	# Maximum BG flow velocity Umag
		for j in range(0,N):
			U01[j] = a * np.exp((l**2 - y[j]**2) / sigma**2) - a;		# -a ensures U0 is zero on the boundaries
			H01[j] = - a * (np.sqrt(np.pi) * f0 * sigma * np.exp(l**2 / sigma**2) * erf(y[j] / sigma) / 2 
					- beta * sigma**2 * np.exp((l**2 - y[j]**2) / sigma**2) / 2
					- f0 * y[j] - beta * y[j]**2 / 2) / g + Hflat; #erf(0);
	
		# U02 - 1.5 * strenth BG flow
		Umag = 0.3;
		sigma = 0.3 * Ly;			
		l = Ly / 2;
		a = Umag / (np.exp(l**2 / sigma**2) - 1);	
		for j in range(0,N):
			U02[j] = a * np.exp((l**2 - y[j]**2) / sigma**2) - a;		
			H02[j] = - a * (np.sqrt(np.pi) * f0 * sigma * np.exp(l**2 / sigma**2) * erf(y[j] / sigma) / 2 
					- beta * sigma**2 * np.exp((l**2 - y[j]**2) / sigma**2) / 2
					- f0 * y[j] - beta * y[j]**2 / 2) / g + Hflat;
	
		# U03 - half strength BG flow
		Umag = 0.1;
		sigma = 0.3 * Ly;			
		l = Ly / 2;
		a = Umag / (np.exp(l**2 / sigma**2) - 1);	
		for j in range(0,N):
			U03[j] = a * np.exp((l**2 - y[j]**2) / sigma**2) - a;		
			H03[j] = - a * (np.sqrt(np.pi) * f0 * sigma * np.exp(l**2 / sigma**2) * erf(y[j] / sigma) / 2 
					- beta * sigma**2 * np.exp((l**2 - y[j]**2) / sigma**2) / 2
					- f0 * y[j] - beta * y[j]**2 / 2) / g + Hflat;
	
	
		# U04 - 'sharp' jet
		Umag = 0.2;
		sigma = 0.2 * Ly;			
		l = Ly / 2;
		a = Umag / (np.exp(l**2 / sigma**2) - 1);	
		for j in range(0,N):
			U04[j] = a * np.exp((l**2 - y[j]**2) / sigma**2) - a;		
			H04[j] = - a * (np.sqrt(np.pi) * f0 * sigma * np.exp(l**2 / sigma**2) * erf(y[j] / sigma) / 2 
					- beta * sigma**2 * np.exp((l**2 - y[j]**2) / sigma**2) / 2
					- f0 * y[j] - beta * y[j]**2 / 2) / g + Hflat;

		# U04 - 'sharper' jet
		Umag = 0.15;
		sigma = 0.2 * Ly;			
		l = Ly / 2;
		a = Umag / (np.exp(l**2 / sigma**2) - 1);	
		for j in range(0,N):
			U05[j] = a * np.exp((l**2 - y[j]**2) / sigma**2) - a;		
			H05[j] = - a * (np.sqrt(np.pi) * f0 * sigma * np.exp(l**2 / sigma**2) * erf(y[j] / sigma) / 2 
					- beta * sigma**2 * np.exp((l**2 - y[j]**2) / sigma**2) / 2
					- f0 * y[j] - beta * y[j]**2 / 2) / g + Hflat;
	
		# U05 - 'wide' jet
		Umag = 0.2;
		sigma = 0.4 * Ly;			
		l = Ly / 2;
		a = Umag / (np.exp(l**2 / sigma**2) - 1);	
		for j in range(0,N):
			U06[j] = a * np.exp((l**2 - y[j]**2) / sigma**2) - a;		
			H06[j] = - a * (np.sqrt(np.pi) * f0 * sigma * np.exp(l**2 / sigma**2) * erf(y[j] / sigma) / 2 
					- beta * sigma**2 * np.exp((l**2 - y[j]**2) / sigma**2) / 2
					- f0 * y[j] - beta * y[j]**2 / 2) / g + Hflat;
	
		U01 = U01 / U;
		U02 = U02 / U;
		U03 = U03 / U;
		U04 = U04 / U;
		U05 = U05 / U;
		U06 = U06 / U;
	
		H01 = H01 / H;
		H02 = H02 / H;
		H03 = H03 / H;
		H04 = H04 / H;
		H05 = H05 / H;
		H06 = H06 / H;
	
		plt.figure(1);
		plt.subplot(121);
		plt.plot(U01,y_nd,'b-',label='REFRENCE',linewidth=2);
		plt.plot(U02,y_nd,'r-',label='STRONG',linewidth=2);
		plt.plot(U03,y_nd,'g-',label='WEAK',linewidth=2);
		plt.plot(U04,y_nd,'k-',label='SHARP',linewidth=2);
		plt.plot(U05,y_nd,'m-',label='SHARPER',linewidth=2);
		plt.plot(U06,y_nd,'c-',label='WIDE',linewidth=2);
		plt.title('BG FLOW U0');
		plt.yticks((-1./2,0,1./2));
		plt.legend();
		plt.subplot(122);
		plt.plot(H01,y_nd,'b-',label='REFRENCE',linewidth=2);
		plt.plot(H02,y_nd,'r-',label='STRONG',linewidth=2);
		plt.plot(H03,y_nd,'g-',label='WEAK',linewidth=2);
		plt.plot(H04,y_nd,'k-',label='SHARP',linewidth=2);
		plt.plot(H05,y_nd,'m-',label='SHARPER',linewidth=2);
		plt.plot(H06,y_nd,'c-',label='WIDE',linewidth=2);
		plt.title('BG SSH H0')
		plt.yticks((-1./2,0,1./2));
		plt.legend();
		plt.tight_layout();
		plt.show();

# Plots the Rossby def. rad. against y
elif option == 2:
	Ld = np.zeros(N);
	for j in range(0,N):
		Ld[j] = np.sqrt(g * r0) / f[j];
	plt.plot(Ld,y)
	plt.show()

elif option == 3:
	E50 = np.load('/home/mike/Documents/GulfStream/Code/DATA/1L/EEFs/EEF_om50_y0.npy');
	E60 = np.load('/home/mike/Documents/GulfStream/Code/DATA/1L/EEFs/EEF_om60_y0.npy');
	E70 = np.load('/home/mike/Documents/GulfStream/Code/DATA/1L/EEFs/EEF_om70_y0.npy');
	NU = np.shape(E50)[0];
	#U0 = np.linspace(-0.3,0.5,NU);
	plt.plot(E50,'b-',label='50 days',linewidth=2);
	plt.plot(E60,'r-',label='60 days',linewidth=2);
	plt.plot(E70,'g-',label='70 days',linewidth=2);
	#plt.axhline(0,color='k',ls='--');
	#plt.axvline(0,color='k',ls='--');
	#plt.xlim(U0[0],U0[NU-1]);
	plt.title('Equivalent Eddy FLuxes',fontsize=18);
	plt.ylabel('EEF',fontsize=18);
	#plt.xlabel('U0',fontsize=18);
	plt.legend();
	plt.tight_layout();
	plt.show()

elif option == 4:
	
	N_set = [32,64,128,256,512,1024];
	p_set = [5,6,7,8,9,10];

	error1 = [1.16981294267e-19,1.45815391948e-19,1.46022581344e-19,1.10596154905e-19,8.14040437205e-20];
	error2 = [6.70739732654e-20,7.4529325698e-20,5.18529854606e-20,4.16850431505e-20,4.2138040004e-20];
	error3 = [1.16981294267e-19,1.45815391948e-19,1.46022581344e-19,1.10596154905e-19,8.14040437205e-20];

	plt.plot(N_set,error1);
	plt.show();

elif option == 10000:
	# Now overwrite the values with their derivatives
	uq = diff(uq,1,1,dx_nd);
	uQ = diff(uQ,1,1,dx_nd);
	Uq = diff(Uq,1,1,dx_nd);
	vQ = diff(vQ,0,0,dy_nd);
	vq = np.zeros((N,N,Nt));
	vq = diff(vq,0,0,dy_nd);	

	plt.figure(2);

	plt.subplot(321);
	plt.contourf(uq);
	plt.title('uq');
	plt.colorbar();

	plt.subplot(322);
	plt.contourf(uQ);
	plt.title('uQ');
	plt.colorbar();

	plt.subplot(323);
	plt.contourf(Uq);
	plt.title('Uq');
	plt.colorbar();

	plt.subplot(324);
	plt.contourf(vq);
	plt.title('vq');
	plt.colorbar();

	plt.subplot(325);
	plt.contourf(vQ);
	plt.title('vQ');
	plt.colorbar();

	plt.show();

	# It can be seen that vQ and uQ are relatively small. Let's look at zonal averages instead.
	uq_av = np.trapz(uq,x_nd[:N],dx_nd,axis=1);
	vQ_av = np.trapz(vQ,x_nd[:N],dx_nd,axis=1);
	uQ_av = np.trapz(uQ,x_nd[:N],dx_nd,axis=1);
	Uq_av = np.trapz(Uq,x_nd[:N],dx_nd,axis=1);
	vq_av = np.zeros((N,Nt));
	for ti in range(0,Nt):
		vq_av[:,ti] = np.trapz(vq[:,:,ti],x_nd[:N],dx_nd,axis=1);
		
	
	plt.figure(3);

	plt.subplot(321);
	plt.contourf(uq);
	plt.title('uq');
	plt.colorbar();
	plt.subplot(322);	
	plt.plot(uq_av,y_nd);

	plt.subplot(323);
	plt.contourf(vq[:,:,ts]);
	plt.title('vq');
	plt.colorbar();
	plt.subplot(324);	
	plt.plot(vq_av,y_nd);

	plt.subplot(325);
	plt.contourf(Uq);
	plt.title('Uq');
	plt.colorbar();
	plt.subplot(326);	
	plt.plot(Uq_av,y_nd);
	
	plt.show()

	plt.figure(4);
	plt.subplot(221);
	plt.contourf(vq[:,:,20]);
	plt.subplot(222);
	plt.contourf(vq[:,:,40]);
	plt.subplot(223);
	plt.contourf(vq[:,:,60]);
	plt.subplot(224);
	plt.contourf(vq[:,:,100]);
	plt.show()

#===================

#this code can be copied into RSW to test different PV contributions on the result of the footprint.

	plt.subplot(221);	
	plt.contourf(PV_prime1[:,:,ts]);
	plt.colorbar();
	plt.subplot(222);
	plt.contourf(PV_prime2[:,:,ts]);
	plt.colorbar();
	plt.subplot(223);	
	plt.contourf(PV_prime1[:,:,ts]+PV_prime2[:,:,ts]);
	plt.colorbar();
	plt.subplot(224);
	plt.contourf(PV_prime[:,:,ts]);
	plt.colorbar();
	plt.show();

	vq1 = v_nd * PV_prime1;
	vq2 = v_nd * PV_prime2;
	
	vq1 = diagnostics.timeAverage(vq1,T_nd,Nt);
	vq2 = diagnostics.timeAverage(vq2,T_nd,Nt);

	vq1_y = - diagnostics.diff(vq1,0,0,dy_nd);
	vq2_y = - diagnostics.diff(vq2,0,0,dy_nd);

	uq_x = diagnostics.timeAverage(uq,T_nd,Nt);
	uq_x = - diagnostics.diff(uq_x,1,1,dx_nd);

	if True:	
		plt.subplot(221);
		plt.contourf(vq1_y);
		plt.colorbar();
		plt.title('vq1_y');
		plt.subplot(222);
		plt.contourf(vq2_y);
		plt.colorbar();
		plt.title('vq2_y');
		plt.subplot(223);
		plt.contourf(vq1_y+vq2_y+uq_x);
		plt.colorbar();
		plt.title('vq1+vq2');
		plt.subplot(224);
		plt.contourf(P);
		plt.colorbar();
		plt.title('P');
		plt.show();

	vq1_y = diagnostics.extend(vq1_y);
	vq2_y = diagnostics.extend(vq2_y);
	uq_x = diagnostics.extend(uq_x);

	vq1 = np.trapz(vq1_y,x_nd,dx_nd,axis=1);
	vq2 = np.trapz(vq2_y,x_nd,dx_nd,axis=1);
	uq = np.trapz(uq_x,x_nd,dx_nd,axis=1);
	
	plt.plot(vq1,label='vq1');
	plt.plot(vq2,label='vq2');
	plt.plot(uq,label='uq');
	plt.legend();
	plt.show();

	vq = diagnostics.timeAverage(vq,T_nd,Nt);
	plotting.vqPlot(x_grid,y_grid,y_nd,v_nd,PV_prime,vq,P,P_xav,EEF,U0,ts);
	sys.exit();


