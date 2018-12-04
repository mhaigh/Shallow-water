# plotting.py
# File containing numerous plotting functions.
#====================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

#====================================================

# vqPlot
def vqPlot(x_grid,y_grid,y_nd,v,PV_prime,vq,P,P_xav,EEF,U0,ts):
	
	U0 = max(U0);		
	EEF = round(10e6*EEF,5);
	
	vlim = np.max(abs(v[:,:,ts]));
	PVlim = np.max(abs(PV_prime[:,:,ts]));	
	vqlim = np.max(abs(vq));

	vmax = round(np.max(abs(v)),3);
	PVmax = round(np.max(abs(PV_prime)),5);
	vqmax = round(np.max(abs(vq)),5);
	Pmax = round(np.max(abs(P)),5);

	vlim = 0.15;
	PVlim = 0.0024;
	vqlim = 0.00010;
	Plim = 0.01;
	Pavlim = 0.00035;

	plt.figure(1,figsize=[16,16]);
	plt.subplot(221);
	plt.pcolor(x_grid,y_grid,v[:,:,ts],cmap='bwr',vmin=-vlim,vmax=vlim);
	plt.text(0.4,0.4,r'$v^{\prime}$',fontsize=26);
	plt.text(-0.4,-0.4,'U0='+str(U0),fontsize=18);
	plt.text(-0.4,0.4,'vmax='+str(vmax),fontsize=18);
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.xticks((-1./2,-1./4,0,1./4,1./2));
	plt.yticks((-1./2,-1./4,0,1./4,1./2));
	plt.colorbar();

	plt.subplot(222);
	plt.pcolor(x_grid,y_grid,vq,cmap='bwr',vmin=-vqlim,vmax=vqlim);
	plt.text(0.3,0.4,r'$v^{\prime}q^{\prime}$',fontsize=26)
	plt.text(-0.4,-0.4,'PVmax='+str(PVmax),fontsize=18);
	plt.text(-0.4,-0.3,'vqmax='+str(vqmax),fontsize=18);
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.colorbar();
	plt.xticks((-1./2,-1./4,0,1./4,1./2));
	plt.yticks((-1./2,-1./4,0,1./4,1./2));

	plt.subplot(223);
	plt.pcolor(x_grid,y_grid,P,cmap='bwr',vmin=-Plim,vmax=Plim);
	plt.text(0.3,0.4,r'$P$',fontsize=26);
	plt.text(-0.4,-0.4,'Pmax='+str(Pmax),fontsize=18);
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.colorbar();
	plt.xticks((-1./2,-1./4,0,1./4,1./2));
	plt.yticks((-1./2,-1./4,0,1./4,1./2));

	plt.subplot(224);
	plt.plot(P_xav,y_nd,'k-',linewidth=2)
	plt.text(0.8*max(abs(P_xav)),0.40,r'$\langle P\rangle$',fontsize=26)
	plt.yticks((-1./2,-1./4,0,1./4,1./2));
	plt.ylim(-0.5,0.5);
	plt.text(-0.0003,-0.4,'E='+str(EEF),fontsize=18);	
	# Change these next two lines depending on which BG flow we're working with.
	#plt.xlim(-7.,7);	
	#plt.xticks((-6.,-4.,-2.,0.,2.,4.,6.));
	plt.xlim(-Pavlim,Pavlim);
	# ===
	plt.ylabel('y',fontsize=18);
	plt.grid(b=True, which='both', color='0.65',linestyle='--');

	plt.tight_layout();
	plt.savefig('vq_U0=' + str(U0) + '.png');
	plt.show();

	
	

#====================================================

# forcingPlot_save
def forcingPlot_save(x_grid,y_grid,F3_nd,FORCE,BG,Fpos,N):

	aa = 1./24;
	Fmax = np.max(np.max(F3_nd,axis=0));
	Flim = np.max(abs(F3_nd/(1.0*Fmax)));
	#F3[0,0] = - Fmax;

	plt.pcolor(x_grid,y_grid,F3_nd/(1.1*Fmax),vmin=0.0,vmax=1.0);
	plt.xlabel('x',fontsize=22);
	plt.ylabel('y',fontsize=22);
	plt.text(0.4,0.4,r'$F_{3}$',fontsize=26,color='w');
	plt.arrow(-aa,2*aa+0.25,2*aa,0,head_width=0.02, head_length=0.02,color='w');
	plt.arrow(2*aa,aa+.25,0,-2*aa,head_width=0.02, head_length=0.02,color='w');
	plt.arrow(aa,-2*aa+.25,-2*aa,0,head_width=0.02, head_length=0.02,color='w');
	plt.arrow(-2*aa,-aa+.25,0,2*aa,head_width=0.02, head_length=0.02,color='w');
	plt.xticks((-1./2,-1./4,0,1./4,1./2));
	plt.yticks((-1./2,-1./4,0,1./4,1./2));	
	plt.xlabel('x',fontsize=16);
	plt.ylabel('y',fontsize=16);
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.colorbar();
	plt.tight_layout();
	plt.show()

	#plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/F_' + str(Fpos) + '_'  + str(N) + '.png');
	#plt.close();

#====================================================

# solutionPlots
def solutionPlots(x_nd,y_nd,u,v,h,ts,FORCE,BG,Fpos,N,x_grid,y_grid,div):

	ulim = np.max(abs(u[:,:,ts]));
	vlim = np.max(abs(v[:,:,ts]));
	etalim = np.max(abs(h[:,:,ts]));
	
	if div:
			
		ulim = np.max(abs(u[:,:,ts])) / 2
		vlim = np.max(abs(v[:,:,ts])) / 2
		etalim = np.max(abs(h[:,:,ts])) / 2

		plt.figure(1,figsize=(22,6.4));

		plt.subplot(131);
		plt.pcolor(x_grid, y_grid, u[:,:,ts], cmap='bwr', vmin=-ulim, vmax=ulim);
		plt.text(0.3,0.45,'u',fontsize=22);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));	
		plt.xlabel('x',fontsize=16);
		plt.ylabel('y',fontsize=16);
		plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()])
		plt.colorbar();
		plt.grid(b=True, which='both', color='0.65',linestyle='--');

		plt.subplot(132);
		plt.pcolor(x_grid, y_grid, v[:,:,ts], cmap='bwr', vmin=-vlim, vmax=vlim);
		plt.text(0.3,0.45,'v',fontsize=22);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));	
		plt.xlabel('x',fontsize=16);
		plt.ylabel('y',fontsize=16);
		plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
		plt.colorbar();
		plt.grid(b=True, which='both', color='0.65',linestyle='--');

		plt.subplot(133);
		plt.pcolor(x_grid, y_grid, h[:,:,ts], cmap='bwr', vmin=-etalim, vmax=etalim);
		plt.text(0.3,0.45,'eta',fontsize=22);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));	
		plt.xlabel('x',fontsize=16);
		plt.ylabel('y',fontsize=16);
		plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
		plt.colorbar();
		plt.grid(b=True, which='both', color='0.65',linestyle='--');

		plt.tight_layout();
		#plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/' + str(Fpos) + '_'  + str(N) + '.png');
		plt.show();

		# Another good option is 'seismic' (slightly darker shades).

	else:

		plt.figure(1,figsize=(22,6.4));

		plt.subplot(131);
		plt.contourf(x_nd[0:N],y_nd,u[:,:,ts]);
		plt.text(0.3,0.45,'u',fontsize=22);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));	
		plt.xlabel('x',fontsize=16);
		plt.ylabel('y',fontsize=16);
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.clim(-ulim,ulim);
		plt.colorbar();

		plt.subplot(132);
		plt.contourf(x_nd[0:N],y_nd,v[:,:,ts]);
		plt.text(0.3,0.45,'v',fontsize=22);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.clim(-vlim,vlim);
		plt.colorbar();

		plt.subplot(133);
		plt.contourf(x_nd[0:N],y_nd,h[:,:,ts]);
		plt.text(0.3,0.45,'eta',fontsize=22);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.clim(-etalim,etalim);
		plt.colorbar()

		plt.tight_layout();
		#plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/' + str(Fpos) + '_'  + str(N) + '.png');
		plt.show();

		#cmap='coolwarm'

#====================================================

# solutionPlots_save
def solutionPlots_save(x_nd,y_nd,u,v,h,ts,FORCE,BG,Fpos,N,U0_str,x_grid,y_grid,div):
# Function that saves plots of the solutions (including PV) separately.

	ulim = np.max(abs(u[:,:,ts]));
	vlim = np.max(abs(v[:,:,ts]));
	etalim = np.max(abs(h[:,:,ts]));

	u = u / ulim;
	v = v / vlim;
	h = h / etalim

	u = extend(u);
	v = extend(v);
	h = extend(h);

	u_str = 'max=' + str(round(ulim,4));
	v_str = 'max=' + str(round(vlim,4));
	eta_str = 'max=' + str(round(etalim,4));	

	if div:
			
		plt.pcolor(x_grid, y_grid, u[:,:,ts], cmap='bwr', vmin=-1., vmax=1.);
		plt.text(0.4,0.4,r'$u^{\prime}$',fontsize=26);
		plt.text(-0.45,0.4,U0_str,color='k',fontsize=22);
		plt.text(-0.45,-0.4,u_str,color='k',fontsize=18);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));	
		plt.xlabel('x',fontsize=16);
		plt.ylabel('y',fontsize=16);
		plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.tight_layout();
		plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/u_' + str(Fpos) + '_'  + str(N) + '.png');
		plt.close();

		plt.pcolor(x_grid, y_grid, v[:,:,ts], cmap='bwr', vmin=-1., vmax=1.);
		plt.text(0.4,0.4,r'$v^{\prime}$',fontsize=26);
		#plt.text(-0.45,0.4,U0_str,color='k',fontsize=22);
		plt.text(-0.45,-0.4,v_str,color='k',fontsize=18);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2),fontsize=0);	
		plt.xlabel('x',fontsize=16);
		plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.tight_layout();
		plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/v_' + str(Fpos) + '_'  + str(N) + '.png');
		plt.close();

		plt.pcolor(x_grid, y_grid, h[:,:,ts], cmap='bwr', vmin=-1., vmax=1.);
		plt.text(0.4,0.4,r'$\eta^{\prime}$',fontsize=26);
		#plt.text(-0.45,0.4,U0_str,color='k',fontsize=22);
		plt.text(-0.45,-0.4,eta_str,color='k',fontsize=18);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2),fontsize=0);	
		plt.xlabel('x',fontsize=16);
		plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.colorbar();
		plt.tight_layout();
		plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/eta_' + str(Fpos) + '_'  + str(N) + '.png');
		plt.close();



	else:

		plt.figure(1)
		plt.contourf(x_nd,y_nd,u[:,:,ts]);
		plt.text(0.4,0.4,r'$u^{\prime}$',fontsize=26);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));	
		plt.xlabel('x',fontsize=18);
		plt.ylabel('y',fontsize=18);
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.clim(-ulim,ulim);
		plt.colorbar();
		plt.tight_layout();
		plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/u_' + str(Fpos) + '_'  + str(N) + '.png');
		plt.close();

		plt.figure(2)
		plt.contourf(x_nd,y_nd,v[:,:,ts]);
		plt.text(0.4,0.4,r'$v^{\prime}$',fontsize=26);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));
		plt.xlabel('x',fontsize=18);
		plt.ylabel('y',fontsize=18);
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.clim(-vlim,vlim);
		plt.colorbar();
		plt.tight_layout();
		plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/v_' + str(Fpos) + '_'  + str(N) + '.png');
		plt.close();

		plt.figure(3)
		plt.contourf(x_nd,y_nd,h[:,:,ts]);
		plt.text(0.4,0.4,r'$\eta^{\prime}$',fontsize=26);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));
		plt.xlabel('x',fontsize=18);
		plt.ylabel('y',fontsize=18);
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.clim(-etalim,etalim);
		plt.colorbar();
		plt.tight_layout();
		plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/eta_' + str(Fpos) + '_'  + str(N) + '.png');
		plt.close();
		
#====================================================

# solutionPlotsAmp
# Plots of amplitude 
def solutionPlotsAmp(x_grid,y_grid,u,v,h,ts,FORCE,BG,Fpos,U0_name,U0_str,N):

	ulim = np.max(abs(u));
	vlim = np.max(abs(v));
	etalim = np.max(abs(h));

	u = u / ulim;
	v = v / vlim;
	h = h / etalim

	u = extend(u);
	v = extend(v);
	h = extend(h);

	plt.pcolor(x_grid, y_grid, np.absolute(u), vmin=0., vmax=0.5);
	plt.text(0.4,0.4,r'$u^{\prime}$',fontsize=26,color='w');
	plt.text(-0.45,0.4,U0_str,fontsize=22,color='w');
	plt.xticks((-1./2,-1./4,0,1./4,1./2));
	plt.yticks((-1./2,-1./4,0,1./4,1./2));	
	plt.xlabel('x',fontsize=16);
	plt.ylabel('y',fontsize=16);
	plt.yticks((-1./2,-1./4,0,1./4,1./2));
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.tight_layout();
	plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/u_Amp_' + U0_name + '.png');
	plt.close();
	
	plt.pcolor(x_grid, y_grid, np.absolute(v), vmin=0., vmax=0.5);
	plt.text(0.4,0.4,r'$v^{\prime}$',fontsize=26,color='w');
	plt.xticks((-1./2,-1./4,0,1./4,1./2));
	plt.yticks((-1./2,-1./4,0,1./4,1./2),fontsize=0);	
	plt.xlabel('x',fontsize=16);
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.tight_layout();
	plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/v_Amp_' + U0_name + '.png');
	plt.close();

	plt.pcolor(x_grid, y_grid, np.absolute(h), vmin=0., vmax=0.5);
	plt.text(0.4,0.4,r'$\eta^{\prime}$',fontsize=26,color='w');
	plt.xticks((-1./2,-1./4,0,1./4,1./2));
	plt.yticks((-1./2,-1./4,0,1./4,1./2),fontsize=0);
	plt.xlabel('x',fontsize=16);
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.colorbar();
	plt.tight_layout();
	plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/eta_Amp_' + U0_name + '.png');
	plt.close();


#====================================================

# solutionPlotsPhase
# Plots of phase 
def solutionPlotsPhase(x_grid,y_grid,u,v,h,ts,FORCE,BG,Fpos,U0_name,U0_str,N):

	u = extend(u)
	v = extend(v)
	h = extend(h)

	plt.pcolor(x_grid, y_grid, np.angle(u), cmap='hsv',vmin=-np.pi, vmax=np.pi);
	plt.text(0.4,-0.4,r'$u^{\prime}$',fontsize=26,color='k');
	plt.text(-0.45,-0.4,U0_str,fontsize=22,color='k');
	plt.xticks((-1./2,-1./4,0,1./4,1./2));
	plt.yticks((-1./2,-1./4,0,1./4,1./2),);	
	plt.xlabel('x',fontsize=16);
	plt.ylabel('y',fontsize=16);
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.tight_layout();
	plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/u_Phase_' + U0_name + '.png');
	plt.close();

	plt.pcolor(x_grid, y_grid, np.angle(v),cmap='rainbow',vmin=-np.pi, vmax=np.pi);
	plt.text(0.4,-0.4,r'$v^{\prime}$',fontsize=26,color='k');
	plt.xticks((-1./2,-1./4,0,1./4,1./2));
	plt.yticks((-1./2,-1./4,0,1./4,1./2),fontsize=0);
	plt.xlabel('x',fontsize=16);
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.tight_layout();
	plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/v_Phase_' + U0_name + '.png');
	plt.close();

	fig, ax = plt.subplots()
	cax = ax.pcolor(x_grid, y_grid, np.angle(h),vmin=-np.pi,vmax=np.pi);
	cbar = fig.colorbar(cax, ticks=[-np.pi, 0, np.pi])
	cbar.ax.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'],fontsize=18)  # vertically oriented colorbar
	plt.text(0.4,-0.4,r'$\eta^{\prime}$',fontsize=26,color='k');
	plt.xticks((-1./2,-1./4,0,1./4,1./2));	
	plt.yticks((-1./2,-1./4,0,1./4,1./2),fontsize=0);
	plt.xlabel('x',fontsize=16);
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.tight_layout();
	plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/eta_Phase_' + U0_name + '.png');
	plt.close();


#====================================================

# pvPlots
# Plots of PV and footprint
def pvPlots(PV_full,PV_prime,x_nd,y_nd):

	N = len(y_nd);

	plt.figure(1,figsize=[13,6]);

	plt.subplot(121);
	plt.contourf(x_nd[0:N],y_nd,PV_full[:,:,1]);
	plt.text(0.05,0.4,'PV FULL',color='k',fontsize=22);
	plt.xticks((-1./2,-1./4,0,1./4,1./2));
	plt.yticks((-1./2,-1./4,0,1./4,1./2));
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.colorbar();

	plt.subplot(122);
	plt.contourf(x_nd[0:N],y_nd,PV_prime[:,:,1]);
	plt.text(0.05,0.4,'PV PRIME',color='k',fontsize=22);
	plt.xticks((-1./2,-1./4,0,1./4,1./2));
	plt.yticks((-1./2,-1./4,0,1./4,1./2));
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.colorbar();

	plt.tight_layout()
	plt.show()

#====================================================

# pvPlots_save
def pvPlots_save(PV_full,PV_prime,x_nd,y_nd,ts,FORCE,BG,Fpos,N,U0_str,x_grid,y_grid,U0_name):

	PV_full_lim = np.max(abs(PV_full[:,:,ts]));
	PV_prime_lim = np.max(abs(PV_prime[:,:,ts]));

	PV_full = PV_full / PV_full_lim;
	PV_prime = PV_prime / PV_prime_lim;

	PV1_str = 'max=' + str(round(PV_full_lim,2));
	PV2_str = 'max=' + str(round(PV_prime_lim,2));

	plt.figure(1);
	plt.pcolor(x_grid, y_grid, PV_full[:,:,ts], cmap='bwr', vmin=-.5, vmax=.5);
	plt.text(0.4,0.4,r'$q$',color='k',fontsize=26);
	plt.text(-0.45,0.4,U0_str,color='k',fontsize=22);
	#plt.text(-0.45,-0.4,PV1_str,color='k',fontsize=18);
	plt.xticks((-1./2,-1./4,0,1./4,1./2));
	plt.yticks((-1./2,-1./4,0,1./4,1./2));
	plt.xlabel('x',fontsize=18);
	plt.ylabel('y',fontsize=18);
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.tight_layout();
	plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/PV_full_' + str(Fpos) + '_' + U0_name + '.png');
	plt.close();

	plt.figure(2);
	plt.pcolor(x_grid, y_grid, PV_prime[:,:,ts], cmap='bwr', vmin=-.5, vmax=.5);
	plt.text(0.4,0.4,r'$q^{\prime}$',color='k',fontsize=26);
	plt.text(-0.45,0.4,U0_str,color='k',fontsize=22);
	#plt.text(-0.45,-0.4,PV2_str,color='k',fontsize=18);
	plt.xticks((-1./2,-1./4,0,1./4,1./2));
	plt.yticks((-1./2,-1./4,0,1./4,1./2));
	plt.xlabel('x',fontsize=18);
	plt.ylabel('y',fontsize=18);
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.tight_layout();
	plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/PV_prime_' + str(Fpos) + '_'  + U0_name + '.png');
	plt.close();

#====================================================

# footprintPlots_save
def footprintPlots_save(P,P_xav,x_nd,y_nd,ts,FORCE,BG,Fpos,N,U0_str,x_grid,y_grid,U0_name):

	Plim = np.max(np.absolute(P));
	
	P = P / Plim;
	
	P_xav = P_xav * 1.0e5;

	P_str = 'max=' + str(round(Plim,2));

	plt.figure(1);
	plt.pcolor(x_grid, y_grid, P, cmap='bwr',vmin=-.5,vmax=.5);
	plt.text(0.4,0.4,r'$P$',fontsize=26);
	#plt.text(-0.45,-0.4,P_str,color='k',fontsize=18);
	#plt.text(0.25,0.4,str(Fpos),fontsize=18);		# Comment out this line if text on the plot isn't wanted.
	#plt.text(0.15,0.4,'r0 = '+str(r0/1000) + ' km' ,fontsize=18);	
	#plt.text(0.25,0.4,str(int(period_days))+' days',fontsize=18)
	#plt.text(0.25,0.4,'U0 = ' + str(U*U0_nd[0]),fontsize=18);
	#plt.text(0.25,0.4,r'$\nu$ = ' + str(int(nu)),fontsize=18);
	plt.xticks((-1./2,-1./4,0,1./4,1./2),);
	plt.yticks((-1./2,-1./4,0,1./4,1./2),fontsize=0);
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.xlabel('x',fontsize=18);
	#plt.ylabel('y',fontsize=18);
	plt.colorbar();
	plt.tight_layout();
	plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/P_' + str(Fpos) + '_' + U0_name + '.png');
	plt.close();
		
	plt.figure(2);
	plt.plot(P_xav,y_nd,'k-',linewidth=2)
	plt.text(0.8*max(abs(P_xav)),0.40,r'$\langle P\rangle$',fontsize=26)
	plt.yticks((-1./2,-1./4,0,1./4,1./2));
	plt.ylim(-0.5,0.5);
	# Change these next two lines depending on which BG flow we're working with.
	#plt.xlim(-7.,7);	
	#plt.xticks((-6.,-4.,-2.,0.,2.,4.,6.));
	plt.xlim(-1.1*np.max(abs(P_xav)),1.1*np.max(abs(P_xav)));
	# ===
	plt.ylabel('y',fontsize=18);
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.xlabel('x',color='white',fontsize=18);
	plt.tight_layout()
	plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/P_xav_' + str(Fpos) + '_' + U0_name + '.png');
	plt.close();
	

#====================================================

# forcingPlots
# Forcing plots 
def footprintPlots(x_nd,y_nd,P,P_xav,Fpos,BG,FORCE,nu,r0,period_days,U0_nd,U,N):
# Function that plots the forcing, and its Fourier representation.

	Umag = np.max(U0_nd);
	Plim = np.max(abs(P));

	plt.figure(1,figsize=(15,7))
	plt.subplot(121)
	#plt.contourf(x_nd,y_nd,P,cmap='coolwarm')
	plt.contourf(x_nd,y_nd,P);
	plt.text(0.0,0.4,'PV FOOTPRINT',fontsize=22);
	#plt.text(0.25,0.4,str(Fpos),fontsize=18);		# Comment out this line if text on the plot isn't wanted.
	#plt.text(0.15,0.4,'r0 = '+str(r0/1000) + ' km' ,fontsize=18);	
	#plt.text(0.25,0.4,str(int(period_days))+' days',fontsize=18)
	#plt.text(0.25,0.4,'U0 = ' + str(U*U0_nd[0]),fontsize=18);
	#plt.text(0.25,0.4,r'$\nu$ = ' + str(int(nu)),fontsize=18);
	plt.xticks((-1./2,-1./4,0,1./4,1./2));
	plt.yticks((-1./2,-1./4,0,1./4,1./2));
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.xlabel('x',fontsize=16);
	plt.ylabel('y',fontsize=16);
	plt.clim(-Plim,Plim)
	plt.colorbar()
	plt.subplot(122)
	plt.plot(P_xav,y_nd,linewidth=2)
	plt.text(10,0.40,'ZONAL AVERAGE',fontsize=22)
	plt.yticks((-1./2,0,1./2))
	plt.ylim(-0.5,0.5)
	plt.xlim(-1.1*np.max(abs(P_xav)),1.1*np.max(abs(P_xav)));
	plt.tight_layout()
	plt.show();
	
	# These if loops are for constantly altering depending on the test being done.
	if BG == 'GAUSSIAN':
		plt.figure(2)
		plt.contourf(x_nd,y_nd,P)
		plt.plot(U*U0_nd/(1.5*Umag)-0.5,y_nd,'k--',linewidth=2);
		#plt.text(0.25,0.4,str(period_days)+' days',fontsize=18)
		#plt.text(0.25,0.4,str(Fpos),fontsize=18);
		#plt.plot(P_xav[:,ts],y_nd,linewidth=2)
		#plt.text(0.25,0.4,'r0 = '+str(r0/1000),fontsize=18);	
		plt.colorbar()
		plt.ylim([-0.5,0.5]);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.xlabel('x');
		plt.ylabel('y');
		plt.tight_layout()

	if BG == 'UNIFORM':
		plt.figure(2)
		plt.contourf(x_nd,y_nd,P);
		plt.text(0.25,0.4,'U0 = ' + str(U*U0_nd[0]),fontsize=18);
		plt.colorbar()
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.tight_layout()
		#plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/FOOTPRINT_U0=' + str(U*U0_nd[0]) + '.png');

#====================================================

# footprintComponentsPlot
# A function that plots the footprint components.
def footprintComponentsPlot(uq,Uq,uQ,vq,vQ,P,P_uq,P_Uq,P_uQ,P_vq,P_vQ,P_xav,P_uq_xav,P_uQ_xav,P_Uq_xav,P_vq_xav,P_vQ_xav,x_nd,y_nd,N,Nt):

	uq_tav = np.zeros((N,N));
	Uq_tav = np.zeros((N,N));
	uQ_tav = np.zeros((N,N));
	vq_tav = np.zeros((N,N));
	vQ_tav = np.zeros((N,N));
	for j in range(0,N):
		for i in range(0,N):
			uq_tav[i,j] = sum(uq[i,j,:]) / Nt;
			Uq_tav[i,j] = sum(Uq[i,j,:]) / Nt;
			uQ_tav[i,j] = sum(uQ[i,j,:]) / Nt;
			vq_tav[i,j] = sum(vq[i,j,:]) / Nt;
			vQ_tav[i,j] = sum(vQ[i,j,:]) / Nt;

	print('Plotting time-averages of footprint components');

	plt.figure(1);

	plt.subplot(231);
	plt.contourf(uq_tav);
	plt.title('uq');
	plt.colorbar();

	plt.subplot(232);
	plt.contourf(uQ_tav);
	plt.title('uQ');
	plt.colorbar();

	plt.subplot(233);
	plt.contourf(Uq_tav);
	plt.title('Uq');
	plt.colorbar();

	plt.subplot(234);
	plt.contourf(vq_tav);
	plt.title('vq');
	plt.colorbar();

	plt.subplot(235);
	plt.contourf(vQ_tav);
	plt.title('vQ');
	plt.colorbar();

	plt.tight_layout();
	plt.show();

	#==

	print('Plotting contribution to footprint of each component (i.e. differentiated in space)');

	plt.figure(2);

	plt.subplot(231);
	plt.contourf(P_uq);
	plt.title('uq');
	plt.colorbar();

	plt.subplot(232);
	plt.contourf(P_uQ);
	plt.title('uQ');
	plt.colorbar();

	plt.subplot(233);
	plt.contourf(P_Uq);
	plt.title('Uq');
	plt.colorbar();

	plt.subplot(234);
	plt.contourf(P_vq);
	plt.title('vq');
	plt.colorbar();

	plt.subplot(235);
	plt.contourf(P_vQ);
	plt.title('vQ');
	plt.colorbar();

	plt.subplot(236);
	plt.contourf(P);
	plt.title('P');
	plt.colorbar();

	plt.tight_layout();
	plt.show();

	#==

	print('Plotting zonal averages');	
	
	plt.figure(3);

	plt.subplot(231);
	plt.plot(P_uq_xav,y_nd,linewidth=2);
	plt.title('uq');

	plt.subplot(232);
	plt.plot(P_uQ_xav,y_nd,linewidth=2);
	plt.title('uQ');

	plt.subplot(233);
	plt.plot(P_Uq_xav,y_nd,linewidth=2);
	plt.title('Uq');

	plt.subplot(234);
	plt.plot(P_vq_xav,y_nd,linewidth=2);
	plt.title('vq');

	plt.subplot(235);
	plt.plot(P_vQ_xav,y_nd,linewidth=2);
	plt.title('vQ');

	plt.subplot(235);
	plt.plot(P_xav,y_nd,linewidth=2);
	plt.title('P_xav');

	plt.tight_layout()
	plt.show();

#====================================================

# plotFluxes
def plotPrimaryComponents(P_uq,P_vq,P_uq_xav,P_vq_xav,x_nd,y_nd,FORCE,BG,Fpos,N):
# This function plots two footprint components uq and vq, and saves the output.
# These are the two fluxes that we are most interested in.

	print('Now plotting Primary components: contributions from uq and vq');
	
	P_uq_lim = np.max(abs(P_uq));
	P_vq_lim = np.max(abs(P_vq));

	U0_str = r'$U_{0}=0.16$';

	plt.figure(10);
	plt.contourf(x_nd,y_nd,P_uq);
	plt.text(0.4,0.4,r'$P_{u}$',color='k',fontsize=26);
	plt.xticks((-1./2,-1./4,0,1./4,1./2));
	plt.yticks((-1./2,-1./4,0,1./4,1./2));
	plt.xlabel('x',fontsize=18);
	plt.ylabel('y',fontsize=18);
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.clim(-P_uq_lim,P_uq_lim);
	plt.colorbar();
	plt.tight_layout();
	plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/P_uq_' + str(Fpos) + '_'  + str(N) + '.png');
	plt.close();
	
	plt.figure(11);
	plt.contourf(x_nd,y_nd,P_vq);
	plt.text(0.4,0.4,r'$P_{v}$',color='k',fontsize=26);
	plt.xticks((-1./2,-1./4,0,1./4,1./2));
	plt.yticks((-1./2,-1./4,0,1./4,1./2));
	plt.xlabel('x',fontsize=18);
	plt.ylabel('y',fontsize=18);
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.clim(-P_vq_lim,P_vq_lim);
	plt.colorbar();
	plt.tight_layout();
	plt.savefig('/home/mike/Documents/GulfStream/RSW/IMAGES/1L/' + str(FORCE) + '/' + str(BG) +  '/P_vq_' + str(Fpos) + '_'  + str(N) + '.png');
	plt.close();

	plt.figure(12);
	fig, ax1 = plt.subplots();
	ax1.plot(P_uq_xav,y_nd,'b-',linewidth=2);
	#ax1.set_xlabel(r'$\langle P_{v}\rangle$',color='b',fontsize=24);
	ax1.tick_params('x',colors='b');
	# Make the y-axis label, ticks and tick labels match the line color.
	ax1.set_yticks((-1./2,-1./4,0,1./4,1./2));
	ax1.set_ylabel('y',fontsize=18);
	ax1.text(-0.05,0.4,r'$\langle P_{v}\rangle$',color='r',fontsize=26);
	ax1.text(0.04,-0.4,r'$\langle P_{u}\rangle$',color='b',fontsize=26);
	#ax1.tick_params('y');
	ax2 = ax1.twiny();Arid()
	#ax1.tick_params('y');
	ax2 = ax1.twinx();
	ax2.plot(y_nd,U,'r-',linewidth=1.3);
	#ax2.set_xlabel(r'$\langle P_{v}\rangle$', color='r',fontsize=24);
	ax2.tick_params('y',colors='r');
	ax2.text(-0.4,0.1,r'$Q_{y}$',color='b',fontsize=26);
	ax2.text(0.4,0.1,r'$U_{0}$',color='r',fontsize=26);
	#ax2.tick_params('y', colors='r');
	fig.tight_layout();
	plt.show()

#====================================================

# plot_xshift
def plot_xshift():

	xshift=np.load('/home/mike/Documents/GulfStream/RSW/DATA/1L/Paper1/xshift.npy');	

	N = len(xshift);
	U0 = np.linspace(-0.5,0.5,N);

	plt.plot(U0,xshift,'k',linewidth=2);
	plt.ylim(-.12,.12);
	plt.xlim(-0.5,0.5);
	plt.xticks((-0.5,-.4,-.3,-.2,-.1,0.,0.1,0.2,0.3,0.4,0.5));
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.xlabel('U0',fontsize=18)
	plt.ylabel('Footprint zonal shift',fontsize=18);
	plt.show();	


#====================================================

# forcingPlots
def forcingPlots(x_nd,y_nd,F1_nd,F2_nd,F3_nd,Ftilde1_nd,Ftilde2_nd,Ftilde3_nd,N):
# Function that plots the forcing, and its Fourier representation.

	plt.figure(1);

	plt.subplot(331);
	plt.contourf(x_nd,y_nd,F1_nd);
	plt.xticks((-1./2,-1./4,0,1./4,1./2));
	plt.yticks((-1./2,-1./4,0,1./4,1./2));
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.colorbar();
	plt.subplot(332);
	plt.contourf(x_nd,y_nd,F2_nd);
	plt.xticks((-1./2,-1./4,0,1./4,1./2));
	plt.yticks((-1./2,-1./4,0,1./4,1./2));
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.colorbar();
	plt.subplot(333);
	plt.contourf(x_nd,y_nd,F3_nd);
	plt.xticks((-1./2,-1./4,0,1./4,1./2));
	plt.yticks((-1./2,-1./4,0,1./4,1./2));
	plt.grid(b=True, which='both', color='0.65',linestyle='--');
	plt.colorbar()

	plt.subplot(334);
	plt.contourf(np.real(Ftilde1_nd));
	plt.colorbar()
	plt.subplot(335);
	plt.contourf(np.real(Ftilde2_nd));
	plt.colorbar()
	plt.subplot(336);
	plt.contourf(np.real(Ftilde3_nd));
	plt.colorbar()

	plt.subplot(337);
	plt.contourf(np.imag(Ftilde1_nd));
	plt.colorbar()
	plt.subplot(338);
	plt.contourf(np.imag(Ftilde2_nd));
	plt.colorbar()
	plt.subplot(339);
	plt.contourf(np.imag(Ftilde3_nd));
	plt.colorbar()

	plt.show();


#====================================================

# extend
def extend(f):
# A function used to replace the extra x-gridpoint on a solution.

	dimx = np.shape(f)[1];
	dimy = np.shape(f)[0];
	if f.size != dimx * dimy:
		dimt = np.shape(f)[2];

		f_new = np.zeros((dimy,dimx+1,dimt),dtype=f.dtype);
		for i in range(0,dimx):
			f_new[:,i,:] = f[:,i,:];
	
		f_new[:,dimx,:] = f[:,0,:];
	
	else:
		f_new = np.zeros((dimy,dimx+1),dtype=f.dtype);
		for i in range(0,dimx):
			f_new[:,i] = f[:,i];
	
		f_new[:,dimx] = f[:,0];

	return f_new


