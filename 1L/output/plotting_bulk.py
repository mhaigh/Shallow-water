# plotting_bulk.py

# This modules contains plotting functions that plot a bulk of arrays in one large figure.
# These plots are intended to be optimally presented.

#====================================================
#====================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

#====================================================
#====================================================

# plotSolutions
def plotSolutions(u,v,eta,N,x_grid,y_grid,row,nrows,string):
# Function that saves plots of the solutions (including PV) separately.

	u = np.real(u)
	v = np.real(v)
	eta = np.real(eta)

	ulim = np.max(abs(u));
	vlim = np.max(abs(v));
	etalim = np.max(abs(eta));

	u = u / ulim;
	v = v / vlim;
	eta = eta / etalim

	fs = 18
	x_ticks = (-1./2,-1./4,0,1./4,1./2)
	y_ticks = (-1./2,-1./4,0,1./4,1./2)

	u_str = 'max=' + str(round(ulim,4));
	v_str = 'max=' + str(round(vlim,4));
	eta_str = 'max=' + str(round(etalim,4));	

	# Define width ratio of each subplot
	grsp = gs.GridSpec(nrows,3,width_ratios=[1,1,1.2])
	x_grid = x_grid[0:N]

	cmap = 'bwr'
	#cmap = 'jet'

	lm = 1.0 

	# Now plots
	#==========
	
	plt.subplot(grsp[row,0])	
	plt.pcolor(x_grid, y_grid, u, cmap=cmap,vmin=-lm,vmax=lm)
	plt.yticks(y_ticks,fontsize=fs)
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()])
	if row == nrows-1:
		plt.xlabel('x',fontsize=fs+2)
		plt.xticks(x_ticks,fontsize=fs)
	else:
		plt.xticks(x_ticks,fontsize=0);
	plt.ylabel('y',fontsize=fs+2);
	plt.text(0.4,0.4,r'$u^{\prime}$',color='k',fontsize=fs+12);
	plt.text(-0.4,0.4,string,color='k',fontsize=fs+12);
	plt.text(-0.4,-0.4,u_str,color='k',fontsize=fs+4);
	plt.grid()

	plt.subplot(grsp[row,1])	
	plt.pcolor(x_grid, y_grid, v, cmap=cmap,vmin=-lm,vmax=lm);
	plt.yticks(y_ticks,fontsize=0);
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.text(0.4,0.4,r'$v^{\prime}$',fontsize=fs+12);
	plt.text(-0.4,-0.4,v_str,color='k',fontsize=fs+4);
	if row == nrows-1:
		plt.xlabel('x',fontsize=fs+2);
		plt.xticks(x_ticks,fontsize=fs);
	else:
		plt.xticks(x_ticks,fontsize=0);
	plt.grid()

	plt.subplot(grsp[row,2])	
	plt.pcolor(x_grid, y_grid, eta, cmap=cmap,vmin=-lm,vmax=lm)
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.yticks(y_ticks,fontsize=0);
	if row == nrows-1:
		plt.xlabel('x',fontsize=fs+2);
		plt.xticks(x_ticks,fontsize=fs);
	else:
		plt.xticks(x_ticks,fontsize=0);
	plt.text(0.4,0.4,r'$\eta^{\prime}$',fontsize=fs+12);
	plt.text(-0.4,-0.4,eta_str,color='k',fontsize=fs+4);
	plt.grid();
	plt.colorbar()

#====================================================

# plotSolutionsAmpPhase
def plotSolutionsAmpPhase(u,v,eta,N,x_grid,y_grid,row,nrows,string,fig):
# Function that saves plots of the solutions (including PV) separately.

	ulim = np.max(abs(u));
	vlim = np.max(abs(v));
	etalim = np.max(abs(eta));

	u = u / ulim;
	v = v / vlim;
	eta = eta / etalim

	fs = 18
	x_ticks = (-1./2,-1./4,0,1./4,1./2)
	y_ticks = (-1./2,-1./4,0,1./4,1./2)

	# Define width ratio of each subplot
	grsp = gs.GridSpec(2*nrows,3,width_ratios=[1,1,1.2])
	x_grid = x_grid[0:N]

	# Now plots
	#==========
	
	# Phase
	plt.subplot(grsp[2*row,0])	
	plt.pcolor(x_grid, y_grid, np.angle(u),vmin=-np.pi,vmax=np.pi);
	plt.yticks(y_ticks,fontsize=fs);	
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.xticks(x_ticks,fontsize=0);
	plt.ylabel('y',fontsize=fs+2);
	plt.text(0.4,-0.4,r'$u^{\prime}$',color='k',fontsize=fs+12);
	plt.text(-0.4,-0.4,string,color='k',fontsize=fs+12);
	plt.grid()

	plt.subplot(grsp[2*row,1])	
	plt.pcolor(x_grid, y_grid, np.angle(v),vmin=-np.pi,vmax=np.pi);
	plt.yticks(y_ticks,fontsize=0);
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.text(0.4,-0.4,r'$v^{\prime}$',fontsize=fs+12);
	plt.xticks(x_ticks,fontsize=0);
	plt.grid()

	plt.subplot(grsp[2*row,2])	
	cax = plt.pcolor(x_grid, y_grid, np.angle(eta),vmin=-np.pi,vmax=np.pi)
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.yticks(y_ticks,fontsize=0);
	plt.xticks(x_ticks,fontsize=0);
	plt.text(0.4,-0.4,r'$\eta^{\prime}$',fontsize=fs+12);
	plt.grid();
	cbar = fig.colorbar(cax, ticks=[-np.pi, 0, np.pi])
	cbar.ax.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'],fontsize=fs) 

	# Amplitude
	plt.subplot(grsp[2*row+1,0])	
	plt.pcolor(x_grid, y_grid, np.absolute(u),vmin=0.,vmax=0.5);
	plt.yticks(y_ticks,fontsize=fs);	
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	if row == nrows-1:
		plt.xlabel('x',fontsize=fs+2);
		plt.xticks(x_ticks,fontsize=fs);
	else:
		plt.xticks(x_ticks,fontsize=0);
	plt.ylabel('y',fontsize=fs);
	plt.text(0.4,0.4,r'$u^{\prime}$',color='w',fontsize=fs+12);
	plt.text(-0.4,0.4,string,color='w',fontsize=fs+12);
	plt.grid()

	plt.subplot(grsp[2*row+1,1])	
	plt.pcolor(x_grid, y_grid, np.absolute(v),vmin=0.,vmax=0.5);
	plt.yticks(y_ticks,fontsize=0);
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.text(0.4,0.4,r'$v^{\prime}$',color='w',fontsize=fs+12);
	if row == nrows-1:
		plt.xlabel('x',fontsize=fs+2);
		plt.xticks(x_ticks,fontsize=fs);
	else:
		plt.xticks(x_ticks,fontsize=0);
	plt.grid()

	plt.subplot(grsp[2*row+1,2])	
	plt.pcolor(x_grid, y_grid, np.absolute(eta),vmin=0.,vmax=0.5)
	plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
	plt.yticks(y_ticks,fontsize=0);
	if row == nrows-1:
		plt.xlabel('x',fontsize=fs+2);
		plt.xticks(x_ticks,fontsize=fs);
	else:
		plt.xticks(x_ticks,fontsize=0);
	plt.text(0.4,0.4,r'$\eta^{\prime}$',color='w',fontsize=fs+12);
	plt.grid();
	plt.colorbar()

#====================================================

# fp_PV_plot
def fp_PV_plot(PV_prime,P,P_xav,N,x_grid,y_grid,y_nd,row,nrows,string):

	PV_prime_lim = np.max(np.absolute(PV_prime));
	PV_prime = PV_prime / PV_prime_lim;

	Plim = np.max(np.absolute(P));
	P = P / Plim;
	
	P_xav = P_xav * 1.0e3;

	fs = 18
	x_ticks = (-1./4,0,1./4)
	#y_ticks = (-1./4,0,1./4)
	y_ticks = (0.,1./4,1./2)
	
	y_loc = 0.45
	#y_loc = 0.2

	#axs = [-1./4,1./4,-1./4,1./4]
	axs = [-1./4,1./4,0.,1./2]

	lim = 0.5

	grsp = gs.GridSpec(nrows,3,width_ratios=[1,1.25,1])

	#cmap = 'bwr'
	cmap = 'jet'

	#plt.subplot(nrows,3,1+3*row)
	plt.subplot(grsp[row,0])	
	plt.pcolor(x_grid, y_grid, PV_prime, cmap=cmap,vmin=-lim,vmax=lim);
	plt.yticks(y_ticks,fontsize=fs);	
	plt.axis(axs);
	if row == nrows-1:
		plt.xlabel('x',fontsize=fs+2);
		plt.xticks(x_ticks,fontsize=fs);
	else:
		plt.xticks(x_ticks,fontsize=0);
	plt.ylabel('y',fontsize=fs+2);
	plt.text(0.2,y_loc,r'$q^{\prime}$',color='k',fontsize=fs+12);
	plt.text(-0.2,y_loc,string,color='k',fontsize=fs+12);
	plt.grid(axs)
	#plt.gca().set_aspect('equal', adjustable='box')

	#plt.subplot(nrows,3,2+3*row)
	plt.subplot(grsp[row,1])	
	plt.pcolor(x_grid, y_grid, P, cmap=cmap,vmin=-lim,vmax=lim);
	plt.yticks(y_ticks,fontsize=0);
	plt.axis(axs);
	plt.text(0.2,y_loc,r'$P$',fontsize=fs+12);
	plt.colorbar()
	if row == nrows-1:
		plt.xlabel('x',fontsize=fs+2);
		plt.xticks(x_ticks,fontsize=fs);
	else:
		plt.xticks(x_ticks,fontsize=0);
	plt.grid(axs)
	
	#plt.subplot(nrows,3,3+3*row)
	plt.subplot(grsp[row,2])	
	plt.plot(P_xav,y_nd,'k-',linewidth=2)
	plt.text(0.8*max(abs(P_xav)),y_loc,r'$\langle P\rangle$',fontsize=fs+12)
	plt.yticks(y_ticks,fontsize=fs);
	plt.xticks(fontsize=fs)
	plt.xlim(-1.1*max(P_xav),1.1*max(P_xav))
	#plt.ylim(-1./4,1./4)
	plt.ylim(0,1./2)	
	plt.ylabel('y',fontsize=fs+2);
	plt.grid();


#====================================================

def plotPandN(P_xav,N_,y_nd,lim,y_ticks):


	fs = 12

	plt.plot(P_xav,y_nd,'k-',linewidth=1.5,label=r'$\langle P\rangle$')
	plt.plot(N_,y_nd,'r--',linewidth=2.,label=r'$\langle N_{yy}\rangle$')
	plt.yticks(y_ticks,fontsize=fs);
	plt.xticks(fontsize=fs)
	plt.xlim(-1.4*max(P_xav),1.4*max(P_xav))
	plt.ylim(lim[0],lim[1])
	#plt.ylim(0.125,0.375)	
	plt.ylabel('y',fontsize=fs+2);
	plt.grid();
	plt.show()

#====================================================

def plotKMN(K,M,N_,x_grid,y_grid,N,row,nrows,string):

	##M = M/K; N_ = N_/K

	Klim = np.max(np.abs(K))
	Nlim = np.max(np.abs(N_))
	Mlim = np.max(np.abs(M))

	Ks = 'max=' + str(round(Klim,4))
	Ms = 'max=' + str(round(Mlim,4))
	Ns = 'max=' + str(round(Nlim,4))

	if False:
		xlim = [-0.125,0.125]
		ylim = [0.125,0.375]
		xloc = -0.1
		yloc1 = 0.14
		yloc2 = 0.35
		xticks = np.linspace(-0.125,0.125,5)
		yticks = np.linspace(0.125,0.375,5)
	elif False:
		xlim = [-0.125,0.125]
		ylim = [-0.125,0.125]
		xloc = -0.1
		yloc1 = 0.14
		yloc2 = 0.35
		xticks = np.linspace(-0.125,0.125,5)
		yticks = np.linspace(-0.125,0.125,5)
	else:
		xlim = [-0.5,0.5]
		ylim = [-0.5,0.5]
		xloc1 = -0.4
		xloc2 = 0.4
		yloc1 = -0.4
		yloc2 = 0.4
		xticks = [-0.5,-0.25,0.0,0.25,0.5]
		yticks = [-0.5,-0.25,0.0,0.25,0.5]
		clr1 = 'k'
		clr2 = 'k'
		Ms = ''
		Ns = ''


	fs = 18

	cm = 'bwr'

	K /= Klim
	#M /= Mlim
	#N_ /= Nlim

	grsp = gs.GridSpec(nrows,3,width_ratios=[1.25,1.,1.25])
	x_grid = x_grid[0:N]

	plt.subplot(grsp[row,0])
	plt.pcolor(x_grid,y_grid,K,vmin=0,vmax=.5)
	plt.text(xloc2,yloc2,r'$K$',fontsize=fs+8,color='w')
	plt.text(xloc1,yloc2,string,fontsize=fs+8,color='w')
	plt.text(xloc1,yloc1,Ks,fontsize=fs,color='w')

	plt.xlim(xlim)
	plt.ylim(ylim)
	if row == nrows-1:
		plt.xlabel('x',fontsize=fs+2);
		plt.xticks(xticks,fontsize=fs);
	else:
		plt.xticks(xticks,fontsize=0);
	plt.yticks(yticks,fontsize=fs)
	plt.ylabel('y',fontsize=fs+2)
	plt.colorbar()
	plt.grid()

	#==
	
	plt.subplot(grsp[row,1])
	plt.pcolor(x_grid,y_grid,M,cmap=cm,vmin=-.5,vmax=.5)
	plt.text(xloc2,yloc2,r'$\hat{M}$',fontsize=fs+12,color=clr1)
	plt.text(xloc1,yloc1,Ms,fontsize=fs,color='k')
	plt.xlim(xlim)
	plt.ylim(ylim)

	if row == nrows-1:
		plt.xlabel('x',fontsize=fs+2);
		plt.xticks(xticks,fontsize=fs);
	else:
		plt.xticks(xticks,fontsize=0);
	plt.yticks(yticks,fontsize=fs)
	plt.ylabel('y',fontsize=fs+2)
	plt.grid()

	#==

	plt.subplot(grsp[row,2])
	plt.pcolor(x_grid,y_grid,N_,cmap=cm,vmin=-.5,vmax=.5)
	plt.text(xloc2,yloc2,r'$\hat{N}$',fontsize=fs+12,color=clr2)
	plt.text(xloc1,yloc1,Ns,fontsize=fs,color='k')
	plt.xlim(xlim)
	plt.ylim(ylim)

	if row == nrows-1:
		plt.xlabel('x',fontsize=fs+2);
		plt.xticks(xticks,fontsize=fs);
	else:
		plt.xticks(xticks,fontsize=0);
	plt.yticks(yticks,fontsize=0)
	plt.grid()
	plt.colorbar()




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
