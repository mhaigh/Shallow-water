# correlation.py
#====================================================

# A set of functions for calculation of terms relating 
# to the correlation matrix, C = (K,0;0,K) + (M,N;N,M)

#====================================================

import numpy as np
import matplotlib.pyplot as plt

from diagnostics import timeAverage, diff

#====================================================

def M(u,v,T):
	"A function that calculates one of the two eddy shape/orientation measures"

	return 0.5 * timeAverage(u**2 - v**2, T, len(T)-1)

#====================================================

def N(u,v,T):
	"A function that calculates one of the two eddy shape/orientation measures"

	return timeAverage(u*v, T, len(T)-1)

#====================================================

def K(u,v,T):
	"A function that calculates the time-mean kinetic energy per unit mass"

	return 0.5 * timeAverage(u**2 + v**2, T, len(T)-1)

#====================================================

def D(u,v,Nt,dx,dy):
	"A function that calculates the horizontal divergence"

	D = np.zeros(np.shape(u))
	for ti in range(0,Nt):
		D[:,:,ti] = diff(u[:,:,ti],1,1,dx) + diff(v[:,:,ti],0,0,dy)

	return D

#====================================================

def Curl_uD(u,v,D_,T,dx,dy):
	"A function that calculates the time-mean curl of uD"

	uD = timeAverage(u * D_, T, len(T)-1)
	vD = timeAverage(v * D_, T, len(T)-1)

	return diff(vD,1,1,dx) - diff(uD,0,0,dy)	
	
#====================================================

def Curl_uD_components(u,v,D_,T,dx,dy):
	"A function that calculates the time-mean curl of uD"

	uD = timeAverage(u * D_, T, len(T)-1)
	vD = timeAverage(v * D_, T, len(T)-1)

	return diff(vD,1,1,dx), -diff(uD,0,0,dy)	
	
#====================================================

def orientation(M,N):
	"Find orientation from M,N"

	return 0.5 * np.arctan(N/M)

#====================================================

def anisotropy(M,K):
	"Find anisotropy from M, K"

	return M / K

#====================================================

def Mnorm(u,v,T):
	"Normalised M"

	return 0.5 * timeAverage((u**2 - v**2)/(u**2 + v**2), T, len(T)-1)

#====================================================

def Nnorm(u,v,T):
	"Normalised N"

	return timeAverage(u*v/(u**2+v**2), T, len(T)-1)

#====================================================

def plotOrientation(theta,K,x,y):
	"Plot orientation using quiver plot."

	scale = 5
	
	xq = x[::scale]
	yq = y[::scale]

	tx = np.cos(theta)[::scale,::scale]
	ty = np.sin(theta)[::5,::5]

	Klim = 0.05
	Kq = K[::scale,::scale]**0.5
	Kq = np.where(Kq>Klim*np.max(Kq),Klim*np.max(Kq),Kq) 
	
	tx *= Kq; ty *= Kq

	plt.quiver(xq,yq,tx,ty,headlength=0,headwidth=1,pivot='middle')
	plt.title('Orientation')
	plt.show()

#====================================================

# arrayCorr
def arrayCorr(u,v):
# Takes as input a pair of 2D arrays (e.g. solution snapshots), 
# and calculates their correlation by rewriting them as 1D lists.

	Ny, Nx = np.shape(u)
	
	dim = Ny * Nx

	u_list = np.zeros((dim))
	v_list = np.zeros((dim))

	# Rearrange square arrays into lists.
	for i in range(0,Nx):
		for j in range(0,Ny):
			u_list[i*Nx+j] = u[j,i]
			v_list[i*Nx+j] = v[j,i]

	# Correlation between two lists.
	return np.corrcoef(u_list,v_list)[0,1];
	
#====================================================

# arrayCorrTime
def arrayCorrTime(u,v):
# Uses the above-defined function to calculate the average correlation between two time-dependent arrays.
	
	Nt = np.shape(u)[2];
	
	# Initialise the correlation.
	corr = 0;

	# Add the correlation between u and v at each time step.
	for ti in range(0,Nt):
		corr = corr + arrayCorr(u[:,:,ti],v[:,:,ti]);
	
	# Average.
	return corr / Nt;

#====================================================

def plotComponents(x,y,M,N,K,D):
	
	import matplotlib.pyplot as plt
	
	Mlim = np.max(np.abs(M))
	Nlim = np.max(np.abs(N))
	Klim = np.max(np.abs(K))
	Dlim = np.max(np.abs(D))
	
	plt.subplot(221)
	plt.contourf(x[0:len(x)-1],y,M,vmin=-Mlim,vmax=Mlim)
	plt.colorbar()
	plt.xlim(-0.1,0.1)
	plt.ylim(-0.1,0.1)
	plt.title('M')
	plt.grid()

	plt.subplot(222)
	plt.contourf(x[0:len(x)-1],y,N,vmin=-Nlim,vmax=Nlim)
	plt.colorbar()
	plt.xlim(-0.1,0.1)
	plt.ylim(-0.1,0.1)
	plt.title('N')
	plt.grid()

	plt.subplot(223)
	plt.contourf(x[0:len(x)-1],y,K,vmin=-Klim,vmax=Klim)
	plt.colorbar()
	plt.xlim(-0.1,0.1)
	plt.ylim(-0.1,0.1)
	plt.title('K')
	plt.grid()

	plt.subplot(224)
	plt.contourf(x[0:len(x)-1],y,D,vmin=-Dlim,vmax=Dlim)
	plt.colorbar()
	plt.xlim(-0.1,0.1)
	plt.ylim(-0.1,0.1)
	plt.title('D')
	plt.grid()

	plt.tight_layout()
	plt.show()


















