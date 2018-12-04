# eigDiagnostics.py
# File containing functions to be called by the master script EIG.py.
#====================================================

import sys

import numpy as np
import matplotlib.pyplot as plt

#====================================================

# eigPlot
def eigPlots(u_proj,v_proj,eta_proj,u,v,h,x_nd,y_nd,x_grid,y_grid,sol):
	
	ulim = np.max(abs(u));
	vlim = np.max(abs(v));
	etalim = np.max(abs(h));

	if sol:

		plt.figure(1,figsize=[21,10]);

		plt.subplot(231);
		plt.pcolor(x_grid,y_grid,u_proj, cmap='bwr', vmin=-ulim, vmax=ulim);
		plt.text(0.3,0.45,'u proj',fontsize=22);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));	
		plt.xlabel('x',fontsize=16);
		plt.ylabel('y',fontsize=16);
		plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
		plt.colorbar();

		plt.subplot(232);
		plt.pcolor(x_grid,y_grid,v_proj, cmap='bwr', vmin=-vlim, vmax=vlim);
		plt.text(0.3,0.45,'v proj',fontsize=22);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));	
		plt.xlabel('x',fontsize=16);
		plt.ylabel('y',fontsize=16);
		plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
		plt.colorbar();

		plt.subplot(233);
		plt.pcolor(x_grid,y_grid,eta_proj, cmap='bwr', vmin=-etalim, vmax=etalim);
		plt.text(0.3,0.45,'eta proj',fontsize=22);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));	
		plt.xlabel('x',fontsize=16);
		plt.ylabel('y',fontsize=16);
		plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
		plt.colorbar();
	
		plt.subplot(234);
		plt.pcolor(x_grid,y_grid,u, cmap='bwr', vmin=-ulim, vmax=ulim);
		plt.text(0.3,0.45,'u',fontsize=22);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));	
		plt.xlabel('x',fontsize=16);
		plt.ylabel('y',fontsize=16);
		plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
		plt.colorbar();

		plt.subplot(235);
		plt.pcolor(x_grid,y_grid,v, cmap='bwr', vmin=-vlim, vmax=vlim);
		plt.text(0.3,0.45,'v',fontsize=22);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));	
		plt.xlabel('x',fontsize=16);
		plt.ylabel('y',fontsize=16);
		plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
		plt.colorbar();

		plt.subplot(236);
		plt.pcolor(x_grid,y_grid,h, cmap='bwr', vmin=-etalim, vmax=etalim);
		plt.text(0.3,0.45,'eta',fontsize=22);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));	
		plt.xlabel('x',fontsize=16);
		plt.ylabel('y',fontsize=16);
		plt.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]);
		plt.colorbar();

		plt.tight_layout();
		plt.show();

	# Code never executed, unless we want contourf plots.
	elif 1==0:

		u = extend(u);
		v = extend(v);
		h = extend(h);

		plt.figure(1,figsize=[21,6]);
		plt.subplot(231)
		plt.contourf(x_nd,y_nd,u_proj,vmin=-ulim,vmax=ulim);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));	
		plt.xlabel('x');
		plt.ylabel('y');
		plt.clim(-ulim,ulim);
		plt.colorbar();

		plt.subplot(232)
		plt.contourf(x_nd,y_nd,v_proj,vmin=-vlim,vmax=vlim);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));
		plt.xlabel('x');
		plt.ylabel('y');
		plt.clim(-vlim,vlim);
		plt.colorbar();

		plt.subplot(233)
		plt.contourf(x_nd,y_nd,eta_proj,vmin=-etalim,vmax=etalim);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));	
		plt.xlabel('x');
		plt.ylabel('y');
		plt.clim(-etalim,etalim);
		plt.colorbar();
	
		plt.subplot(234);
		plt.contourf(x_nd,y_nd,u,vmin=-ulim,vmax=ulim);
		plt.text(0.3,0.45,'u',fontsize=22);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));	
		plt.xlabel('x',fontsize=16);
		plt.ylabel('y',fontsize=16);
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.clim(-ulim,ulim);
		plt.colorbar();

		plt.subplot(235);
		plt.contourf(x_nd,y_nd,v,vmin=-vlim,vmax=vlim);
		plt.text(0.3,0.45,'v',fontsize=22);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.clim(-vlim,vlim);
		plt.colorbar();

		plt.subplot(236);
		plt.contourf(x_nd,y_nd,h,vmin=-etalim,vmax=etalim);
		plt.text(0.3,0.45,'eta',fontsize=22);
		plt.xticks((-1./2,-1./4,0,1./4,1./2));
		plt.yticks((-1./2,-1./4,0,1./4,1./2));
		plt.grid(b=True, which='both', color='0.65',linestyle='--');
		plt.clim(-etalim,etalim);
		plt.colorbar();

		plt.tight_layout();
		plt.show();


	else:	

		plt.figure(1,figsize=[21,6]);
		plt.subplot(131)
		plt.contourf(x_nd,y_nd,u_proj);
		plt.xticks((-1./2,0,1./2));
		plt.yticks((-1./2,0,1./2));	
		plt.xlabel('x');
		plt.ylabel('y');
		plt.colorbar();
		plt.subplot(132)
		plt.contourf(x_nd,y_nd,v_proj);
		plt.xticks((-1./2,0,1./2));
		plt.yticks((-1./2,0,1./2));	
		plt.xlabel('x');
		plt.ylabel('y');
		plt.colorbar();
		plt.subplot(133)
		plt.contourf(x_nd,y_nd,eta_proj);
		plt.xticks((-1./2,0,1./2));
		plt.yticks((-1./2,0,1./2));	
		plt.xlabel('x');
		plt.ylabel('y');
		plt.colorbar();
		plt.tight_layout();
		plt.show();

#====================================================

# scatterWeight
def scatterWeight(k,l,theta,theta_abs_tot,dom_index,Nm,Nk_neg,Nk_pos,Fpos):
	
	Nk = Nk_neg + Nk_pos + 1;

	dim = Nk * Nm;

	# Produce a set of weights normalised by theta_abs_tot, which is either
	# the total weight of all wavenumbers or just the first Nm wavenumbers.
	theta_normalised = np.zeros(dim);
	for i in range(0,Nk):
		for j in range(0,Nm):
			theta_normalised[i*Nm+j] = np.abs(theta[j,i] / theta_abs_tot[i]);

	theta_max = max(theta_normalised);
	
	# Calculate the performance of the decompisition at each wavenumber; 1 is perfect.
	perf = np.zeros(Nk);
	for i in range(0,Nk):
		perf[i] = sum(theta_normalised[i*Nm:(i+1)*Nm]);

	# Some points may be repeated, the next few lines allow repeated points to be plotted as different shapes.
	# First arrange all points as list of tuples
	L = [];
	L_theta = [];
	for ii in range(0,dim):
		L.append(tuple([k[ii],l[ii]]));
		L_theta.append(tuple([k[ii],l[ii],theta_normalised[ii]]));

	# Turn this list into a set and count how many time each element in the set S appears in the list L.	
	S = set(L);
	S_theta = set(L_theta);
	F = {};
	for ii in list(S):	# Loop through each element in the set S
		F[ii] = L.count(ii);	# And count how many times it appears in L

	# We also need a dictionary containing the weights of each tuple (or maximum weight if a tuple is repeated).
 	F_theta = {};
	for wi in range(0,dim):
		ii = L[wi];
		if ii in F_theta:							# Check if this key is already in the dictionary.
			if theta_normalised[wi] > F_theta[ii]:	# If so, check if new theta is larger than the previous...
				F[ii] = theta_normalised[wi];		# and if so, overwrite the old theta value.
		else:
			F_theta[ii] = theta_normalised[wi];		# Create a new key, with corresponding value theta

	# Now loop through all the tuples (keys) in the dictionary, storing each tuple in an array depending on the number of times it is repeated.
	k1 = []; l1 = []; theta1 = [];
	k2 = []; l2 = []; theta2 = [];
	k3 = []; l3 = []; theta3 = [];
	k4 = []; l4 = []; theta4 = [];
	for ii in list(S):
		if F[ii] == 1:
			k1.append(ii[0]); l1.append(ii[1]); theta1.append(F_theta[ii]);
		elif F[ii] == 2:
			k2.append(ii[0]); l2.append(ii[1]); theta2.append(F_theta[ii]);
		elif F[ii] == 3:
			k3.append(ii[0]); l3.append(ii[1]); theta3.append(F_theta[ii]);
		else:
			k4.append(ii[0]); l4.append(ii[1]); theta4.append(F_theta[ii]);
		
	k1 = np.array(k1); l1 = np.array(l1); theta1 = np.array(theta1);
	k2 = np.array(k2); l2 = np.array(l2); theta2 = np.array(theta2);
	k3 = np.array(k3); l3 = np.array(l3); theta3 = np.array(theta3);
	k4 = np.array(k4); l4 = np.array(l4); theta4 = np.array(theta4);

	colors1 = theta1; colors2 = theta2;
	colors3 = theta3; colors4 = theta4;
	
	l_max = max(l);
	if l_max % 2 == 0:
		y_max = l_max + 2;
	else:
		y_max = l_max + 1;

	#print(colors2);
	
	y_ticks = np.linspace(0,y_max,y_max/2+1);

	cm = 'YlOrRd';

	plt.scatter(k1,l1,c=colors1,cmap=cm,vmin=0,vmax=theta_max,marker='s',s=50);
	plt.scatter(k2,l2,c=colors2,cmap=cm,vmin=0,vmax=theta_max,marker='o',s=50);
	plt.scatter(k3,l3,c=colors3,cmap=cm,vmin=0,vmax=theta_max,marker='^',s=50);
	plt.scatter(k4,l4,c=colors4,cmap=cm,vmin=0,vmax=theta_max,marker='*',s=50);
	plt.ylim([0,y_max+4]);	
	plt.yticks(y_ticks);
	plt.colorbar();#ticks=np.linspace(0,theta_max,10)
	plt.grid();
	for i in range(-Nk_neg,Nk_pos+1):
		plt.text(i-0.25,y_max+0.8,str(round(perf[i],2)),fontsize=14,rotation=90);
	plt.title('Eigenmode decomposition - ' + str(Fpos),fontsize=22);
	plt.text(Nk_pos-4,y_max-6,'Weight',fontsize=22);
	plt.xlabel('k',fontsize=22);
	plt.ylabel('l',fontsize=22);
	plt.show();	
	

#====================================================

# scatterPeriod
def scatterPeriod(k,l,p,dom_index,Nm,Nk_neg,Nk_pos,Fpos):
	
	Nk = Nk_neg + Nk_pos + 1;

	dim = Nk * Nm;

	p = abs(p);	# Interested in period regardless of direction.
	pmin = 0;
	pmax = 2*60.0;#np.max(p);

	# Some points may be repeated, the next few lines allow repeated points to be plotted as different shapes.
	# First arrange all points as list of tuples
	L = [];
	L_p = [];
	for ii in range(0,dim):
		L.append(tuple([k[ii],l[ii]]));
		L_p.append(tuple([k[ii],l[ii],p[ii]]));

	# Turn this list into a set and count how many time each element in the set S appears in the list L.	
	S = set(L);
	S_p = set(L_p);
	F = {};
	for ii in list(S):			# Loop through each element in the set S...
		F[ii] = L.count(ii);	# and count how many times it appears in L

	# We also need a dictionary containing the periods of each tuple.
 	F_p = {};
	for wi in range(0,dim):
		ii = L[wi];
		if ii in F_p:				# Check if this key is already in the dictionary.
			F[ii] = p[wi];		
		else:
			F_p[ii] = p[wi];		# Create a new key, with corresponding value p[wi].

	# Now loop through all the tuples (keys) in the dictionary, storing each tuple in an array depending on the number of times it is repeated.
	k1 = []; l1 = []; p1 = [];
	k2 = []; l2 = []; p2 = [];
	k3 = []; l3 = []; p3 = [];
	k4 = []; l4 = []; p4 = [];
	for ii in list(S):
		if F[ii] == 1:
			k1.append(ii[0]); l1.append(ii[1]); p1.append(F_p[ii]);
		elif F[ii] == 2:
			k2.append(ii[0]); l2.append(ii[1]); p2.append(F_p[ii]);
		elif F[ii] == 3:
			k3.append(ii[0]); l3.append(ii[1]); p3.append(F_p[ii]);
		else:
			k4.append(ii[0]); l4.append(ii[1]); p4.append(F_p[ii]);
		
	k1 = np.array(k1); l1 = np.array(l1); p1 = np.array(p1);
	k2 = np.array(k2); l2 = np.array(l2); p2 = np.array(p2);
	k3 = np.array(k3); l3 = np.array(l3); p3 = np.array(p3);
	k4 = np.array(k4); l4 = np.array(l4); p4 = np.array(p4);

	colors1 = p1; colors2 = p2;
	colors3 = p3; colors4 = p4;
	
	l_max = max(l);
	if l_max % 2 == 0:
		y_max = l_max + 2;
	else:
		y_max = l_max + 1;

	#print(colors2);
	
	y_ticks = np.linspace(0,y_max,y_max/2+1);

	#cm = 'Set1'
	cm = plt.cm.get_cmap('Set2',8);

	plt.scatter(k1,l1,c=colors1,cmap=cm,vmin=pmin,vmax=pmax,marker='s',s=50);
	plt.scatter(k2,l2,c=colors2,cmap=cm,vmin=pmin,vmax=pmax,marker='o',s=50);
	plt.scatter(k3,l3,c=colors3,cmap=cm,vmin=pmin,vmax=pmax,marker='^',s=50);
	plt.scatter(k4,l4,c=colors4,cmap=cm,vmin=pmin,vmax=pmax,marker='*',s=50);
	plt.ylim([0,y_max+4]);	
	plt.yticks(y_ticks);
	plt.colorbar();#ticks=np.linspace(0,theta_max,10)
	plt.grid();
	plt.title('Eigenmode decomposition - ' + str(Fpos),fontsize=22);
	plt.text(Nk_pos-6,y_max,'Period (days)',fontsize=22);
	plt.xlabel('k',fontsize=22);
	plt.ylabel('l',fontsize=22);
	plt.show();	

#====================================================

# vec2field
def vec2field(u_vec,freq,x_nd,k,N,Ts):

	I = np.complex(0.0,1.0)
	
	u = np.zeros((N,N))
	for i in range(0,N):
		for j in range(0,N):
			u[j,i] = np.real(u_vec[j] * np.exp(2.0 * np.pi * I * (k * x_nd[i] - freq * Ts)))

	return u

#====================================================

# orderEigenmodes
def orderEigenmodes(u_vec,x_nd,k,N,dim):
	'''Find meridional wavenumber by using fft. Returns count, which stores meridional wavenunmber. 
	Some modes may have half-wavenumbers. To best capture this, we store two most dominant wavenumbers
	along with the ratio of their amplitudes in fft decomposition.'''
	
	Nl = N//2+1
	uxx = np.zeros((Nl,dim))

	# Define cos and sin at reference longitude 
	theta = 2*np.pi*k*x_nd[0]
	cosx = np.cos(theta); sinx = np.sin(theta)

	# u at reference lon
	ux = np.real(u_vec) * cosx - np.imag(u_vec) * sinx

	ux = np.abs(np.fft.fft(ux,axis=0))

	uxx[0,:] = ux[0,:]
	uxx[1:Nl,:] = ux[1:Nl,:] + ux[N:N-Nl:-1,:]

	count_tmp = np.argsort(-uxx,axis=0)
	count = count_tmp[0]
	count2 = count_tmp[1]

	ratio = np.zeros(dim)
	for wi in range(0,dim):
		ratio[wi] = uxx[count[wi],wi] / uxx[count2[wi],wi]

	#print(count[606]); print(count2[606]); print(ratio[606])
	#plt.plot(uxx[:,606]); plt.xlim(0,15); plt.show()
	
	i_count = np.argsort(count)

	return count, count2, ratio, i_count;

#====================================================

# orderEigenmodes2
def orderEigenmodes2(vec,val,N,VECS):
	'''A function that takes the set of eigenmodes, given by vec, and orders them according to the number of zero crossings.
		When two or more eigenmodes cross zeros the same amount of times, they are ordered by their frequency, smallest first.'''
	
	dim = np.size(val);

	if VECS:
		vec = np.array(vec);
		u_vec = vec[0,:,:];
		v_vec = vec[1,:,:];
		h_vec = vec[2,:,:];

	else:
		u_vec = vec[0:N,:];		# Extract the eigenmodes.
		v_vec = vec[N:2*N,:];
		h_vec = vec[2*N:3*N,:];		

	# Initialise a counter for the number of zero crossings. 
	count = np.zeros((dim),dtype=int);
	for wi in range(0,dim):
		u_abs = np.abs(u_vec[:,wi]);
		u_av = np.mean(u_abs);
		for j in range(1,N):
			if (u_abs[j-1] >= u_av and u_abs[j] <= u_av) or (u_abs[j-1] <= u_av and u_abs[j] >= u_av):
				count[wi] = count[wi] + 1;
	
	for wi in range(0,dim):
		count[wi] = int(np.floor(count[wi] / 2));

	i_count = np.argsort(count);

	return count, i_count;

#====================================================

# updateCount
def updateCount(count,wii):
# A function that takes raw input to manually update the zero-crossings count of an eigenvector.
# Type end to quit updating.

	update_count = raw_input('-->');		# The first step updates count, but i_count will no longer match.									
	if update_count != '' and update_count != 'end':	
		print('updated');
		count_new = int(update_count);	# Stores the new count, to be used to rearrange the vectors.
 	elif update_count == 'end':
		count_new = count;
		wii = wii + dim; 	# End the loop, and don't update any more modes.	
	else:
		count_new = count;

	return count_new, wii;

#====================================================

# vec2vecs
def vec2vecs(vec,N,dim,BC):
# A function to take the full eigenvector, vec = (u,v,eta), and 
# extract u, v, eta, depending on the boundary condition.

	if BC == 'FREE-SLIP':
		u_vec = vec[0:N,:];
		v_vec = np.zeros((N,dim),dtype=complex);
		v_vec[1:N-1,:] = vec[N:2*N-2,:];
		h_vec = vec[2*N-2:3*N-2,:];
	elif BC == 'NO-SLIP':
		u_vec = np.zeros((N,dim),dtype=complex);
		u_vec[1:N-1,:] = vec[0:N-2,:];
		v_vec = np.zeros((N,dim),dtype=complex);
		v_vec[1:N-1,:] = vec[N:2*N-2,:];
		h_vec = vec[2*N-4:3*N-4,:];	
	else:
		sys.exit('ERROR: choose valid BC');

	return u_vec, v_vec, h_vec;

#====================================================

# vecs2vec
def vecs2vec(u_vec,v_vec,h_vec,N,dim,BC):
# The inverse operation of the above function.

	vec = np.zeros((dim,dim),dtype=complex);

	if BC == 'FREE-SLIP':
		vec[0:N,:] = u_vec;		
		vec[N:2*N-2,:] = v_vec[1:N-1,:];
		vec[2*N-2:3*N-2,:] = h_vec;
	elif BC == 'NO-SLIP':
		vec[0:N-2,:] = u_vec[1:N-1,:];
		vec[N:2*N-2,:] = v_vec[1:N-1,:];
		vec[2*N-4:3*N-4,:] = h_vec;	
	else:
		sys.exit('ERROR: choose valid BC');

	return vec;

#====================================================

# diff
def diff(f,d,p,delta):
# Function for differentiating a vector
# f[y,x] is the function to be differentiated.
# d is the direction in which the differentiation is to be taken:
# d=0 for differentiation over the first index, d=1 for the second.
# d=2 for a 1D vector
# p is a periodic switch:
# p=1 calculates periodic derivative.
# Need to be careful with periodic derivatives, output depends on whether f[0]=f[dim-1] or =f[dim].
	
	if d != 2:
		dimx = np.shape(f)[1];		# Finds the number of gridpoints in the x and y directions
		dimy = np.shape(f)[0];
		df = np.zeros((dimy,dimx),dtype=f.dtype);
	else:
		dimy = np.shape(f)[0];
		df = np.zeros(dimy,dtype=f.dtype);
	
	if p == 0:
		# Solid boundary derivative.
		# Note multiplication by 2 of boundary terms are to
		# invert the division by 2 at the end of the module.
		if d == 0:
		
			df[1:dimy-1,:] = f[2:dimy,:] - f[0:dimy-2,:];
		
			df[0,:] = 2 * (f[1,:] - f[0,:]);
			df[dimy-1,:] = 2 * (f[dimy-1,:] - f[dimy-2,:]);
	
		elif d == 1:

			df[:,1:dimx-1] = f[:,2:dimx] - f[:,0:dimx-2];

			df[:,0] = 2 * (f[:,1] - f[:,0]);
			df[:,dimx-1] = 2 * (f[:,0] - f[:,dimx-2]);
		
		elif d == 2:

			df[1:dimy-1] = f[2:dimy] - f[0:dimy-2];

			df[0] = 2 * (f[1] - f[0]);
			df[dimy-1] = 2 * (f[dimy-1] - f[dimy-2]);

		else:
			print('error')

	elif p == 1:
		# Periodic option

		if d == 0:

			df[1:dimy-1,:] = f[2:dimy,:] - f[0:dimy-2,:];	

			df[0,:] = f[1,:] - f[dimy-1,:];
			df[dimy-1,:] = f[0,:] - f[dimy-2,:];
	
		elif d == 1:

			df[:,1:dimx-1] = f[:,2:dimx]-f[:,0:dimx-2];

			df[:,0] = f[:,1] - f[:,dimx-1];
			df[:,dimx-1] = f[:,0] - f[:,dimx-2];

		elif d == 2:

			df[1:dimy-1] = f[2:dimy] - f[0:dimy-2];

			df[0] = f[1] - f[dimy-2];
			df[dimy-1] = f[1] - f[dimy-2];
			
			#print(str(df[0])+'='+str(f[1])+'+'+str(f[dimy-1]));
			#print(str(df[dimy-1])+'='+str(f[0])+'+'+str(f[dimy-2]));
		
		else:
			print('error')

	else:
		print('error')

	df = 0.5 * df / delta;

	return df

#====================================================

# mode
def mode(A):
# Mode of an array A

	# List A
	A = list(A)
	
	# Set A
	S = set(A)

	# Length of A
	N = len(S)
	
	# Initialise tally
	tally = np.zeros(N)
	for i in range(0,N):
		s = list(S)[i]
		tally[i] = A.count(s)

	# Index of mode
	mode_i = np.argsort(-tally)[0]

	# Mode
	Mode = list(S)[mode_i]

	return Mode	

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


#====================================================

def switchKsign(sol,N):
	'''Switch the sign convention of zonal wavenumbers in solution'''

	sol_new = sol.copy()

	for i in range(1,N//2):
		sol_new[:,i] = sol[:,N-i]
		sol_new[:,N-i] = sol[:,i]

	#for i in range(1,N//2):
	#	sol_new[i] = sol[N-i]
	#	sol_new[N-i] = sol[i]

	return sol_new


	
	
