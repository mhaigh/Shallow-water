# Gold_forcing

import numpy as np
import matplotlib.pyplot as plt
import diagnostics

OPT = 3;

N = 256;
y = np.linspace(1,N,N);
x = np.linspace(1,N,N);
dy = y[1] - y[0];
dx = x[1] - x[0];
Lx = x[N-1] - x[0];
Amp = 1.0;

j2 = int((y[N-1] - y[0]) / 2) + int(y[0]);
jSo = int((j2 - y[0]) / 2) + int(y[0]);
jNo = int((y[N-1] - j2)/ 2) + j2;


f = np.zeros(N);

#==========================================================================



if OPT == 1:

	
	N = 400;
	y = np.linspace(1,N,N);
	x = np.linspace(1,N,N);

	asym = 1.5
	tau = 0.08

	m = 0.0
	yf = y[N/2+1];	
	y_8 = y[N/8+1] - y[0]
	
	

	# The axis lies at m * x - y - yf = 0.

	taux = np.zeros((N,N));
	tauy = np.zeros((N,N));
	h1 = 300.0 * np.ones((N,N));
	for j in range(0,N):
		Pr = asym * j / N
		for i in range(0,N):
			if (m * x[i] - y[j] + yf > 0):
				taux[j,i] = tau * (1.0 + np.cos(2.0 * np.pi *(m * x[i] - y[j] + yf) / ((1.+m)*N)));
			else:
				taux[j,i] = tau * (1.0 + np.cos(2.0 * np.pi *(m * x[i] - y[j] + yf) / ((1.+m)*asym*N)));
			tauy[j,i] = m * taux[j,i];
			if y[j] - yf - y_8 < m*x[i] < y[j] - yf + y_8:
				h1[j,i] = 300.0 + np.sin(1.0 * np.pi * (m * x[i] - y[j] - yf) / y_8);
	
	axis = np.zeros(N);	
	for i in range(0,N):
		axis[i] = np.argsort(-taux[:,i])[0];


		
	plt.plot(taux[:,0]);
	plt.show();

	#plt.contourf(h1);
	#plt.show();
	
	# Now wind stress curl.
	# The curl is [ nabla x (taux, tauy) ] = (tauy_x, -taux_y).
	taux_y = diagnostics.diff(taux,0,0,dy);
	tauy_x = diagnostics.diff(tauy,1,2,dx);
	
	Nv = 16;
	yv = np.linspace(0,N-1,Nv,dtype=int);
	xv = np.linspace(0,N-1,Nv,dtype=int);

	taux_vec = np.zeros((Nv,Nv));
	tauy_vec = np.zeros((Nv,Nv));
	taux_y_vec = np.zeros((Nv,Nv));
	tauy_x_vec = np.zeros((Nv,Nv));
	for j in range(0,Nv):
		for i in range(0,Nv):
			taux_vec[j,i] = taux[yv[j],xv[i]];
			#taux_vec[j,i] = 0.5 * (1.0 + np.cos(1.0 * 2.0 * np.pi *(m * x[xv[i]] - y[yv[j]] - yf) / N));
			tauy_vec[j,i] = m * taux_vec[j,i];
			
			# 
			taux_y_vec[j,i] = taux_y[yv[j],xv[i]];
			tauy_x_vec[j,i] = m*taux_y[yv[j],xv[i]];
			

	curl = tauy_x - taux_y

	plt.subplot(121);
	plt.contourf(x,y,taux_y);
	plt.colorbar();
	plt.subplot(122);
	plt.contourf(x,y,tauy_x);
	plt.colorbar();
	plt.show();

	#plt.subplot(121);	
	#plt.quiver(xv,yv,taux_vec,tauy_vec);
	#plt.subplot(122);
	#plt.contourf(taux);
	#plt.plot(axis);
	#plt.colorbar();
	#plt.show();

	cmax = np.max(np.absolute(curl))
	xgrid,ygrid = np.meshgrid(x,y)

	plt.subplot(121);	
	plt.pcolor(xgrid,ygrid,curl);
	plt.colorbar();
	plt.title('wind stress curl');
	plt.subplot(122);
	plt.quiver(xv,yv,taux_vec,tauy_vec);
	plt.title('wind stress');
	plt.show();

	plt.plot(curl[:,10]);
	plt.show();
	


if OPT == 2:

	
	N = 400;
	y = np.linspace(1,N,N);
	x = np.linspace(1,N,N);

	asym = 0.8
	tau = 0.08

	m = 0.0
	yf = y[N/2+1];	
	y_8 = y[N/8+1] - y[0]
	
	Pr = asym

	# The axis lies at m * x - y - yf = 0.

	taux = np.zeros((N,N));
	tauy = np.zeros((N,N));
	h1 = 300.0 * np.ones((N,N));
	for j in range(0,N):
		Pr = asym * j / N
		for i in range(0,N):
			taux[j,i] = tau * (1.0 + np.cos(2.0 * np.pi *(m * x[i] - y[j] + yf) / ((1.+m)*N)) + Pr);
			tauy[j,i] = m * taux[j,i];
			if y[j] - yf - y_8 < m*x[i] < y[j] - yf + y_8:
				h1[j,i] = 300.0 + np.sin(1.0 * np.pi * (m * x[i] - y[j] - yf) / y_8);
	
	axis = np.zeros(N);	
	for i in range(0,N):
		axis[i] = np.argsort(-taux[:,i])[0];


		
	plt.plot(taux[:,0]);
	plt.show();

	#plt.contourf(h1);
	#plt.show();
	
	# Now wind stress curl.
	# The curl is [ nabla x (taux, tauy) ] = (tauy_x, -taux_y).
	taux_y = diagnostics.diff(taux,0,0,dy);
	tauy_x = diagnostics.diff(tauy,1,2,dx);
	
	Nv = 16;
	yv = np.linspace(0,N-1,Nv,dtype=int);
	xv = np.linspace(0,N-1,Nv,dtype=int);

	taux_vec = np.zeros((Nv,Nv));
	tauy_vec = np.zeros((Nv,Nv));
	taux_y_vec = np.zeros((Nv,Nv));
	tauy_x_vec = np.zeros((Nv,Nv));
	for j in range(0,Nv):
		for i in range(0,Nv):
			taux_vec[j,i] = taux[yv[j],xv[i]];
			#taux_vec[j,i] = 0.5 * (1.0 + np.cos(1.0 * 2.0 * np.pi *(m * x[xv[i]] - y[yv[j]] - yf) / N));
			tauy_vec[j,i] = m * taux_vec[j,i];
			
			# 
			taux_y_vec[j,i] = taux_y[yv[j],xv[i]];
			tauy_x_vec[j,i] = m*taux_y[yv[j],xv[i]];
			

	curl = tauy_x - taux_y

	plt.subplot(121);
	plt.contourf(x,y,taux_y);
	plt.colorbar();
	plt.subplot(122);
	plt.contourf(x,y,tauy_x);
	plt.colorbar();
	plt.show();

	#plt.subplot(121);	
	#plt.quiver(xv,yv,taux_vec,tauy_vec);
	#plt.subplot(122);
	#plt.contourf(taux);
	#plt.plot(axis);
	#plt.colorbar();
	#plt.show();

	cmax = np.max(np.absolute(curl))
	xgrid,ygrid = np.meshgrid(x,y);

	plt.subplot(121);	
	plt.pcolor(xgrid,ygrid,curl,vmin=-cmax,vmax=cmax);
	plt.colorbar();
	plt.title('wind stress curl');
	plt.subplot(122);
	plt.quiver(xv,yv,taux_vec,tauy_vec);
	plt.title('wind stress');
	plt.show();

	plt.plot(curl[:,10]);
	plt.show();
	

# Igor's wind forcing
if OPT == 3:

	a = N-1;
	b = N-1;
	beta = 0.8;
	Amp = 1.0;
	asym = 1.0;

	m = 0.1
	yf = y[N/2+1];

	taux1 = np.zeros((N,N));
	tauy1 = np.zeros((N,N));
	taux2 = np.zeros((N,N));
	for j in range(0,N):
		for i in range(0,N):
			x0 = beta * y[j]
			Pr= (beta + Amp * (- beta + x0 / b)) * asym;			 	# The asymmetry of the wind
			taux1[j,i] = 0.5 * (1.0 + np.cos(2.0 * np.pi * (m * x[i] - y[j] + yf) / N) + asym*y[j]/N);
			tauy1[j,i] = m * taux1[j,i];
			taux2[j,i] = 0.5 * (1.0 - np.cos(2.0 * np.pi * y[j] / N));

	i1 = np.argsort(-taux1[:,0]);
	i2 = np.argsort(-taux2[:,0]);

	#plt.subplot(121)
	#plt.contourf(taux1);
	#plt.subplot(122)
	#plt.contourf(tauy1);
	#plt.show();

	taux1_y = diagnostics.diff(taux1,0,0,dy);
	taux2_y = diagnostics.diff(taux2,0,0,dy);
	tauy1_x = diagnostics.diff(tauy1,1,0,dx);
	#tauy_x = diagnostics.diff(tauy,1,2,dx);

	plt.plot(taux1_y[:,10]);
	plt.plot(taux2_y[:,10]);
	plt.show();
	
	curl = tauy1_x - taux1_y;
	cmax = 0.7*np.max(np.absolute(curl))
	xgrid,ygrid=np.meshgrid(x,y)
	
	
	plt.pcolor(xgrid,ygrid,curl,vmin=-cmax,vmax=cmax);
	plt.colorbar()	
	plt.title('Wind stress curl')
	plt.axis([xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()])
	plt.show();


	
	print('Buoyancy shift = ' + str(int(N / 32.0)))
	print('Wind shift = ' + str(i1[0] - i2[0]));


# Pavel's wind forcing
if OPT == 4:
	A = 0.9;
	B = 0.2;
	
	tau0 = 0.8;

	W = np.zeros((N,N));

	for j in range(0,N):
		for i in range(0,N):
			if y[j] <= B * x[i]:
				W[j,i] = - (np.pi * tau0 * A / N) * np.sin(np.pi * (N + y[j]) / (N + B * x[i]));
			else:
				W[j,i] = (np.pi * tau0 / (A * N)) * np.sin(np.pi * (y[j] - B * x[i]) / (N - B * x[i]));

	#taux_y = diagnostics.diff(taux,0,0,dy);
	#tauy_x = diagnostics.diff(tauy,1,2,dx);

	plt.contourf(W);
	plt.colorbar();
	plt.show();


# Tilted wind forcing and thickness relaxation
if OPT == 5:
	
	N = 400;
	y = np.linspace(1,N,N);
	x = np.linspace(1,N,N);

	a = N-1;
	b = N-1;

	m = 0.1
	yf = y[N/2+1];	
	y_8 = y[N/8+1] - y[0];
	
	theta = np.pi / 8.0;	

	taux = np.zeros((N,N));
	tauy = np.zeros((N,N));
	h1 = 300.0 * np.ones((N,N));
	for j in range(0,N):
		for i in range(0,N):
			taux[j,i] = 0.5 * (1.0 + np.cos(2.0 * np.pi *(m * x[i] - y[j] + yf) / (1.25*N)));
			tauy[j,i] = m * taux[j,i];
			if y[j] - yf - y_8 < m*x[i] < y[j] - yf + y_8:
				h1[j,i] = 300.0 + np.sin(1.0 * np.pi * (m * x[i] - y[j] - yf) / y_8);
	
	axis = np.zeros(N);	
	for i in range(0,N):
		axis[i] = np.argsort(-taux[:,i])[0];
		
	#plt.plot(taux[:,0]);
	#plt.show();

	#plt.contourf(h1);
	#plt.show();
	
	# Now wind stress curl.
	# The curl is [ nabla x (taux, tauy) ] = (tauy_x, -taux_y).
	taux_y = diagnostics.diff(taux,0,0,dy);
	tauy_x = diagnostics.diff(tauy,1,2,dx);
	
	Nv = 16;
	yv = np.linspace(0,N-1,Nv,dtype=int);
	xv = np.linspace(0,N-1,Nv,dtype=int);

	taux_vec = np.zeros((Nv,Nv));
	tauy_vec = np.zeros((Nv,Nv));
	taux_y_vec = np.zeros((Nv,Nv));
	tauy_x_vec = np.zeros((Nv,Nv));
	for j in range(0,Nv):
		for i in range(0,Nv):
			taux_vec[j,i] = taux[yv[j],xv[i]];
			#taux_vec[j,i] = 0.5 * (1.0 + np.cos(1.0 * 2.0 * np.pi *(m * x[xv[i]] - y[yv[j]] - yf) / N));
			tauy_vec[j,i] = m * taux_vec[j,i];
			
			# 
			taux_y_vec[j,i] = taux_y[yv[j],xv[i]];
			tauy_x_vec[j,i] = m*taux_y[yv[j],xv[i]];



	plt.subplot(121);
	plt.contourf(x,y,taux_y);
	plt.colorbar();
	plt.subplot(122);
	plt.contourf(x,y,tauy_x);
	plt.colorbar();
	plt.show();

	plt.subplot(121);	
	plt.quiver(xv,yv,taux_vec,tauy_vec);
	plt.subplot(122);
	plt.contourf(taux);
	plt.plot(axis);
	plt.colorbar();
	plt.show();

	plt.subplot(121);	
	plt.contourf(xv,yv,tauy_x_vec-taux_y_vec);
	plt.colorbar();
	plt.title('wind stress curl');
	plt.subplot(122);
	plt.quiver(xv,yv,taux_vec,tauy_vec);
	plt.title('wind stress');
	plt.show();
	

if OPT == 6:
	
	N = 20;
	y = np.linspace(1,N,N);
	x = np.linspace(1,N,N);

	a = N-1;
	b = N-1;
	beta = 0.8;
	Amp = 1.0;
	wind_asym1 = 1.5;
	
	theta = np.pi / 8;	

	taux = np.zeros((N,N));
	for j in range(0,N):
		x0 = beta * y[j]
		Pr= (beta + Amp * (- beta + x0 / b)) * wind_asym1;			 	# The asymmetry of the wind
		taux[j,:] = 0.5 * (1.0 - np.cos(2.0 * np.pi * y[j] / N) + Pr);
		
	Rtaux = np.zeros((N,N));
	Rtauy = np.zeros((N,N));
	for j in range(0,N):
		Rtaux[j,:] = np.cos(theta) * taux[j,:];
		Rtauy[j,:] = np.sin(theta) * taux[j,:];

	plt.subplot(121);
	plt.contourf(Rtaux);
	plt.subplot(122);	
	plt.contourf(Rtauy);
	plt.show();
	
	plt.subplot(121);
	plt.quiver(x,y,taux,np.zeros((N,N)));
	plt.subplot(122);
	plt.quiver(x,y,Rtaux,Rtauy);
	plt.show();

if OPT == 7:
	N = 512;
	nj = N;

	js = 0 ; je = N;
	i1 = 0 ; ie = N;

  	j_50 = int((je - js) / 2) + js ;
  	j_25 = int((j_50 - js) / 2) + js ; j_75 = int((je - j_50) / 2) + j_50
  	j_12 = int((j_25 - js) / 2) + js ; j_37 = int((j_50 - j_25) / 2) + j_25
  	j_62 = int((j_75 - j_50) / 2) + j_50 ; j_87 = int((je - j_75) / 2) + j_75

	dh = 100.0;

	m = 0.1;
	yf = j_50 - m * i1;
	L = j_62-j_37;
	pi = 3.14159265;

	h1_restore = np.zeros((N,N));
	sine1 = np.zeros(N);
	sine2 = np.zeros(N);
	h1_restore[:,:] = 300.0 
	for i in range(i1,ie):
		for j in range(js,je):
			if (m * i + yf - 0.5 * L < j and j < m * i + yf + 0.5 * L):
				xy = 2.0 * np.pi * (m * i - j + yf) / L; 
				#sine1 = xy - xy**3 / 6.0 + xy**5 / 120.0 - xy**7 / 5040.0 + xy**9 / 362880.0 - xy**11 / 39916800.0 + xy**13 / 6227020800.0 - xy**15 / 1307674368000 + xy**17 / 3.55687428e14 - xy**19 / 1.216451e17
				sine2 = np.sin(xy);				
				h1_restore[j,i] = 300.0 + dh * sine2

	
	plt.contourf(h1_restore);	
	plt.colorbar();
	plt.show();







	
