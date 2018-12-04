# FFT test
#========================================

import numpy as np
import matplotlib.pyplot as plt

#========================================

# test = 1 for Gaussian, 2 for cosine, 3 for one-sided decaying exponential, 4 for derivative of Gaussian

test = 7;

I = np.complex(0,1);

Lx = 1.;
N = 128;
a = 1000;

x = np.linspace(-Lx/2,Lx/2,N);
dx = x[1]-x[0];

K = np.fft.fftfreq(N,Lx/N);			# Zero freq at index 0
Kshifted = np.fft.fftshift(K);		# Zero freq in the middle

if test == 1:
	y1 = np.zeros(N);
	ytilde1 = np.zeros(N);

	for i in range(0,N):
		y1[i] = np.exp(- a * x[i]**2);

	FFTy1 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(y1),axis=0));

	for i in range(0,N):
		ytilde1[i] = np.sqrt(np.pi / a) * np.exp(- np.pi**2 * Kshifted[i]**2 / (a));

	IFFTytilde1 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(ytilde1),axis=0));	

	IFFTFFTy1 = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(FFTy1),axis=0));

	# The FFT is a factor of dx greater than the analytic FT. WHY???
	
	error1a = ytilde1 - dx * FFTy1;
	error1b = IFFTytilde1 / dx - y1;
	error1c = IFFTFFTy1 - y1;

	plt.figure(1);

	plt.subplot(131)
	plt.plot(Kshifted,ytilde1);

	plt.subplot(132);
	plt.plot(Kshifted,dx*FFTy1);

	plt.subplot(133);
	plt.plot(Kshifted,error1a);

	plt.show()

	# Here the ytilde1 = FFTy1 / (N/2)

	plt.figure(2);

	plt.subplot(231);
	plt.plot(x,y1);

	plt.subplot(232);
	plt.plot(x,IFFTytilde1/dx);

	plt.subplot(233);
	plt.plot(x,IFFTFFTy1);

	plt.subplot(235);
	plt.plot(x,error1b);

	plt.subplot(236);
	plt.plot(x,error1c);

	plt.show()

	# And here y1 = IFFTytilde1 * N/2

	# CONCLUSIONS ABOUT THE SCALING...

	# 1. Difference between y1 and FFTy1 is a factor of dx i.e. y1 = FFTy1 / dx

#========================================


if test == 2:

	y2 = np.zeros(N);

	for i in range(0,N):
		y2[i] = np.cos(5 * np.pi * x[i]);

	FFTy2 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(y2),axis=0));
	IFFTFFTy2 = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(FFTy2),axis=0));
	NIFFTy2 = N * np.fft.fftshift(np.fft.ifft(np.fft.fftshift(y2),axis=0));
	
	plt.figure(1)

	plt.subplot(221)
	plt.plot(x,y2);

	plt.subplot(222);
	plt.plot(x,IFFTFFTy2);

	plt.subplot(223);
	plt.plot(Kshifted,FFTy2);

	plt.subplot(224);
	plt.plot(Kshifted,NIFFTy2);

	plt.show()

if test == 3:
	y3 = np.zeros(N);
	
	for i in range(N/2,N):
		y3[i] = np.exp(- x[i]);

	FFTy3 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(y3),axis=0)) - 0.5 * dx;
	
	ytilde3 = np.zeros((N),dtype=complex);
	for i in range(0,N):
		ytilde3[i] = 1 / (2 * np.pi * I * Kshifted[i] + 1);
	

	error3a = np.real(ytilde3) - dx*np.real(FFTy3);
	error3b = np.imag(ytilde3) - dx*np.imag(FFTy3);

	plt.figure(1);
	
	plt.subplot(231);
	plt.plot(Kshifted,np.real(ytilde3));

	plt.subplot(232);
	plt.plot(Kshifted,dx*np.real(FFTy3));

	plt.subplot(233);
	plt.plot(Kshifted,error3a);

	plt.subplot(234);
	plt.plot(Kshifted,np.imag(ytilde3));
	
	plt.subplot(235);
	plt.plot(Kshifted,dx*np.imag(FFTy3));

	plt.subplot(236);
	plt.plot(Kshifted,error3b);
	
	plt.show()

# Conclude that the DFT is a differs by the integral Fourier transform IFT by a factor of dx
# IFT = dx * AFT.	

if test == 4:
	x1 = np.linspace(-Lx/2,Lx/2,N);
	x2 = np.linspace(-Lx/2,Lx/2,N);
	y4 = np.zeros((N,N));
	for j in range(0,N):
		for i in range(0,N):
			y4[j,i] = np.cos(4 * 2 * np.pi * x2[j]) + np.cos(2 * 2 * np.pi * x1[i]);

	Fx_y4 = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(y4,axes=1),axis=1),axes=1);
	Fy_y4 = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(y4,axes=0),axis=0),axes=0);

	plt.figure(1)
	plt.subplot(131)
	plt.contourf(x2,x1,y4)
	plt.colorbar()
	plt.subplot(132)
	plt.contourf(x2,Kshifted,Fx_y4)
	plt.subplot(133)
	plt.contourf(Kshifted,x1,Fy_y4)
	plt.show()

# Test on derivatives of FFT. Does there need to be a factor of 2 pi?
if test == 5:

	y = np.zeros(N);
	y_x = np.zeros(N);
	ytilde = np.zeros(N);

	for i in range(0,N):
		y[i] = np.exp(- a * x[i]**2);
		y_x[i] = - 2 * a * x[i] * y[i];

	FFTy = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(y)));
	FFTy_x = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(y_x)));
	IkFFTy = 2 * np.pi * I * Kshifted * FFTy;

	plt.figure(1,figsize=[21,6]);
	plt.subplot(131);
	plt.plot(Kshifted,FFTy_x);
	plt.subplot(132);
	plt.plot(Kshifted,IkFFTy);
	plt.subplot(133);
	plt.plot(Kshifted,FFTy_x-IkFFTy);
	plt.show();
	
	# Shows that derivatives do need a factor of 2 pi

if test == 6:
	L = 100.;
	N = 512;
	x = np.linspace(-L/2,L/2,N+1);
	dx = x[1]-x[0];
	
	a = L/20;

	K = np.fft.fftfreq(N,dx);			# Zero freq at index 0
	Kshifted = np.fft.fftshift(K);		# Zero freq in the middle
	print K
	f1 = np.zeros(N+1);
	f2 = np.zeros(N+1);

	for i in range(0,N+1):
		if x[i] >= - a/2 and x[i] <= a/2:
			f1[i] = 1;
			f2[N-i] = 1;

	#plt.plot(x,f1-f2); # To check func is even
	#plt.show();
	
	F1 = dx * np.fft.fftshift(np.fft.fft(np.fft.fftshift(f1),N));

	F1_analytic = np.zeros(N);
	for i in range(0,N):
		F1_analytic[i] = a * np.sinc(a*Kshifted[i]);

	print(1./a);


	plt.figure(1)
	plt.subplot(221);
	plt.plot(Kshifted,np.real(F1));
	plt.subplot(222);
	plt.plot(Kshifted,np.imag(F1));
	plt.subplot(223);
	plt.plot(Kshifted,F1_analytic);
	plt.subplot(224);
	plt.plot(Kshifted,F1_analytic-np.real(F1));
	plt.show();

	F12 = dx * np.fft.fftshift(np.fft.hfft(np.fft.fftshift(f1),N));

	plt.figure(2)
	plt.subplot(221);
	plt.plot(Kshifted,F12);
	plt.subplot(222);
	plt.plot(Kshifted,np.real(F1)-F12);
	plt.subplot(223);
	plt.plot(Kshifted,F1_analytic);
	plt.subplot(224);
	plt.plot(Kshifted,F1_analytic-F12);
	plt.show();

	
	f1i = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(F1_analytic),n=N+1)) / dx;	
	print f1i.shape
	plt.plot(x,f1i[:]);
	plt.plot(x,f1);
	plt.show();

# Conclude from 6: no factor of pi needed.


if test == 7:

	N = 256
	F1 = np.zeros((N,N));
	Lx = 3840000.		# Zonal lengthscale (m)
	Ly = 3840000.		# Meridional lengthscale (m)
	Hflat = 4000.		# Motionless ocean depth (i.e. without BG flow SSH adjustment) (m)		
	L=Lx;	

	y = np.linspace(-Ly/2,Ly/2,N);    # Array of all gridpoints in physical space
	x = np.linspace(-Lx/2,Lx/2,N+1);
	x_nd = x / L;
	y_nd = y / L;
	K = np.fft.fftfreq(N,Lx/N);
	K_nd = K * L;

	dx = x[1] - x[0];
	dy = y[1] - y[0];
	dx_nd = dx / L;
	dy_nd = dy / L;	
	
	r0 = 120 * 1000;
	g = 9.81;
	AmpF = 1;
	f0 = 0.0001214; #0.83e-4;      		# Base value of Coriolis parameter (s-1)
	beta = 2e-11;     		# Planetary vorticity gradient (m-1 s-1)
	f = f0 + beta * y;      # Coriolis frequency (s-1)
	y0 = 0;
	U = 0.1;
	

	F1 = np.zeros((N,N+1));
	F2 = np.zeros((N,N));
	F3 = np.zeros((N,N));

	count = 0;
	mass = 0;
	for i in range(0,N):
		for j in range(0,N):
			r = np.sqrt(x[i]**2 + (y[j]-y0)**2);
			if r < r0:
				count = count + 1;
				F1[j,i] = AmpF * np.pi * g * (y[j]-y0) / (2 * r0 * f[j] * r) * np.sin((np.pi / 2) * r / r0);
				F2[j,i] = - AmpF * np.pi * g * x[i] / (2 * r0 * f[j] * r) * np.sin((np.pi / 2) * r / r0);
				F3[j,i] = AmpF * np.cos((np.pi / 2) * r / r0);
				mass = mass + F3[j,i];
	mass = mass / (N**2 - count);
	for i in range(0,N):
		for j in range(0,N):
			r = np.sqrt(x[i]**2 + (y[j]-y0)**2);
			if r >= r0:
				F3[j,i] = - mass;
	
	F3scale = g / (f0 * U**2);
	# Non-dimensionalise the forcings. Scale F1 and F2 by f0*U and scale F3 by g/(f0*U**2). 
	F1_nd = F1 / (f0 * U);
	F2_nd = F2 / (f0 * U);
	F3_nd = F3 * F3scale; 
	# Note that for normalising the solutions, we can use either of the scaling options. We will use f0*U.
	
	# Lastly, Fourier transform the three forcings in the x-direction
		
	Ftilde1 = dx * np.fft.hfft(np.fft.ifftshift(F1,axes=1),N,axis=1);	# Multiply by dx_nd as FFT differs by this factor compared to FT.
	Ftilde3 = dx * np.fft.hfft(np.fft.ifftshift(F3,axes=1),N,axis=1); 
	print Ftilde1.shape
	Ftilde2 = np.zeros((N,N),dtype=complex);
	for j in range(0,N):
		for i in range(0,N):
			Ftilde2[j,i] = 2 * np.pi * g * I * K[i] * Ftilde3[j,i] / f[j]; 
	#Ftilde2 = dx * np.fft.fft(np.fft.ifftshift(F2,axes=1),axis=1);
	# Note that F3_tilde is real and even in x, so we apply HFFT, to reduce numerical noise and ensure that its Fourier transform is also real and even.

	
	plt.figure(1);

	plt.subplot(331);
	plt.contourf(F1);
	plt.colorbar();
	plt.subplot(332);
	plt.contourf(F2);
	plt.colorbar();
	plt.subplot(333);
	plt.contourf(F3);
	plt.colorbar()

	plt.subplot(334);
	plt.contourf(np.real(Ftilde1));
	plt.colorbar()
	plt.subplot(335);
	plt.contourf(np.real(Ftilde2));
	plt.colorbar()
	plt.subplot(336);
	plt.contourf(np.real(Ftilde3));
	plt.colorbar()

	plt.subplot(337);
	plt.contourf(np.imag(Ftilde1));
	plt.colorbar()
	plt.subplot(338);
	plt.contourf(np.imag(Ftilde2));
	plt.colorbar()
	plt.subplot(339);
	plt.contourf(np.imag(Ftilde3));
	plt.colorbar()

	plt.show();

	F1i = np.fft.ihfft(Ftilde1,axis=1) / dx;
	F2 = np.fft.fftshift(np.fft.ifft(Ftilde2,axis=1),axes=1) / dx;
	F3i = np.fft.ihfft(Ftilde3,axis=1) / dx;

	plt.contourf(F1i)
	plt.show()
	
	for i in range(0,N/2):
		F1[:,i] = F1i[:,i];
		F1[:,N-1-i] = F1i[:,i]; 
		F3[:,i] = F3i[:,i];
		F3[:,N-1-i] = F3i[:,i];

	F1 = np.fft.fftshift(F1,axes=1);
	F3 = np.fft.fftshift(F3,axes=1);
	

	plt.figure(2);

	plt.subplot(331);
	plt.contourf(x,y,F1);
	plt.colorbar();
	plt.subplot(332);
	plt.contourf(x,y,F2);
	plt.colorbar();
	plt.subplot(333);
	plt.contourf(x,y,F3);
	plt.colorbar()

	plt.subplot(334);
	plt.contourf(np.real(Ftilde1));
	plt.colorbar()
	plt.subplot(335);
	plt.contourf(np.real(Ftilde2));
	plt.colorbar()
	plt.subplot(336);
	plt.contourf(np.real(Ftilde3));
	plt.colorbar()

	plt.subplot(337);
	plt.contourf(np.imag(Ftilde1));
	plt.colorbar()
	plt.subplot(338);
	plt.contourf(np.imag(Ftilde2));
	plt.colorbar()
	plt.subplot(339);
	plt.contourf(np.imag(Ftilde3));
	plt.colorbar()

	plt.show();





