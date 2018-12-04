#pp_test.py
#==========================================================================

# A test function for parallel python

#==========================================================================

import numpy as np
import matplotlib.pyplot as plt

import sys
import pp

def make_h1(m,yf,L,dh,pi,N,i1,ie,js,je):
	
	#np = numpy;

	h1_restore = np.zeros((N,N));
	sine1 = np.zeros(N);
	sine2 = np.zeros(N);
	h1_restore[:,:] = 300.0 
	for i in range(i1,ie):
		for j in range(js,je):
			if (m * i + yf - 0.5 * L < j and j < m * i + yf + 0.5 * L):
				xy = 2.0 * np.pi * (m * i - j + yf) / L; 
				sine = np.sin(xy);				
				h1_restore[j,i] = 300.0 + dh * sine

	return h1_restore

N = 512;
nj = N;
js = 0 ; je = N;
i1 = 0 ; ie = N;

j_50 = int((je - js) / 2) + js ;
j_25 = int((j_50 - js) / 2) + js ; j_75 = int((je - j_50) / 2) + j_50
j_12 = int((j_25 - js) / 2) + js ; j_37 = int((j_50 - j_25) / 2) + j_25
j_62 = int((j_75 - j_50) / 2) + j_50 ; j_87 = int((je - j_75) / 2) + j_75

dh = 100.0;

m = 0;
yf = j_50 - m * i1;
L = j_62-j_37;
pi = 3.14159265;

#==========================================================================

ppservers = ();

if len(sys.argv) > 1:
	ncpus = int(sys.argv[1])
	# Creates jobserver with ncpus workers
	job_server = pp.Server(ncpus, ppservers=ppservers)
	print('1');
else:
	# Creates jobserver with automatically detected number of workers
	job_server = pp.Server(ppservers=ppservers)

print(job_server.get_ncpus());

#==========================================================================

job1 = job_server.submit(make_h1, (m,yf,L,dh,pi,N,i1,ie,js,je,), (), ("numpy as np",));
	
h1_restore = job1();

plt.contourf(h1_restore);	
plt.colorbar();
plt.show();
