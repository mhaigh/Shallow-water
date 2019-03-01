# diags.py

# Functions for use in parallel modules.

import os
import numpy as np

#=========================================================

def splitDomain(test_set,Nu,pe):
	'''Split test_set (testing domain) into pe sets.'''

	sets = []

	d = Nu // pe + 1	  	# Number of elements in each set (apart from the last one).
	r = Nu % d

	for pe_no in range(1,pe):
		sets.append(test_set[(pe_no-1)*d:pe_no*d])

	for pe_no in range(pe,pe+1):
		sets.append(test_set[(pe_no-1)*d:(pe_no-1)*d+r])

	return sets

#=========================================================

def buildArray(filename,Nu,pe):
	'''Rebuild array from parallel outputs. First index should be Nu, length of test_set.'''

	# Read first temporary file
	fname = filename+'0.npy'
	array_tmp = np.load(fname)
	os.remove(fname)

	# Determine dimension of array	
	shape = np.shape(array_tmp)
	yn = shape[0]
	
	ndim = len(shape)
	if ndim == 1:
		array = np.zeros((Nu), dtype = array_tmp.dtype)
	elif ndim == 2:
		array = np.zeros((Nu,shape[1]), dtype = array_tmp.dtype)
	elif ndim == 3:
		array = np.zeros((Nu,shape[1],shape[2]), dtype = array_tmp.dtype)
	
	array[0:yn,] = array_tmp

	yn_count = yn
	# Read remaining temporary files 
	for pi in range(1,pe):
		fname = filename+str(pi)+'.npy'
		array_tmp = np.load(fname)
		yn = np.shape(array_tmp)[0]
		array[yn_count:yn_count+yn,] = array_tmp
		yn_count += yn
		os.remove(fname)

	np.save(filename,array)		

#=========================================================

