import numpy as np

A = [[1,2],[4,3]];

val,vec = np.linalg.eig(A);

print(val)
print(vec)
print(vec[0,:]);

l1 = val[0];
print (1. * vec[0,0] + 2. * vec[1,0] - l1 * vec[0,0])

# So in the output of eigenvectors, the second index corresponds to the eigenvalue, first index gives the entry within the eigenvector.
