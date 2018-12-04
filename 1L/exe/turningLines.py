# turningLines.py

#=================================================================

import numpy as np
import matplotlib.pyplot as plt

from inputFile import *

#=================================================================

# Take U0 to be Gaussian, first find its second derivative.
Uy = diff(U0,2,0,dy)
Uyy = diff(Uy,2,0,dy)

# Pick a zonal wavenumber and define its zonal phase speed
k = 2. * np.pi * 5. / L
c = omega / k

print(c)


