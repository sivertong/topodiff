import scipy.io as io
import numpy as np
io.savemat('testDataMAT.mat', mdict=np.load('D:\CodeSave\GitCode\\topodiff\scripts\generated\samples_200x11x1.npz'))
