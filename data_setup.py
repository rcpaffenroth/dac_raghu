import glob
import os
import numpy as np

all_files = glob.glob(os.path.join("data/lti_1d_trajectories", "*.csv"))

data = []
for file in all_files:
    data.append(np.loadtxt(file))
data_1d = np.array(data)

all_files = glob.glob(os.path.join("data/lti_2d_trajectories", "*.csv"))

data = []
for file in all_files:
    data.append(np.loadtxt(file))
data_2d = np.array(data)

np.savetxt('data_1d.txt', data_1d, delimiter=',')
np.savetxt('data_2d.txt', data_2d, delimiter=',') 

