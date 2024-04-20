import numpy as np
import scipy.io

# Load the .mat file
mat = scipy.io.loadmat('results.mat')

# Check what variables are in the .mat file
#print(mat.keys())

rssi_values_list = np.array(np.double(mat['RSSI'][0]))

for index, value in enumerate(rssi_values_list, start=1):
    print(f'RSSI{index}: {value}')
