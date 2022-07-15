import numpy as np
import matplotlib.pyplot as plt

n_files = 50
#path = './experiments/test1'
#name = '/random_noisy/test1_random_noisy'
path = './experiments/test3'
name = '/uKG_cf_noiseless/test3_uKG_cf'

n_iterations = 100
data = np.zeros((n_iterations,n_files))

for i in range(n_files):
    data[:,i] = np.loadtxt(path + name+str(i+1)+'.txt', unpack=True)
    #print(average.shape)
    #print(aux.shape)

data_stats = np.zeros((n_iterations, 2))
data_stats[:, 0] = np.mean(data, axis=1)
data_stats[:, 1] = np.std(data, axis=1)
    
np.savetxt(path + name + '_stats.txt', data_stats)
plt.plot(data_stats[:, 0], label=name)
plt.plot(data_stats[:, 0] + data_stats[:, 1])
plt.plot(data_stats[:, 0] - data_stats[:, 1])
plt.show()
