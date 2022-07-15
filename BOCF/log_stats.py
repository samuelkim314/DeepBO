import numpy as np
import matplotlib.pyplot as plt

n_files = 50
path = './experiments_cluster/test1_EIh'
name = '/test1_EIh_noiseless'

n_iterations = 50
log_regret = np.zeros((n_iterations,n_files))
time  = np.zeros((n_iterations,n_files))
real_opt = 0.#3.32236801#-0.42973173728582004
for i in range(n_files):
    log_regret[:, i] = np.log10(real_opt - np.loadtxt(path + name+str(i)+'.txt', unpack=True)[0, 0:n_iterations])
    time[:, i] = np.loadtxt(path + name + str(i) + '.txt', unpack=True)[1, 0:n_iterations]/60

log_regret_stats = np.zeros((n_iterations, 2))
log_regret_stats[:, 0] = np.mean(log_regret, axis=1)
log_regret_stats[:, 1] = np.std(log_regret, axis=1)

time_stats = np.zeros((n_iterations, 2))
time_stats[:, 0] = np.mean(time, axis=1)
time_stats[:, 1] = np.std(time, axis=1)
    
np.savetxt(path + name + '_log_regret_stats.txt', log_regret_stats)
np.savetxt(path + name + '_time_tats.txt', time_stats)
plt.figure()
plt.plot(log_regret_stats[:, 0], label=name)
plt.plot(log_regret_stats[:, 0] + log_regret_stats[:, 1])
plt.plot(log_regret_stats[:, 0] - log_regret_stats[:, 1])
plt.figure()
plt.plot(time_stats[:, 0], label=name)
plt.plot(time_stats[:, 0] + time_stats[:, 1])
plt.plot(time_stats[:, 0] - time_stats[:, 1])
plt.show()