import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

with open('reward_count1.txt') as f:
    reward_counts = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
reward_count = [float(x.strip('\n')) for x in reward_counts]
# smooth plot line
N = 10
cumsum, moving_aves = [0], []
for i, x in enumerate(reward_count, 1):
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        #can do stuff with moving_ave here
        moving_aves.append(float(moving_ave))
index = [float(x+1) for x in range(len(moving_aves))]
xnew = np.linspace(float(1), float(len(moving_aves)), 1000)
smoothed_line = spline(index, moving_aves, xnew)
plt.plot(xnew, smoothed_line)
plt.show()