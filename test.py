import numpy as np
temps = np.arange(0.05, 1.05, 0.1)
temps = [round(float(t), 2) for t in temps]

top_ps = np.arange(0, 1.05, 0.1)
top_ps = [round(float(t), 2) for t in top_ps]

top_ks = [1, 2, 5, 10, 20, 50, 100, 200, 500]

print(temps)
print(top_ps)
print(top_ks)