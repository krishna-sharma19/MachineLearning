from random import *
import numpy as np

x1 = [random() for _ in range(10000000)]
x2 = [2*random() for _ in range(10000000)]

x =[]
for x11,x22 in zip(x1,x2):
    x.append(max(x11,x22))

aa = np.stack((x, x1), axis=0)
print(np.cov(aa))