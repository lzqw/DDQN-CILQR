import numpy as np
a=np.tile(np.array([1., 1.,10, 10]), (1, 10 + 1))[0]
print(a.shape)
print(a)