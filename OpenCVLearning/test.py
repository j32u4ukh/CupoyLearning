import numpy as np

x = np.random.randn(3, 2, 2, 2)
print(x)
print("==========")

np.random.shuffle(x)
print(x)
