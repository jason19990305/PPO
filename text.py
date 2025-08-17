import numpy as np

s_ = np.zeros((8, 3))
print(s_)
row = np.array([1,2,3])
print(s_[0, :].shape)
print(row.shape)
for j in range(8):
    s_[j] = row + j
print(s_)