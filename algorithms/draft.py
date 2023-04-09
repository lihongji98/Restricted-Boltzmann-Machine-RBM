import numpy as np
num = np.array([[i] for i in range(512)])
num1 = np.repeat(num, 14, axis = 1)
phx = np.random.normal(-0.1, 0.1, size = (512,20))
a = np.dot(num1.T, phx)

b = np.dot(num.T, phx)
b = np.repeat(b, 14, axis = 0)
#print(a.shape, b.shape)

phx = np.random.normal(-0.1, 0.1, size = (14, 9, 20))
s = np.sum(phx, axis = 0)
print(s.shape)

# write a RBM algorithm
