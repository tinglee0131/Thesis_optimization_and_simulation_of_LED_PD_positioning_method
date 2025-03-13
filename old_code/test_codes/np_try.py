import numpy as np
import matplotlib.pyplot as plt
nrm1 = np.array([[0, 0, 1], [0, 0, 1]])
print(nrm1)
print(nrm1.shape)
nrm2 = np.array([[[1, 2, 3, 1, 1, 1], [4, 5, 6, 1, 1, 1], [7, 8, 9, 1, 1, 1]],
                 [[11, 12, 13, 1, 1, 1], [14, 15, 16, 1, 1, 1], [17, 18, 19, 1, 1, 1]]])
print(nrm2)
print(nrm2.shape)
print(nrm2[1, 1:, :])
print(np.size(nrm1), np.size(nrm2))



# 2D plot
def f(x):
    return x**2*np.exp(-x**2)
x = np.linspace ( start = 0.    # lower limit
                , stop = 3      # upper limit
                , num = 51      # generate 51 points between 0 and 3
                )
y = f(x)    # This is already vectorized, that is, y will be a vector!
plt.plot(x, y)
plt.show()