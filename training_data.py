# benedikt zimmermann
# my nn from scratch

# Test git upload to git

# a single neuron with numpy

import numpy as np
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import nnfs
nnfs.init()

X, y = spiral_data(samples=100, classes=3)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()


