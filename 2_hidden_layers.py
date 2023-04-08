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

plt.scatter(X[:, 0], X[:, 1])
plt.show()

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
weights1 = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases1 = [2.0, 3.0, 0.5]
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]


layer_outputs1 = np.dot(inputs, np.array(weights1).T) + biases1
layer_outputs2 = np.dot(layer_outputs1, np.array(weights2).T) + biases2
    
print("Dynamic outputs: ", layer_outputs2)


