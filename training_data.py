# benedikt zimmermann
# my nn from scratch

# Test git upload to git

# a single neuron with numpy
from nn_classes import *
import numpy as np
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import nnfs
nnfs.init()
# Create dataset
X, y = spiral_data(samples=100, classes=3)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()

# Create dense layer with two input features and 3 output values
dense1 = layer_dense(2, 3)
# Create ReLU activation to be used with this layer
activation1 = activation_relu()

# Create second dense layer with 3 inputs (out of dense1)
dense2 = layer_dense(3, 3)

# 

# Create softmax activation to be used with dense layer
activation2 = activation_softmax()

# Create loss function
loss_function = loss_categoricalCrossentropy()

# Make forward pass of training data true layer 1
dense1.forward(X)

# Make forward and take the out of dense layer 1
activation1.forward(dense1.output)

# Make a foward pass through second dense layer, takes output of activation1 as 
# input
dense2.forward(activation1.output)

# Make foward with activation2 for layer dense 2
activation2.forward(dense2.output)

print(activation2.output[:5])

# Perform a foward pass through activation function
# it takes the output of second dense layer here and returns loss
loss = loss_function.calculate(activation2.output, y)
print("Loss: ", loss)

# Calculate accuracy from output of activation2 and targets
# calculation values along first axis
predictions = np.argmax(loss_activation.output, y)






