# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:06:50 2023

@author: bzi55
"""

import numpy as np

# Dense Layer
class layer_dense:
    
    # Layer initialisation
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # TODO implement real random numbers from monte carlo implementation
        
    # Forward pass
    def forward(self,inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
        
    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        
# ReLU activation function
class activation_relu:
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        
    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()
        
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
        
class activation_softmax:
    
    # Forward pass
    def forward(self, inputs):
        
        # Get unnomalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities
        
    # Backward pass
    def backward(self, dvalues):
        
        # Create unititialized array
        self.dinputs = np.empty_like(dvalues) 
        
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
            enumerate( zip( self.output, dvalues ) ):
                
                # Flatten output array
                single_output = single_output.reshape(-1, 1)
                
                # Calculate Jacobian matrix of the output
                jacobian_matrix = np.diagflat(single_output) - \
                    np.dot( single_output, single_output.T )
                    
                # Calculate sample-wise gradient
                # and add it to the array of sample gradients
                self.dinputs[index] = np.dot( jacobian_matrix, 
                                             single_dvalues)
                
                
        
# Common loss class
class loss:
    
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        
        # Calc sample losses
        sample_losses = self.forward(output, y)
        
        # Calc mean loss
        data_loss = np.mean(sample_losses)

        return data_loss
    
class loss_categoricalCrossentropy(loss):
    
    # Forward pass
    def forward(self, y_pred, y_true):

        #Number of samples in a batch
        samples = len(y_pred)
        
        # Clip data to prevent div by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Probabilties for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), 
                y_true]
        # Mask values - only for one-hot encoded lables
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
            
        # Losses
        negativ_log_likelihoods = -np.log(correct_confidences)
        
        return negativ_log_likelihoods
        
    # Backward pass    
    def backward(self, dvalues, y_true):
        
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We will use the first sample to count them
        labels = len(dvalues[0])
        
        # If labels are sparse , turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
            
        # Calculate gradient
        self.dinputs = - y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        

    
        
        