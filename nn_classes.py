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
        
# ReLU activation function
class activation_relu:
    
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
class activation_softmax:
    
    # Forward pass
    def forward(self, inputs):
        
        # Get unnomalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities
        
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
    def foward(self, y_pred, y_true):
        
        #Number of samples in a batch
        samples = len(y_pred)
        
        # Clip data to prevent div by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Probabilties for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), y_pred]
        # Mask values - only for one-hot encoded lables
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
            
        # Losses
        negativ_log_likelihoods = -np.log(correct_confidences)
        return negativ_log_likelihoods
        
        