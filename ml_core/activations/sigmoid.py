# ml_core/layers/sigmoid.py

import numpy as np

class Sigmoid:
    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass for sigmoid activation function.

        Parameters:
            input: Input data (numpy array)
                
        Returns:
            Sigmoid activation: a = 1 / (1 + e^(-input))
        """
        self.output = 1 / (1 + np.exp(-input))
        return self.output
        
    def backward(self, loss_gradient: np.ndarray, learning_rate: float = None) -> np.ndarray:
        """
        Backward pass for sigmoid activation function.
        
        Computes the gradient of the loss with respect to the input of the sigmoid function.
        
        The chain rule gives us:
            - a = sigmoid(z) where z = Wx + b
            - output of this layer is dL/dz = dL/da * da/dz
            - dL/da is simply the loss_gradient
            - da/dz = a(1 - a)
        
        Parameters:
            loss_gradient: Gradient of the loss with respect to the output of this layer (dL/da)
            learning_rate: Not used for activation functions, but kept for API consistency
            
        Returns:
            Gradient of the loss with respect to the input of this layer (dL/dz)
        """
        sigmoid_derivative = self.output * (1 - self.output)
        return loss_gradient * sigmoid_derivative