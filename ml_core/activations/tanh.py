# ml_core/activations/tanh.py

import numpy as np

class Tanh:
    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass for hyperbolic tangent activation function.

        Parameters:
            input: Input data (numpy array)

        Returns:
            Hyperbolic tangent activation: tanh(input)
        """
        self.output = np.tanh(input)
        return self.output

    def backward(self, loss_gradient: np.ndarray, learning_rate: float = None) -> np.ndarray:
        """
        Backward pass for hyperbolic tangent activation function.
        
        Parameters:
            loss_gradient: Gradient of the loss with respect to the output
            learning_rate: Not used for activation functions, but kept for API consistency
        
        Returns:
            Gradient of the loss with respect to the input
        """
        return loss_gradient * (1 - np.tanh(self.output) ** 2)
    