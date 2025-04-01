# ml_core/activations/relu.py

import numpy as np

class ReLU:
    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass for ReLU activation function.

        Parameters:
            input: Input data (numpy array)

        Returns:
            ReLU activation: max(0, input)
        """
        self.output = np.maximum(0, input)
        return self.output

    def backward(self, loss_gradient: np.ndarray, learning_rate: float = None) -> np.ndarray:
        """
        Backward pass for ReLU activation function.

        Parameters:
            loss_gradient: Gradient of the loss with respect to the output
            learning_rate: Not used for activation functions, but kept for API consistency

        Returns:
            Gradient of the loss with respect to the input
        """
        input_gradient = (self.output > 0).astype(float)
        return input_gradient * loss_gradient
