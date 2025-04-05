# ml_core/layers/dense.py

import numpy as np

class DenseLayer:
    def __init__(self, input_size: int, output_size: int, initializer: str = 'xavier'):
        """
        Initialize the layer.

        Parameters:
        - input_size: number of input features
        - output_size: number of output features
        - initializer: weight initialization method ('xavier', 'he', or 'normal')
        """
        self.input_size = input_size
        self.output_size = output_size

        if initializer == 'xavier':
            limit = np.sqrt(6 / (input_size + output_size))
            self.weights = np.random.uniform(-limit, limit, (output_size, input_size))
        elif initializer == 'he':
            std = np.sqrt(2 / input_size)
            self.weights = np.random.randn(output_size, input_size) * std
        elif initializer == 'normal':
            self.weights = np.random.randn(output_size, input_size)
        else:
            raise ValueError(f"Unknown initializer: '{initializer}'")

        self.bias = np.zeros((output_size, 1))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass through the dense layer.

        Calculates the linear transformation Z = W * X^T + b, where X is the input,
        W are the weights, and b is the bias. The input is expected to be
        (num_samples, input_size), and it's transposed for matrix multiplication
        with weights (output_size, input_size). The bias (output_size, 1)
        is added using broadcasting. The result is then transposed back to
        (num_samples, output_size).

        Args:
            inputs: Input data of shape (num_samples, input_size).

        Returns:
            Output of the layer, shape (num_samples, output_size).
        """
        # Ensure input is a numpy array and store it
        self.input = np.array(inputs)
        # Z = W * X^T + b
        output_transposed = np.dot(self.weights, self.input.T) + self.bias
        self.output = output_transposed.T
        return self.output

    def backward(self, loss_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Performs the backward pass (backpropagation) through the dense layer.

        Updates weights and biases based on the gradient of the loss with respect
        to the layer's output (loss_gradient). It calculates the gradients
        with respect to weights (dL/dW), bias (dL/db), and the layer's input
        (dL/dX_prev), then updates W and b using the learning rate.

        Args:
            loss_gradient: Gradient of the loss w.r.t. the output
                            of this layer (dL/dA). Shape (num_samples, output_size).
            learning_rate: The learning rate for parameter updates.

        Returns:
            Gradient of the loss w.r.t. the input of this layer (dL/dX_prev).
            Shape (num_samples, input_size).

        Calculation Details:
            Let m = num_samples.
            loss_gradient (dL/dA) shape: (m, output_size)
            Need dL/dZ = dL/dA.T, shape (output_size, m)

            1. dL/dW = (1/m) * dL/dZ @ X_prev.T
               - dL/dZ shape: (output_size, m)
               - self.input (X_prev) shape: (m, input_size)
               - Result dL/dW shape: (output_size, input_size)

            2. dL/db = (1/m) * sum(dL/dZ, axis=1)
               - Summing dL/dZ along the samples axis.
               - Result dL/db shape: (output_size, 1)

            3. dL/dX_prev = W.T @ dL/dZ
               - self.weights.T shape: (input_size, output_size)
               - dL/dZ shape: (output_size, m)
               - Result dL/dX_prev (transposed) shape: (input_size, m)

            The final returned input_gradient is dL/dX_prev.T, shape (m, input_size).
        """
        num_samples = self.input.shape[0]
        loss_gradient_transposed = loss_gradient.T # Shape (output_size, num_samples)

        # Calculate dL/dW
        weights_gradient = np.dot(loss_gradient_transposed, self.input) / num_samples # Shape (output_size, input_size)

        # Calculate dL/db
        bias_gradient = np.sum(loss_gradient_transposed, axis=1, keepdims=True) / num_samples # Shape (output_size, 1)

        # Calculate dL/dX_prev (transposed)
        input_gradient_transposed = np.dot(self.weights.T, loss_gradient_transposed) # Shape (input_size, num_samples)

        # Update weights and bias
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        # Return dL/dX_prev (transposed back)
        input_gradient = input_gradient_transposed.T # Shape (num_samples, input_size)
        return input_gradient