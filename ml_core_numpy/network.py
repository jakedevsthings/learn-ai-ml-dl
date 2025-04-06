# ml_core/network.py

import numpy as np

class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        Apply each layer's forward method to the inputs.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, loss_gradient: np.ndarray, learning_rate: float) -> None:
        """
        Backward pass through the network.
        Apply each layer's backward method to the loss gradient.
        """
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient=loss_gradient, learning_rate=learning_rate)

    def get_all_activations(self, inputs: np.ndarray) -> list[np.ndarray]:
        """
        Get the activations of all layers in the network.
        """
        activations = []
        for layer in self.layers:
            inputs = layer.forward(inputs)
            activations.append(inputs.copy())
        return activations