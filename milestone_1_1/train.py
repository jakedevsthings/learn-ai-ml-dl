# milestone_1_1/train.py

import numpy as np

from ml_core.losses import BinaryCrossentropy as BCE_Loss
from ml_core import Network as Neural_Network
from ml_core.layers import DenseLayer as Dense_Layer
from ml_core.activations import Sigmoid as Sigmoid_Activation
from milestone_1_1.xor_dataset import inputs, labels
from milestone_1_1.visualize import plot_decision_boundary

# Convert labels to numpy array for calculations
labels = np.array(labels)

# Train on the XOR dataset
num_epochs = 10000
learning_rate = 0.1

# Initialize the loss function
bce_loss = BCE_Loss()

# Initialize the network
# 2 inputs, 4 hidden, 1 output
neural_network = Neural_Network([
    Dense_Layer(input_size=2, output_size=4),
    Sigmoid_Activation(),
    Dense_Layer(input_size=4, output_size=1),
    Sigmoid_Activation(),
])

for epoch in range(num_epochs):
    # Forward pass
    predictions = neural_network.forward(inputs=inputs)
    loss = bce_loss.forward(predictions=predictions, labels=labels)
    gradient = bce_loss.backward()
    
    # Backward pass
    neural_network.backward(loss_gradient=gradient, learning_rate=learning_rate)

# Visualize the decision boundary
plot_decision_boundary(model=neural_network, X=inputs, y=labels)
