# milestone_1_1/train.py

import sys
import os
import numpy as np

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml_core.losses import BinaryCrossentropy as BCE_Loss
from ml_core import Network as Neural_Network
from ml_core.layers import DenseLayer as Dense_Layer
from ml_core.activations import Sigmoid as Sigmoid_Activation
from ml_core.activations import ReLU as ReLU_Activation
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
    Dense_Layer(input_size=2, output_size=4, initializer='he'),
    ReLU_Activation(),
    Dense_Layer(input_size=4, output_size=1, initializer='xavier'),
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
