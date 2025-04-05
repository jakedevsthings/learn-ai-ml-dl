# Milestone 1.1: Trivial Neural Network Summary

## Implementation Overview
In this milestone, I created a trivial neural network from scratch with the help of Numpy. Along with my sidekick ChatGPT, I filled in the classes for the layers of the network beginning with the dense layer, then learned about and implemented different types of activations. I learned that these activations are the essential part of making a network non-linear and able to learn complex patterns. I also learned the importance of backpropagation and gradient descent in updating the weights of the network. This milestone culminated in a working simple neural network with one hidden layer that correctly classified the XOR problem.

## Key Learnings
- I learned about backpropogation through gradient descent, activations for nonlinearity, and how the weights and biases affect the output of my network
- I learned about the gradient of the loss with respect to the weights and biases through the chain rule, which I remember from my calculus classes!
- I learned (with the help of an assistant) about visualization techniques for my network, such as decision boundaries and layer activations, to literally see the network learn what to do about the XOR data and classify it nonlinearly.
- The major insights I gained from this milestone were the importance of backpropagation and gradient descent in updating the weights of the network, and the importance of activations for nonlinearity in making the network able to learn complex patterns, and why exactly this type of neural network is necessary to be able to learn the XOR problem.

## Technical Implementation Details
- Architecture:
  - The architecture of my simple neural network to classify the XOR problem was a single hidden layer with 2 neurons, and a single output neuron.
  - The activation function for the hidden layer was switched around during different visualizations but eventually settled on tanh.
  - The activation function for the output layer was sigmoid due to the binary classification nature of the XOR problem.

- Training Process:
  - The training data was the truth table for the binary XOR function, found in [xor_dataset.py](xor_dataset.py)
  - The network was trained for 5000 epochs with a learning rate of 0.2. I learned about the impact of learning rate on the training process.

## Results
- As can be seen in [xor_visualization.ipynb](../notebooks/xor_visualization.ipynb), the network was able to correctly classify the XOR problem after training, with an accuracy of 100% and a final loss of 0.024100.
- I observed a key insight in the training behavior that due to the tanh activation function, the accuracy jumped at around epoch 1500, and then the loss also started to decrease. 
- I was also able to directly visualize the decision boundary of the network, which was a hyperbolic line, and the layer activations, which were the inputs to the hidden layer.

## Limitations
- One limitation of this implementation is that it runs on my CPU and is not optimized for speed. This is due to the direct implementation from scratch only using Numpy.
- In future phases, I plan to implement a GPU-accelerated implementation using PyTorch or TensorFlow.

## Next Steps
- Next steps are onwards to milestone 1.2: MNIST Digit Recognition!

## Training Results and Visualizations
[Training results and visualizations](../notebooks/xor_visualization.ipynb)

