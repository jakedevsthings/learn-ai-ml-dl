# milestone_1_1/visualize.py

import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """
    Visualize decision boundary of a trained model.
    
    Parameters:
        model: instance of your NeuralNetwork class with .forward(x) method
        X: input points (shape: [num_samples, 2])
        y: true binary labels (shape: [num_samples, 1])
    """
    # Convert inputs to numpy arrays if they aren't already
    X = np.array(X)
    y = np.array(y)
    
    h = 0.01  # Step size in mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Create mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]  # Shape: [num_points, 2]

    # Predict over grid
    Z = model.forward(grid)
    Z = Z.reshape(xx.shape)  # Reshape to match mesh

    # Plot contour map (decision surface)
    plt.contourf(xx, yy, Z, levels=100, cmap="RdBu", alpha=0.6)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)


    # Overlay true points
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", edgecolors='k')
    plt.title(title)
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.grid(True)
    plt.show()

def plot_all_layer_activations(model, resolution=100):
    """
    Plot the activations for all neurons in all layers of the model over 2D input space.
    Assumes input is 2D.
    """
    x_vals = np.linspace(-0.5, 1.5, resolution)
    y_vals = np.linspace(-0.5, 1.5, resolution)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid = np.c_[xx.ravel(), yy.ravel()]  # shape: (resolution², 2)

    layer_outputs = model.get_all_activations(grid)

    for layer_idx, activations in enumerate(layer_outputs):
        num_neurons = activations.shape[1]
        fig, axes = plt.subplots(1, num_neurons, figsize=(4 * num_neurons, 4))
        if num_neurons == 1:
            axes = [axes]
        
        for neuron_idx in range(num_neurons):
            Z = activations[:, neuron_idx].reshape(xx.shape)
            ax = axes[neuron_idx]
            im = ax.contourf(xx, yy, Z, levels=100, cmap='viridis')
            ax.set_title(f"Layer {layer_idx + 1} - Neuron {neuron_idx + 1}")
            fig.colorbar(im, ax=ax)
        
        plt.suptitle(f"Layer {layer_idx + 1} Activations")
        plt.tight_layout()
        plt.show()

def plot_layer_weights(model):

    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer, 'weights'):
            W = layer.weights  # shape: (input_dim, output_dim)
            W = W.T  # shape: (output_dim, input_dim) — one row per neuron

            fig, axes = plt.subplots(1, W.shape[0], figsize=(4*W.shape[0], 4))
            if W.shape[0] == 1:
                axes = [axes]

            vmin, vmax = -1.0, 1.0  # or auto from global weight min/max
            for i, ax in enumerate(axes):
                print(f"Layer {layer_idx+1} Neuron {i+1} weights: {W[i]}")
                ax.imshow(W[i].reshape(1, -1), cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
                ax.set_title(f"Layer {layer_idx + 1}, Neuron {i + 1}")
                ax.set_yticks([])
                ax.set_xticks(range(W.shape[1]))
                ax.set_xticklabels([f"x{j+1}" for j in range(W.shape[1])])
            
            plt.suptitle(f"Weights for Layer {layer_idx + 1}")
            plt.tight_layout()
            plt.show()
