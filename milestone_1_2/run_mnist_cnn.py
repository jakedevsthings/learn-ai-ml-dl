# /learn-ai-ml-dl/milestone_1_2/run_mnist_cnn.py

"""
Run the MNIST CNN.
"""

import torch.nn as nn
import torch.optim as optim
from ml_core_torch.models.mnist_cnn import MNISTCNN
from ml_core_torch.utils.data import load_mnist
from ml_core_torch.train.train_multiclass_model import train_multiclass_model
from ml_core_torch.utils.plot import plot_loss_and_accuracy
from ml_core_torch.config.defaults import DEVICE

# Load data
train_loader, test_loader = load_mnist(batch_size=64)

# Initialize model, loss, optimizer
model = MNISTCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
losses, accuracies = train_multiclass_model(
    model=model,
    dataloader=train_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=DEVICE,
    num_epochs=10,
)

# Plot training metrics
plot_loss_and_accuracy(losses, accuracies)
