# /learn-ai-ml-dl/ml_core_torch/train/train_multiclass_model.py

"""
Train a multiclass model.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_multiclass_model(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    num_epochs: int,
) -> tuple[list[float], list[float]]:
    """Train a multiclass model.

    Args:
        model: The model to train.
        dataloader: The dataloader to train on.
        optimizer: The optimizer to use.
        loss_fn: The loss function to use.
        device: The device to train on.
        num_epochs: The number of epochs to train for.

    Returns:
        A tuple of the loss history and the accuracy history.
    """
    model.to(device)  # Move model to device
    model.train()  # Set model to training mode

    loss_history = []
    accuracy_history = []

    for epoch in range(num_epochs):
        running_loss = 0
        correct = 0
        total = 0

        # Iterate over the dataloader
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the running loss, correct, and total
            running_loss += loss.item() * x_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        # Calculate the epoch loss and accuracy
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        loss_history.append(epoch_loss)
        accuracy_history.append(epoch_acc)

        # Print the epoch loss and accuracy
        if epoch % 1 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc * 100:.2f}%"
            )

    return loss_history, accuracy_history
