# /learn-ai-ml-dl/ml_core_torch/train/train_loop.py

import torch
from torch import nn


def train_model(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    num_epochs: int = 5000,
):
    model.to(device)
    inputs = inputs.to(device)
    labels = labels.to(device)

    loss_history = []
    accuracy_history = []

    for epoch in range(num_epochs):
        model.train()

        preds = model(inputs)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                predicted_labels = (preds > 0.5).float()
                accuracy = (predicted_labels == labels).float().mean().item()

                loss_history.append(loss.item())
                accuracy_history.append(accuracy)

                if epoch % 100 == 0 or epoch == num_epochs - 1:
                    print(
                        f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}"
                    )

    return loss_history, accuracy_history
