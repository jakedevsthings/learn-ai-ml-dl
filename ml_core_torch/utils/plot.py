# /learn-ai-ml-dl/ml_core_torch/utils/plot.py

import matplotlib.pyplot as plt

def plot_loss_and_accuracy(losses, accuracies, interval=10):
    epochs = list(range(0, len(losses) * interval, interval))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.ylim([0, 1.05])
    plt.grid(True)

    plt.tight_layout()
    plt.show()