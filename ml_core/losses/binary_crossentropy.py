import numpy as np

class BinaryCrossentropy:
    """
    Binary cross-entropy loss function.
    """
    def __init__(self) -> None:
        self.predictions = None
        self.labels = None
    
    def forward(self, predictions: np.ndarray, labels: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """
        Compute the binary cross-entropy loss.
        Epsilon is used to avoid log(0)

        BCE Loss = -[labels * log(predictions) + (1 - labels) * log(1 - predictions)]
        """
        self.predictions = np.clip(predictions, epsilon, 1 - epsilon)
        self.labels = labels

        loss = -(labels * np.log(self.predictions) + (1 - labels) * np.log(1 - self.predictions))
        return np.mean(loss)  # avg loss over batch
        
    def backward(self) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to the predictions.
        """
        gradient = -(self.labels / self.predictions) + ((1 - self.labels) / (1 - self.predictions))
        return gradient / self.labels.shape[0]  # avg gradient over batch
