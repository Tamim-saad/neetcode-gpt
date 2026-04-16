import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float],
    ) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        x = np.array(x, dtype=np.float64)
        w1 = np.array(W1, dtype=np.float64)
        b1 = np.array(b1, dtype=np.float64)
        w2 = np.array(W2, dtype=np.float64)
        b2 = np.array(b2, dtype=np.float64)
        y_true = np.array(y_true, dtype=np.float64)

        # Forward Pass
        z1 = np.dot(x, np.transpose(w1)) + b1
        a1 = np.maximum(z1, 0)
        z2 = np.dot(a1, np.transpose(w2)) + b2
        y_hat = z2

        n = len(y_true)
        # Backward Pass
        loss = np.mean(np.square(y_hat - y_true))
        dz2 = (2 / n) * (z2 - y_true)

        dW2 = np.outer(dz2, a1)
        # z2 = W2*a1 + b2 => dW2 = dz2 * a1.T
        db2 = dz2

        da1 = np.dot(np.transpose(w2), dz2)  # z2 = W2*a1 + b2 => da1 = w2.T * dz2
        dz1 = da1 * (z1 > 0)  # a1 = Relu(z1) => dz1 = da1 * Relu'(z1) [Relu'(z1) = 0/1]
        dW1 = np.outer(dz1, x)  # z1 = W1*x + b1 => dW1 = dz1 * x.T
        db1 = dz1

        return {
            "loss": (np.round(loss, 4) + 0.0),
            "dW1": (np.round(dW1, 4) + 0.0).tolist(),
            "db1": (np.round(db1, 4) + 0.0).tolist(),
            "dW2": (np.round(dW2, 4) + 0.0).tolist(),
            "db2": (np.round(db2, 4) + 0.0).tolist(),
        }
