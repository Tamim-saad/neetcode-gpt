import numpy as np
from numpy.typing import NDArray
from typing import List


# Input:
# x = [1.0, 2.0]
# weights = [[[0.1, 0.2], [0.3, 0.4]], [[0.5], [0.6]]]
# biases = [[0.1, 0.1], [0.0]]

# Output: [1.06]


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        
        prev_output = x.copy()
        for i in range(len(weights)):
          prev_output = np.dot(prev_output, weights[i]) + biases[i]
          
          if i<len(weights)-1:
            prev_output = np.maximum(0, prev_output)

          
        return np.round(prev_output, 5)
