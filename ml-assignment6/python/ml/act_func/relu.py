"""ReLU (Rectified Linear Unit) activation function implementation."""
from ml.act_func.interface import IActFunc

class Relu(IActFunc):
    """ReLU (Rectified Linear Unit) activation function implementation."""

    def output(self, value: float) -> float:
        """Compute the activation function output.

        Args:
            value: The activation function input.
        Returns:
            The activation function value at the given input.
        """
        return value if 0.0 < value  else 0.0

    def delta(self, value: float) -> float:
        """Compute the activation function derivative (delta for backpropagation).
        Args:
            value: The activation function input.
        Returns:
            The derivative value at the given input.
        """
        return 1.0 if 0.0 < value else 0.0
