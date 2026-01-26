"""Identity activation function implementation (no activation applied)."""
from ml.act_func.interface import IActFunc

class Identity(IActFunc):
    """Identity activation function implementation (no activation applied)."""

    def output(self, value: float) -> float:
        """Compute the activation function output.

        Args:
            value: The activation function input.
        Returns:
            The activation function value at the given input.
        """
        return value

    def delta(self, value: float) -> float:
        """Compute the activation function derivative (delta for backpropagation).
        Args:
            value: The activation function input.
        Returns:
            The derivative value at the given input.
        """
        return 1.0
