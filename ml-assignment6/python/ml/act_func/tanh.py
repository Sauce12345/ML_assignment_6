"""Tanh (hyperbolic tangent) activation function implementation."""
import math

from ml.act_func.interface import IActFunc

class Tanh(IActFunc):
    """Tanh (hyperbolic tangent) activation function implementation."""

    def output(self, value: float) -> float:
        """Compute the activation function output.

        Args:
            value: The activation function input.
        Returns:
            The activation function value at the given input.
        """
        return math.tanh(value)

    def delta(self, value: float) -> float:
        """Compute the activation function derivative (delta for backpropagation).
        Args:
            value: The activation function input.
        Returns:
            The derivative value at the given input.
        """
        tanh_output = math.tanh(value)
        return 1.0 - tanh_output * tanh_output


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
