"""Activation function interface."""

from abc import ABC, abstractmethod

class IActFunc(ABC):
    """Activation function interface."""

    @abstractmethod
    def output(self, value: float) -> float:
        """Compute the activation function output.

        Args:
            value: The activation function input.
        Returns:
            The activation function value at the given input.
        """

    @abstractmethod
    def delta(self, value: float) -> float:
        """Compute the activation function derivative (delta for backpropagation).
        Args:
            value: The activation function input.
        Returns:
            The derivative value at the given input.
        """
