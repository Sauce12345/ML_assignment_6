"""Convolutional neural network (CNN) interface."""
from abc import ABC, abstractmethod
from ml.types import Matrix1d, Matrix2d

class ICnn(ABC):
    """Convolutional neural network (CNN) interface."""

    @abstractmethod
    def input_size(self) -> int:
        """Get the input size of the CNN.
        
        Returns:
            The input size of the CNN.
        """

    @abstractmethod
    def output_size(self) -> int:
        """Get the output size of the CNN.
        
        Returns:
            The output size of the CNN.
        """

    @abstractmethod
    def predict(self, input_data: Matrix2d) -> Matrix1d:
        """Predict based on the given input.
        
        Args:
            input_data: Input for which to predict.

        Returns:
            The predicted output.
        """
