"""Flatten layer interface."""
from abc import ABC, abstractmethod
from ml.types import Matrix1d, Matrix2d

class IFlatten(ABC):
    """Flatten layer interface."""

    @abstractmethod
    def input_size(self) -> int:
        """Get the input size of the layer.
        
        Returns: 
            The input size of the layer.
        """

    @abstractmethod
    def output_size(self) -> int:
        """Get the output size of the layer.
        
        Returns: 
            The output size of the layer.
        """

    @abstractmethod
    def output(self) -> Matrix2d:
        """Get the output of the layer.
        
        Returns: 
            Matrix holding the output of the layer.
        """

    @abstractmethod
    def input_gradients(self) -> Matrix2d:
        """Get the input gradients of the layer.
        
        Returns: 
            Matrix holding the input gradients of the layer.
        """

    @abstractmethod
    def feedforward(self, input_data: Matrix2d) -> bool:
        """Flatten the input from 2D to 1D.
        
        Args: 
            input_data: Matrix holding input data.
        
        Returns: 
            True on success, false on failure.
        """

    @abstractmethod
    def backpropagate(self, output_gradients: Matrix1d) -> bool:
        """Unflatten the output gradients from 1D to 2D.
        
        Args: 
            output_gradients: Matrix holding output gradients.
        
        Returns: 
            True on success, false on failure.
        """
