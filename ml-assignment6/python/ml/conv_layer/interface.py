"""Convolutional layer interface."""
from abc import ABC, abstractmethod
from ml.types import Matrix2d

class IConvLayer(ABC):
    """Convolutional layer interface."""

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
        """Perform feedforward operation.
        
        Args: 
            input_data: Matrix holding input data.
        
        Returns: 
            True on success, false on failure.
        """

    @abstractmethod
    def backpropagate(self, output_gradients: Matrix2d) -> bool:
        """Perform backpropagation.
        
        Args: 
            output_gradients: Matrix holding gradients from the next layer.
        
        Returns: 
            True on success, false on failure.
        """

    @abstractmethod
    def optimize(self, learning_rate: float) -> bool:
        """
        Perform optimization.
        
        Args: 
            learning_rate: Learning rate to use. Must be in range (0.0, 1.0].
        
        Returns: 
            True on success, false on failure.
        """
