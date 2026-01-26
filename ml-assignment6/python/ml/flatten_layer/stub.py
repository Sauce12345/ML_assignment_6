"""Flatten layer stub."""
from ml import utils
from ml.flatten_layer.interface import IFlatten
from ml.types import Matrix1d, Matrix2d

class FlattenStub(IFlatten):
    """Flatten layer stub."""

    def __init__(self, input_size: int) -> None:
        # Check the input size, throw an exception if invalid.
        if input_size == 0:
            raise ValueError("Input size cannot be 0!")
        self._input_gradients = utils.create_matrix2d(input_size)
        self._output          = utils.create_matrix1d(input_size * input_size)

    def input_size(self) -> int:
        """Get the input size of the layer.
        
        Returns: 
            The input size of the layer.
        """
        return len(self._input_gradients)

    def output_size(self) -> int:
        """Get the output size of the layer.
        
        Returns: 
            The output size of the layer.
        """
        return len(self._output)

    def output(self) -> Matrix2d:
        """Get the output of the layer.
        
        Returns: 
            Matrix holding the output of the layer.
        """
        return self._output

    def input_gradients(self) -> Matrix2d:
        """Get the input gradients of the layer.
        
        Returns: 
            Matrix holding the input gradients of the layer.
        """
        return self._input_gradients

    def feedforward(self, input_data: Matrix2d) -> bool:
        """Flatten the input from 2D to 1D.
        
        Args: 
            input_data: Matrix holding input data.
        
        Returns: 
            True on success, false on failure.
        """
        op = "feedforward in flatten layer"
        return (utils.match_dimensions(self.input_size(), len(input_data), op)
                and utils.is_matrix2d_square(input_data))

    def backpropagate(self, output_gradients: Matrix1d) -> bool:
        """Unflatten the output gradients from 1D to 2D.
        
        Args: 
            output_gradients: Matrix holding output gradients.
        
        Returns: 
            True on success, false on failure.
        """
        op = "backpropagation in flatten layer"
        return utils.match_dimensions(self.output_size(), len(output_gradients), op)
