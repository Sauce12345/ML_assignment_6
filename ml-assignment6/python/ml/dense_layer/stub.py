"""Dense layer stub."""
from ml import utils
from ml.act_func.types import ActFuncType
from ml.dense_layer.interface import IDenseLayer
from ml.types import Matrix1d

# pylint: disable=unused-argument, duplicate-code

class DenseStub(IDenseLayer):
    """Dense layer stub."""

    def __init__(self, input_size: int, output_size: int,
                 act_func_type: ActFuncType = ActFuncType.RELU) -> None:
        # Throw exception if node count or the weight count is 0.
        if output_size == 0:
            raise ValueError("Node count cannot be 0!")
        if input_size == 0:
            raise ValueError("Weight count cannot be 0!")

        # Initialize member variables.
        self._input_gradients = utils.create_matrix1d(input_size)
        self._output          = utils.create_matrix1d(output_size)

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

    def output(self) -> Matrix1d:
        """Get the output of the layer.
        
        Returns: 
            Matrix holding the output of the layer.
        """
        return self._output

    def input_gradients(self) -> Matrix1d:
        """Get the input gradients of the layer.
        
        Returns: 
            Matrix holding the input gradients of the layer.
        """
        return self._input_gradients

    def feedforward(self, input_data: Matrix1d) -> bool:
        """Perform feedforward operation.
        
        Args: 
            input_data: Matrix holding input data.
        
        Returns: 
            True on success, false on failure.
        """
        op = "feedforward in dense layer"
        if not utils.match_dimensions(self.input_size(), len(input_data), op):
            return False
        return True

    def backpropagate(self, output_gradients: Matrix1d) -> bool:
        """Perform backpropagation.
        
        Args: 
            output_gradients: Matrix holding gradients from the next layer.
        
        Returns: 
            True on success, false on failure.
        """
        op = "backpropagation in dense layer"
        if not utils.match_dimensions(self.output_size(), len(output_gradients), op):
            return False
        return True

    def optimize(self, input_data: Matrix1d, learning_rate: float) -> bool:
        """
        Perform optimization.
        
        Args: 
            input_data: Matrix holding input data.
            learning_rate: Learning rate to use. Must be in range (0.0, 1.0].
        
        Returns: 
            True on success, false on failure.
        """
        op = "optimization in dense layer"
        if (not utils.match_dimensions(self.input_size(), len(input_data), op)
            or not utils.check_learning_rate(learning_rate, op)):
            return False
        return True
