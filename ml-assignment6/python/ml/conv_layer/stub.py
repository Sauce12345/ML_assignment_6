"""Convolutional layer stub."""
from enum import IntEnum
from ml import utils
from ml.act_func import factory as act_func_factory
from ml.act_func.types import ActFuncType
from ml.conv_layer.interface import IConvLayer
from ml.types import Matrix2d

class KernelSize(IntEnum):
    """Enumeration of kernel size limits."""
    MIN = 1  # Minimum permitted kernel size.
    MAX = 11 # Maximum permitted kernel size.

# pylint: disable=unused-argument

class ConvStub(IConvLayer):
    """Convolutional layer stub."""

    def __init__(self, input_size: int, kernel_size: int,
                 act_func_type: ActFuncType = ActFuncType.IDENTITY) -> None:
        # Throw if the kernel size is outside range [1, 11] or larger than the input size.
        if kernel_size < KernelSize.MIN.value or kernel_size > KernelSize.MAX.value:
            raise ValueError(f"Invalid kernel size {kernel_size}: kernel size must be in range"
                             f"[{KernelSize.MIN.value}, {KernelSize.MAX.value}]!")
        if input_size < kernel_size:
            raise ValueError("Failed to create convolutional layer: "
                             "kernel size cannot be greater than input size!")

        # Initialize the member variables.
        self._input_gradients = utils.create_matrix2d(input_size)
        self._kernel          = utils.create_matrix2d(kernel_size)
        self._output          = utils.create_matrix2d(input_size)
        self._act_func        = act_func_factory.create(act_func_type)

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
        """Perform feedforward operation.
        
        Args: 
            input_data: Matrix holding input data.
        
        Returns: 
            True on success, false on failure.
        """
        op = "feedforward in convolutional layer"
        return (utils.match_dimensions(self.output_size(), len(input_data), op)
            and utils.is_matrix2d_square(input_data, op))

    def backpropagate(self, output_gradients: Matrix2d) -> bool:
        """Perform backpropagation.
        
        Args: 
            output_gradients: Matrix holding gradients from the next layer.
        
        Returns: 
            True on success, false on failure.
        """
        op = "backpropagation in convolutional layer"
        return (utils.match_dimensions(
            self.output_size(), len(output_gradients), op)
            and utils.is_matrix2d_square(output_gradients, op))

    def optimize(self, learning_rate: float) -> bool:
        """
        Perform optimization.
        
        Args: 
            learning_rate: Learning rate to use. Must be in range (0.0, 1.0].
        
        Returns: 
            True on success, false on failure.
        """
        op = "optimization in convolutional layer"
        return utils.check_learning_rate(learning_rate, op)

class MaxPoolStub(IConvLayer):
    """Max pooling layer stub."""

    def __init__(self, input_size: int, pool_size: int) -> None:
        # Check the pool dimensions, throw if invalid.
        if input_size == 0:
            raise ValueError("Input size cannot be 0!")
        if pool_size == 0:
            raise ValueError("Pool size cannot be 0!")
        if input_size < pool_size:
            raise ValueError("Input size cannot be smaller than the pool size!")
        if input_size % pool_size != 0:
            raise ValueError("Input size must be divisible by pool size!:" \
            f"input_size = {input_size}, pool_size = {pool_size}")

        # Initialize the pool matrices.
        output_size           = input_size // pool_size
        self._input           = utils.create_matrix2d(input_size)
        self._input_gradients = utils.create_matrix2d(input_size)
        self._output          = utils.create_matrix2d(output_size)

    def input_size(self) -> int:
        """Get the input size of the layer.
        
        Returns: 
            The input size of the layer.
        """
        return len(self._input)

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
        """Perform feedforward operation.
        
        Args: 
            input_data: Matrix holding input data.
        
        Returns: 
            True on success, false on failure.
        """
        op = "feedforward in max pooling layer"
        return (utils.match_dimensions(self.input_size(), len(input_data), op)
            and utils.is_matrix2d_square(input_data, op))

    def backpropagate(self, output_gradients: Matrix2d) -> bool:
        """Perform backpropagation.
        
        Args: 
            output_gradients: Matrix holding gradients from the next layer.
        
        Returns: 
            True on success, false on failure.
        """
        op = "backpropagation in max pooling layer"
        return (utils.match_dimensions(
            self.output_size(), len(output_gradients), op)
            and utils.is_matrix2d_square(output_gradients, op))

    def optimize(self, learning_rate: float) -> bool:
        """
        Perform optimization.
        
        Args: 
            learning_rate: Learning rate to use. Must be in range (0.0, 1.0].
        
        Returns: 
            True (optimization is a no-op for pooling layers).
        """
        return True
