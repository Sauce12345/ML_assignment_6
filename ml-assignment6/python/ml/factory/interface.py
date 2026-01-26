"""Machine learning factory interface."""
from abc import ABC, abstractmethod
from ml.act_func.types import ActFuncType
from ml.act_func.interface import IActFunc
from ml.conv_layer.interface import IConvLayer
from ml.dense_layer.interface import IDenseLayer
from ml.flatten_layer.interface import IFlatten

class IFactory(ABC):
    """Machine learning factory interface."""

    @abstractmethod
    def act_func(self, act_func_type: ActFuncType) -> IActFunc:
        """Create an activation function.

        Args:
            act_func_type: The type of activation function to create.

        Returns:
            The new activation function.
        """

    @abstractmethod
    def conv_layer(self, input_size: int, kernel_size: int,
                   act_func_type: ActFuncType) -> IConvLayer:
        """Create a convolutional layer.
        
        Args:
            input_size: Input size. Must be greater than 0.
            kernel_size: Kernel size. Must be greater than 0 and smaller than the input size.
            act_func_type: Activation function to use.

        Returns:
            The new convolutional layer.
        """

    @abstractmethod
    def dense_layer(self, input_size: int, output_size: int,
                    act_func_type: ActFuncType) -> IDenseLayer:
        """Create a dense layer.
        
        Args:
            input_size: Input size. Must be greater than 0.
            output_size: Output size. Must be greater than 0.
            act_func_type: Activation function to use.

        Returns:
            The new dense layer.
        """

    @abstractmethod
    def flatten_layer(self, input_size: int) -> IFlatten:
        """Create a flatten layer.

        Args:
            input_size: Input size. Must be greater than 0.

        Returns:
            The new flatten layer.
        """

    @abstractmethod
    def max_pool(self, input_size: int, pool_size: int) -> IConvLayer:
        """Create a max pooling layer.
        
        Args:
            input_size: Input size. Must be greater than 0.
            pool_size: Pool size. Must divide the input size.

        Returns:
            The new max pooling layer.
        """
