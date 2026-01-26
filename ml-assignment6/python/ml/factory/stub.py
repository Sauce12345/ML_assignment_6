"""Machine learning stub factory."""
from ml.act_func import factory as act_func_factory
from ml.act_func.types import ActFuncType
from ml.act_func.interface import IActFunc
from ml.conv_layer.interface import IConvLayer
from ml.conv_layer.stub import ConvStub, MaxPoolStub
from ml.dense_layer.interface import IDenseLayer
from ml.dense_layer.stub import DenseStub
from ml.flatten_layer.interface import IFlatten
from ml.flatten_layer.stub import FlattenStub
from ml.factory.interface import IFactory

class FactoryStub(IFactory):
    """Machine learning stub factory."""

    def act_func(self, act_func_type: ActFuncType) -> IActFunc:
        """Create an activation function.

        Args:
            act_func_type: The type of activation function to create.

        Returns:
            The new activation function."""
        return act_func_factory.create(ActFuncType.IDENTITY)

    def conv_layer(self, input_size: int, kernel_size: int,
                   act_func_type: ActFuncType) -> IConvLayer:
        """Create a convolutional layer.
        
        Args:
            input_size: Input size. Must be greater than 0.
            kernel_size: Kernel size. Must be greater than 0 and smaller than the input size.
            act_func: Activation function to use.

        Returns:
            The new convolutional layer.
        """
        return ConvStub(input_size, kernel_size, act_func_type)

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
        return DenseStub(input_size, output_size, act_func_type)

    def flatten_layer(self, input_size: int) -> IFlatten:
        """Create a flatten layer.

        Args:
            input_size: Input size. Must be greater than 0.

        Returns:
            The new flatten layer.
        """
        return FlattenStub(input_size)

    def max_pool(self, input_size: int, pool_size: int) -> IConvLayer:
        """Create a max pooling layer.
        
        Args:
            input_size: Input size. Must be greater than 0.
            pool_size: Pool size. Must divide the input size.

        Returns:
            The new max pooling layer.
        """
        return MaxPoolStub(input_size, pool_size)
