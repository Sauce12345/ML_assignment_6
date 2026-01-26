"""Machine learning factory."""
from ml.act_func import factory as act_func_factory
from ml.act_func.types import ActFuncType
from ml.act_func.interface import IActFunc
from ml.conv_layer.interface import IConvLayer
from ml.conv_layer.stub import ConvStub, MaxPoolStub
from ml.dense_layer.interface import IDenseLayer
from ml.dense_layer.dense import Dense
from ml.factory.interface import IFactory
from ml.factory.stub import FactoryStub
from ml.flatten_layer.interface import IFlatten
from ml.flatten_layer.stub import FlattenStub

class Factory(IFactory):
    """Machine learning factory."""

    def act_func(self, act_func_type: ActFuncType) -> IActFunc:
        """Create an activation function.

        Args:
            act_func_type: The type of activation function to create.

        Returns:
            The new activation function."""
        return act_func_factory.create(act_func_type)

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
         # Todo: Replace ConvStub with Conv when implemented.
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
        return Dense(input_size, output_size, act_func_type)

    def flatten_layer(self, input_size: int) -> IFlatten:
        """Create a flatten layer.

        Args:
            input_size: Input size. Must be greater than 0.

        Returns:
            The new flatten layer.
        """
        # Todo: Replace FlattenStub with Flatten when implemented.
        return FlattenStub(input_size)

    def max_pool(self, input_size: int, pool_size: int) -> IConvLayer:
        """Create a max pooling layer.
        
        Args:
            input_size: Input size. Must be greater than 0.
            pool_size: Pool size. Must divide the input size.

        Returns:
            The new max pooling layer.
        """
        # Todo: Replace MaxPoolStub with MaxPool when implemented.
        return MaxPoolStub(input_size, pool_size)

def create(stub: bool = False) -> IFactory:
    """Create a factory.

    Args:
        stub: True to create a stub factory (default = false).

    Returns:
        The new factory.
    """
    return FactoryStub() if stub else Factory()
