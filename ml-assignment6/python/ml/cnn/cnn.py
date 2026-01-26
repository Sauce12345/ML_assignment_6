"""Convolutional neural network (CNN) implementation."""
import random

from ml import utils
from ml.act_func.types import ActFuncType
from ml.cnn.interface import ICnn
from ml.factory.interface import IFactory
from ml.types import Matrix1d, Matrix2d, Matrix3d

# pylint: disable=too-many-arguments

class _TrainOrderList:
    """Training order list implementation."""

    def __init__(self, size) -> None:
        self._data = [0] * size
        for i in range(size):
            self._data[i] = i

    def data(self) -> tuple[int]:
        """Get the train order list.
        
        Returns:
            The train order list as a tuple.
        """
        return self._data

    def shuffle(self) -> None:
        """Shuffle the training order list."""
        size = len(self._data)
        for i in range(size):
            r = random.randint(0, size - 1)
            temp = self._data[i]
            self._data[i] = self._data[r]
            self._data[r] = temp


class Cnn(ICnn):
    """Convolutional neural network (CNN) implementation."""

    def __init__(self, factory: IFactory, conv_input: int, conv_kernel: int,
                 conv_func: ActFuncType, pool_size: int, dense_output: int,
                 dense_func: ActFuncType) -> None:
        self._conv_layers   = []
        self._dense_layers  = []
        self._factory       = factory

        # Add convolutional layers.
        self._conv_layers.append(factory.conv_layer(conv_input, conv_kernel, conv_func))

        self._conv_layers.append(factory.max_pool(self._conv_output_size(), pool_size))

        # Add a flatten layer.
        self._flatten_layer = factory.flatten_layer(self._conv_output_size())

        # Add a dense layer.
        dense_input = self._flatten_layer.output_size()
        self._dense_layers.append(factory.dense_layer(dense_input, dense_output, dense_func))

    def input_size(self) -> int:
        """Get the input size of the CNN.
        
        Returns:
            The input size of the CNN.
        """
        return self._conv_layers[0].input_size()

    def output_size(self) -> int:
        """Get the output size of the CNN.
        
        Returns:
            The output size of the CNN.
        """
        last = len(self._dense_layers) - 1
        return self._dense_layers[last].output_size()

    def _output(self) -> Matrix1d:
        last = len(self._dense_layers) - 1
        return self._dense_layers[last].output()

    def predict(self, input_data: Matrix2d) -> Matrix1d:
        """Predict based on the given input.
        
        Args:
            input_data: Input for which to predict.

        Returns:
            The predicted output.
        """
        self._feedforward(input_data)
        return self._output()

    def add_dense_layer(self, output_size: int, act_func_type: ActFuncType) -> None:
        """Add dense layer.

           The input size is automatically adjusted in accordance with the previous layer.
        
        Args:
            output_size: Output size.
            act_func_type: Activation function to use.
        """
        self._dense_layers.append(
            self._factory.dense_layer(self.output_size(), output_size, act_func_type))

    def train(self, train_in: Matrix3d, train_out: Matrix2d, epoch_count: int,
              learning_rate: float) -> bool:
        """Train the network.
        
        Args:
           train_in: Training input sets.
           train_out: Training output sets.
           epoch_count: Number of epochs to train the model.
           learning_rate: Learning rate to use during training.

           Returns:
               True on success, false on failure.
        """
        op = "training of CNN"
        # Check the input parameters, return false on failure.
        if not utils.check_learning_rate(learning_rate, op):
            return False
        if not utils.check_epoch_count(epoch_count, op):
            return False
        set_count = min(len(train_in), len(train_out))
        if not utils.check_train_set_count(set_count, op):
            return False
        # Create a training order list.
        train_order = _TrainOrderList(set_count)
        # Train the network the specified number of epochs.
        for _ in range(epoch_count):
            # Shuffle the training order list at the start of each epoch.
            train_order.shuffle()
            # Iterate through the training sets, return false on failure.
            for i in train_order.data():
                input_data  = train_in[i]
                output = train_out[i]
                success = (self._feedforward(input_data) and self._backpropagate(output)
                           and self._optimize(learning_rate))
                if not success:
                    return False
        # Return true on success.
        return True

    def _conv_output_size(self) -> int:
        last = len(self._conv_layers) - 1
        return self._conv_layers[last].output_size()

    def _conv_output(self) -> Matrix2d:
        last = len(self._conv_layers) - 1
        return self._conv_layers[last].output()

    def _feedforward(self, input_data) -> bool:
        # Run feedforward operation in the convolutional layers, return false on failure.
        success = self._conv_layers[0].feedforward(input_data)
        for i in range(1, len(self._conv_layers)):
            prev_layer = self._conv_layers[i - 1]
            success &= self._conv_layers[i].feedforward(prev_layer.output())
        if not success:
            return False
        # Flatten the output from the convolutional layers, return false on failure.
        success = self._flatten_layer.feedforward(self._conv_output())
        if not success:
            return False
        # Run feedforward operation in the dense layers.
        success = self._dense_layers[0].feedforward(self._flatten_layer.output())
        for i in range(1, len(self._dense_layers)):
            prev_layer = self._dense_layers[i - 1]
            success &= self._dense_layers[i].feedforward(prev_layer.output())
        # Return true on success.
        return success

    def _backpropagate(self, output: Matrix1d) -> bool:
        # Backpropagate through the dense layers, return false on failure.
        last = len(self._dense_layers) - 1
        success = self._dense_layers[last].backpropagate(output)
        for i in range(last, 0, -1):
            input_gradients = self._dense_layers[i].input_gradients()
            success &= self._dense_layers[i - 1].backpropagate(input_gradients)
        if not success:
            return False
        # Unflatten the input gradients from the first dense layer, return false on failure.
        input_gradients = self._dense_layers[0].input_gradients()
        success = self._flatten_layer.backpropagate(input_gradients)
        if not success:
            return False
        # Backpropagate through the convolutional layers.
        last = len(self._conv_layers) - 1
        input_gradients = self._flatten_layer.input_gradients()
        success = self._conv_layers[last].backpropagate(input_gradients)
        for i in range(last, 0, -1):
            input_gradients = self._conv_layers[i].input_gradients()
            success &= self._conv_layers[i - 1].backpropagate(input_gradients)
        # Return true on success.
        return success

    def _optimize(self, learning_rate: float) -> bool:
        # Optimize the convolutional layers, return false on failure.
        success = self._conv_layers[0].optimize(learning_rate)
        for i in range(1, len(self._conv_layers)):
            success &= self._conv_layers[i].optimize(learning_rate)
        if not success:
            return False
        # Optimize the dense layers, return false on failure.
        success = self._dense_layers[0].optimize(self._flatten_layer.output(), learning_rate)
        for i in range(1, len(self._dense_layers)):
            prev_layer = self._dense_layers[i - 1]
            success &= self._dense_layers[i].optimize(prev_layer.output(), learning_rate)
        # Return true on success.
        return success
