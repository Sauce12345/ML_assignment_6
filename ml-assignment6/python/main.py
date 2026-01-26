"""Training and prediction of a CNN (Convolutional Neural Network)."""
import sys

from ml.factory import factory as ml_factory
from ml.act_func.types import ActFuncType
from ml.cnn.cnn import Cnn
from ml.cnn.interface import ICnn
from ml.types import Matrix3d

# pylint: disable=consider-using-enumerate

def _use_real_impl() -> bool:
    """Check whether real implementations are to be used."""
    return len(sys.argv) > 1 and sys.argv[1] == "real"


def predict_and_print(cnn: ICnn, inputs: Matrix3d) -> None:
    """
    Predict and print with the given CNN (Convolutional Neural Network).

    Args:
        cnn: The CNN with which to predict.
        inputs: Input sets to predict with.
    """
    # Terminate function if no input sets are available.
    if len(inputs) == 0:
        return
    last = len(inputs) - 1
    print("--------------------------------------------------------------------------------")

    # Perform prediction with each input set, print the predicted output in the terminal.
    for i in range(len(inputs)):
        print("Input:", end=" ")
        print(inputs[i])

        print("Prediction:", end=" ")
        print(cnn.predict(inputs[i]))

        # Add a blank line before the next print.
        if i != last:
            print()
    print("--------------------------------------------------------------------------------\n")


def main() -> None:
    """Create and train a CNN (Convolutional Neural Network). Print predictions on success."""
    use_stubs = not _use_real_impl()

    # CNN parameters.
    input_size  = 4
    kernel_size = 2
    pool_size   = 2
    dense_output = 1
    conv_func = ActFuncType.RELU
    dense_func = ActFuncType.TANH

    # Training parameters.
    epoch_count = 20000
    learning_rate = 0.01

    # Input data for training (digits 0 - 1).
    inputs = [
        [[1, 1, 1, 1],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [1, 1, 1, 1]],
        [[0, 1, 0, 0],
         [0, 1, 0, 0],
         [0, 1, 0, 0],
         [0, 1, 0, 0]]]
    # Output data for training (the corresponding numbers).
    outputs = [[0], [1]]

    # Create a machine learning factory.
    factory = ml_factory.create(use_stubs)

    # Create a CNN.
    cnn = Cnn(factory, input_size, kernel_size, conv_func, pool_size, dense_output, dense_func)

    # Train the network.
    success = cnn.train(inputs, outputs, epoch_count, learning_rate)

    # If training was successful, predict with the input matrices.
    if success:
        predict_and_print(cnn, inputs)
    else:
        print("Training failed!")


# Invoke the main function if this is the startup script.
if __name__ == "__main__":
    main()
