"""Machine learning utility functions."""
from enum import Enum
from ml.types import Matrix1d, Matrix2d

# pylint: disable=consider-using-enumerate

class LearningRate(Enum):
    """Enumeration of learning rate limits."""
    MIN = 1e-10 # Minimum learning rate (non-inclusive).
    MAX = 1.0   # Maximum learning rate (inclusive).


def create_matrix1d(size: int = 0) -> Matrix1d:
    """Create one-dimensional matrix with zeros.

    Args: 
        size: The desired size of the matrix (default = 0).
        
    Returns:
        The new matrix.
    """
    return [0.0] * size


def create_matrix2d(row_count: int = 0, col_count: int | None = None) -> Matrix2d:
    """Create two-dimensional matrix with zeros.

    Args: 
        row_count: The desired row count of the matrix (default = 0).
        col_count: The desired column count of the matrix (default = same as the row count).
        
    Returns:
        The new matrix.
    """
    if row_count < 0:
        raise ValueError("Row count cannot be negative!")
    rows = row_count
    cols = col_count if col_count is not None else row_count
    if cols < 0:
        raise ValueError("Column count cannot be negative!")
    return [[0.0] * cols for _ in range(rows)]


def init_matrix1d(matrix: Matrix1d) -> None:
    """Initialize one-dimensional matrix with zeros.

    Args: 
        matrix: The matrix to initialize.
    """
    for _ in range(len(matrix)):
        matrix.append(0.0)


def init_matrix2d(matrix: Matrix2d) -> None:
    """Initialize two-dimensional matrix with zeros.

    Args: 
        matrix: The matrix to initialize.
    """
    for row in range(len(matrix)):
        for col in range(row):
            matrix[row][col] = 0.0


def is_matrix2d_square(matrix: Matrix2d, op_name: str | None = None) -> bool:
    """Check whether given matrix is square. Print an error message if not.
    
    Args:
        matrix: The matrix to check.
        op_name: Operation name (default = none).
    Returns:
        True if given matrix is square, false otherwise.
    """
    for row in matrix:
        if len(row) != len(matrix):
            if op_name is not None:
                print(f"Cannot perform {op_name} due to matrix not being square!")
            else:
                print("Matrix is not square!")
            return False
    return True


def match_dimensions(expected_size: int, actual_size: int, op_name: str | None = None) -> bool:
    """Match dimensions. Print an error message on mismatch.
    
    Args:
        expected_size: Expected size.
        actual_size: Actual size.
        op_name: Operation name (default = none).

    Returns:
        True if the dimensions match, false otherwise.
    """
    if expected_size == actual_size:
        return True
    if op_name is not None:
        print(f"Cannot perform {op_name} due to dimension mismatch: "
            " expected {expected_size}, actual is {actual_size}!")
    else:
        print(f"Dimension mismatch: expected {expected_size}, actual is {actual_size}!")
    return False


def check_learning_rate(learning_rate: float, op_name: str | None = None) -> bool:
    """Check given learning rate. Print an error message if invalid.

    Args:
        learning_rate: Learning rate. Must be in range [1e-10, 1.0] to be considered valid.
        op_name: Operation name (default = none).

    Returns:
        True if given learning rate is valid, false otherwise.
    """
    lr_min   = LearningRate.MIN.value
    lr_max   = LearningRate.MAX.value
    lr_valid = lr_min <= learning_rate <= lr_max

    if not lr_valid:
        range_msg = f"the valid range is [{lr_min}, {lr_max}]!"
        if op_name is not None:
            print(f"Cannot perform {op_name} due to invalid learning rate {learning_rate}: " \
                  f"{range_msg}")
        else:
            print(f"Invalid learning rate {learning_rate}: {range_msg}")
    return lr_valid


def check_epoch_count(epoch_count: float, op_name: str | None = None) -> bool:
    """Check given epoch count. Print an error message if invalid.

    Args:
        epoch_count: Epoch count. Must be greater than 0 to be considered valid.
        op_name: Operation name (default = none).

    Returns:
        True if given epoch count is valid, false otherwise.
    """
    valid = epoch_count > 0
    if not valid:
        if op_name is not None:
            print(f"Cannot perform {op_name} due to invalid epoch count: "
                   "the value must be greater than 0!")
        else:
            print("Invalid epoch count: the value must be greater than 0!")
    return valid


def check_train_set_count(set_count: float, op_name: str | None = None) -> bool:
    """Check given training set count. Print an error message if invalid.

    Args:
        set_count: Training set count. Must be greater than 0 to be considered valid.
        op_name: Operation name (default = none).

    Returns:
        True if given training set count is valid, false otherwise.
    """
    valid = set_count > 0
    if not valid:
        if op_name is not None:
            print(f"Cannot perform {op_name} due to invalid training set count: "
                   "the value must be greater than 0!")
        else:
            print("Invalid training set count: the value must be greater than 0!")
    return valid
