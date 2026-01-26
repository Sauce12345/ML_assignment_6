"""Activation function types."""
from enum import IntEnum

class ActFuncType(IntEnum):
    """Enumeration of activation function types."""
    RELU     = 0 # ReLU (Rectified Linear Unit).
    TANH     = 1 # Hyperbolic tangent.
    IDENTITY = 2 # Identity/no activation function.
