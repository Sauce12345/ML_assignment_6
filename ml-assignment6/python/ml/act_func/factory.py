"""Activation function factory."""
from ml.act_func.types import ActFuncType
from ml.act_func.relu import Relu
from ml.act_func.tanh import Tanh
from ml.act_func.identity import Identity
from ml.act_func.interface import IActFunc

def create(act_func_type: ActFuncType) -> IActFunc:
    """Create an activation function.

    Args:
        act_func_type: The type of activation function to create.

    Returns:
        The new activation function.
    """
    if act_func_type == ActFuncType.RELU:
        return Relu()
    if act_func_type == ActFuncType.TANH:
        return Tanh()
    if act_func_type == ActFuncType.IDENTITY:
        return Identity()
    raise ValueError(f"Invalid activation function {act_func_type}")
