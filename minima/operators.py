# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_operators.ipynb.

# %% auto 0
__all__ = ['EWiseAdd', 'add', 'AddScalar', 'add_scalar', 'EWiseMul', 'multiply', 'MulScalar', 'mul_scalar', 'Negate', 'negate',
           'Exp', 'exp', 'ReLU', 'relu', 'PowerScalar', 'power_scalar', 'EWiseDiv', 'divide', 'DivScalar',
           'divide_scalar']

# %% ../nbs/01_operators.ipynb 2
"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Operator, Tensor, Value, TensorOp, Tuple, Union
from collections import namedtuple
from typing import NamedTuple
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as ARRAY_API

# %% ../nbs/01_operators.ipynb 7
class EWiseAdd(TensorOp):
    """
    Performs element-wise addition of two tensors.

    Example:
    >>> a = Tensor([1, 2, 3])
    >>> b = Tensor([4, 5, 6])
    >>> op = EWiseAdd()
    >>> result = op.compute(a, b)
    >>> print(result)
    Tensor([5, 7, 9])
    """
    
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        """
        Computes the element-wise sum of two tensors.

        Args:
        - a: The first tensor.
        - b: The second tensor.

        Returns:
        The element-wise sum of a and b.
        """
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes the gradient of the element-wise addition operation.

        Args:
        - out_grad: The gradient of the output of the operation.
        - node: The node in the computational graph where the operation was performed.

        Returns:
        The gradients with respect to the inputs.
        """
        return (out_grad, out_grad)

def add(a: Tensor, b: Tensor) -> Tensor:
    """
    Adds two tensors element-wise.

    Args:
    - a: The first tensor.
    - b: The second tensor.

    Returns:
    The element-wise sum of a and b.
    """
    return EWiseAdd()(a, b)

# %% ../nbs/01_operators.ipynb 22
class AddScalar(TensorOp):
    """
    Performs addition of a tensor and a scalar.

    Example:
    >>> a = Tensor([1, 2, 3])
    >>> op = AddScalar(5)
    >>> result = op.compute(a)
    >>> print(result)
    Tensor([6, 7, 8])
    """
    def __init__(self, scalar: Union[int, float]):
        """
        Initializes the operation with a scalar.

        Args:
        - scalar: The scalar to add to the tensor.
        """
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        """
        Computes the sum of a tensor and a scalar.

        Args:
        - a: The tensor.

        Returns:
        The sum of a and the scalar.
        """
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        """
        Computes the gradient of the addition operation.

        Args:
        - out_grad: The gradient of the output of the operation.
        - node: The node in the computational graph where the operation was performed.

        Returns:
        The gradient with respect to the input.
        """
        return (out_grad, )

def add_scalar(a: Tensor, scalar: Union[int, float]) -> Tensor:
    """
    Adds a scalar to a tensor.

    Args:
    - a: The tensor.
    - scalar: The scalar to add.

    Returns:
    The sum of a and the scalar.
    """
    return AddScalar(scalar)(a)

# %% ../nbs/01_operators.ipynb 25
class EWiseMul(TensorOp):
    """
    Performs element-wise multiplication of two tensors.

    Example:
    >>> a = Tensor([1, 2, 3])
    >>> b = Tensor([4, 5, 6])
    >>> op = EWiseMul()
    >>> result = op.compute(a, b)
    >>> print(result)
    Tensor([4, 10, 18])
    """
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        """
        Computes the element-wise product of two tensors.

        Args:
        - a: The first tensor.
        - b: The second tensor.

        Returns:
        The element-wise product of a and b.
        """
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes the gradient of the element-wise multiplication operation.

        Args:
        - out_grad: The gradient of the output of the operation.
        - node: The node in the computational graph where the operation was performed.

        Returns:
        The gradients with respect to the inputs.
        """
        a, b = node.inputs
        return out_grad * b, out_grad * a

def multiply(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiplies two tensors element-wise.

    Args:
    - a: The first tensor.
    - b: The second tensor.

    Returns:
    The element-wise product of a and b.
    """
    return EWiseMul()(a, b)

# %% ../nbs/01_operators.ipynb 28
class MulScalar(TensorOp):
    """
    Performs multiplication of a tensor and a scalar.

    Example:
    >>> a = Tensor([1, 2, 3])
    >>> op = MulScalar(5)
    >>> result = op.compute(a)
    >>> print(result)
    Tensor([5, 10, 15])
    """
    def __init__(self, scalar: Union[int, float]):
        """
        Initializes the operation with a scalar.

        Args:
        - scalar: The scalar to multiply the tensor with.
        """
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        """
        Computes the product of a tensor and a scalar.

        Args:
        - a: The tensor.

        Returns:
        The product of a and the scalar.
        """
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        """
        Computes the gradient of the multiplication operation.

        Args:
        - out_grad: The gradient of the output of the operation.
        - node: The node in the computational graph where the operation was performed.

        Returns:
        The gradient with respect to the input.
        """
        return (out_grad * self.scalar, )
    
def mul_scalar(a: Tensor, scalar: Union[int, float]) -> Tensor:
    """
    Multiplies a tensor by a scalar.

    Args:
    - a: The tensor.
    - scalar: The scalar to multiply.

    Returns:
    The product of a and the scalar.
    """
    return MulScalar(scalar)(a)

# %% ../nbs/01_operators.ipynb 31
class Negate(TensorOp):
    """
    Negates the given tensor.
    
    Example:
    >>> a = Tensor([1, -2, 3])
    >>> op = Negate()
    >>> result = op.compute(a)
    >>> print(result)
    Tensor([-1, 2, -3])
    """
    
    def compute(self, a: NDArray) -> NDArray:
        """
        Computes the negation of a tensor.

        Args:
        - a: The tensor to negate.

        Returns:
        The negation of a.
        """
        return -1 * a

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor,]:
        """
        Computes the gradient of the negation operation.

        Args:
        - out_grad: The gradient of the output of the operation.
        - node: The node in the computational graph where the operation was performed.

        Returns:
        The gradients with respect to the inputs.
        """
        return (negate(out_grad), )


def negate(a: Tensor) -> Tensor:
    """
    Negates the given tensor.

    Args:
    - a: The tensor to negate.

    Returns:
    The negation of a.
    
    Example:
    >>> a = Tensor([1, -2, 3])
    >>> result = negate(a)
    >>> print(result)
    Tensor([-1, 2, -3])
    """
    return Negate()(a)

# %% ../nbs/01_operators.ipynb 34
class Exp(TensorOp):
    """
    Calculates the exponential of the given tensor.
    
    Example:
    >>> a = Tensor([1, 2, 3])
    >>> op = Exp()
    >>> result = op.compute(a)
    >>> print(result)
    Tensor([2.71828183, 7.3890561, 20.08553692])
    """
    
    def compute(self, a: NDArray) -> NDArray:
        """
        Computes the exponential of a tensor.

        Args:
        - a: The tensor.

        Returns:
        The exponential of a.
        """
        self.out = array_api.exp(a)
        return self.out

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor,]:
        """
        Computes the gradient of the exponential operation.

        Args:
        - out_grad: The gradient of the output of the operation.
        - node: The node in the computational graph where the operation was performed.

        Returns:
        The gradients with respect to the inputs.
        """
        return (out_grad * self.out, )

def exp(a: Tensor) -> Tensor:
    """
    Calculates the exponential of the given tensor.

    Args:
    - a: The tensor.

    Returns:
    The exponential of a.
    
    Example:
    >>> a = Tensor([1, 2, 3])
    >>> result = exp(a)
    >>> print(result)
    Tensor([2.71828183, 7.3890561, 20.08553692])
    """
    return Exp()(a)

# %% ../nbs/01_operators.ipynb 37
class ReLU(TensorOp):
    """
    Applies the ReLU (Rectified Linear Unit) activation function to the given tensor.
    
    Example:
    >>> a = Tensor([1, -2, 3])
    >>> op = ReLU()
    >>> result = op.compute(a)
    >>> print(result)
    Tensor([1, 0, 3])
    """
    
    def compute(self, a: NDArray) -> NDArray:
        """
        Computes the ReLU activation function on a tensor.

        Args:
        - a: The tensor.

        Returns:
        The result of applying ReLU to a.
        """
        self.out = array_api.clip(a, a_min=0)
        return self.out

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor,]:
        """
        Computes the gradient of the ReLU operation.

        Args:
        - out_grad: The gradient of the output of the operation.
        - node: The node in the computational graph where the operation was performed.

        Returns:
        The gradients with respect to the inputs.
        """
        return (out_grad * Tensor(node.children[0] >= 0), )

def relu(a: Tensor) -> Tensor:
    """
    Applies the ReLU (Rectified Linear Unit) activation function to the given tensor.

    Args:
    - a: The tensor.

    Returns:
    The result of applying ReLU to a.
    
    Example:
    >>> a = Tensor([1, -2, 3])
    >>> result = relu(a)
    >>> print(result)
    Tensor([1, 0, 3])
    """
    return ReLU()(a)


# %% ../nbs/01_operators.ipynb 40
class PowerScalar(TensorOp):
    """
    The PowerScalar operation raises a tensor to an (integer) power.

    Attributes:
        scalar (int): The power to raise the tensor to.

    Example:
        >>> import numpy as np
        >>> tensor = Tensor(np.array([1, 2, 3]))
        >>> pow_scalar = PowerScalar(2)
        >>> result = pow_scalar.compute(tensor.data)
        >>> print(result)
        array([1, 4, 9])

    """

    def __init__(self, scalar: int):
        """
        Constructs the PowerScalar operation.

        Args:
            scalar (int): The power to raise the tensor to.
        """
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        """
        Computes the power operation on the input tensor.

        Args:
            a (NDArray): The input tensor.

        Returns:
            NDArray: The resulting tensor after the power operation.
        """
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, ]:
        """
        Computes the gradient of the power operation.

        Args:
            out_grad (Tensor): The gradient of the output tensor.
            node (Tensor): The node in the computational graph where the operation was performed.

        Returns:
            Tuple[Tensor, ]: The gradient with respect to the input tensor.
        """
        a = node.children[0]
        return (self.scalar * power_scalar(a, self.scalar - 1) * out_grad, )


def power_scalar(a: Tensor, scalar: int) -> Tensor:
    """
    Raises a tensor to a power.

    Args:
        a (Tensor): The input tensor.
        scalar (int): The power to raise the tensor to.

    Returns:
        Tensor: The resulting tensor after the power operation.

    Example:
        >>> import numpy as np
        >>> tensor = Tensor(np.array([1, 2, 3]))
        >>> result = power_scalar(tensor, 2)
        >>> print(result)
        Tensor([1, 4, 9])
    """
    return PowerScalar(scalar)(a)

# %% ../nbs/01_operators.ipynb 43
class EWiseDiv(TensorOp):
    """
    The EWiseDiv operation divides two tensors element-wise.

    Example:
        >>> import numpy as np
        >>> a = Tensor(np.array([1, 2, 3]))
        >>> b = Tensor(np.array([4, 5, 6]))
        >>> div = EWiseDiv()
        >>> result = div.compute(a.data, b.data)
        >>> print(result)
        array([0.25, 0.4, 0.5])

    """

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        """
        Computes the element-wise division of two tensors.

        Args:
            a (NDArray): The dividend tensor.
            b (NDArray): The divisor tensor.

        Returns:
            NDArray: The resulting tensor after element-wise division.
        """
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes the gradient of the element-wise division operation.

        Args:
            out_grad (Tensor): The gradient of the output tensor.
            node (Tensor): The node in the computational graph where the operation was performed.

        Returns:
            Tuple[Tensor, Tensor]: The gradients with respect to the dividend and divisor tensors.
        """
        a, b = node.inputs
        return divide(out_grad, b), out_grad * negate(divide(a, power_scalar(b, 2)))


def divide(a: Tensor, b: Tensor) -> Tensor:
    """
    Divides two tensors element-wise.

    Args:
        a (Tensor): The dividend tensor.
        b (Tensor): The divisor tensor.

    Returns:
        Tensor: The resulting tensor after element-wise division.

    Example:
        >>> import numpy as np
        >>> a = Tensor(np.array([1, 2, 3]))
        >>> b = Tensor(np.array([4, 5, 6]))
        >>> result = divide(a, b)
        >>> print(result)
        Tensor([0.25, 0.4, 0.5])
    """
    return EWiseDiv()(a, b)


# %% ../nbs/01_operators.ipynb 46
class DivScalar(TensorOp):
    """
    The DivScalar operation divides a tensor by a scalar.

    Example:
        >>> import numpy as np
        >>> a = Tensor(np.array([1, 2, 3]))
        >>> scalar = 2
        >>> div_scalar = DivScalar(scalar)
        >>> result = div_scalar.compute(a.data)
        >>> print(result)
        array([0.5, 1.0, 1.5])

    """

    def __init__(self, scalar: Union[int, float]):
        """
        Initialize the DivScalar operation with the scalar to divide by.

        Args:
            scalar (int, float): The scalar to divide the tensor by.
        """
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        """
        Divides the tensor by the scalar.

        Args:
            a (NDArray): The tensor to divide.

        Returns:
            NDArray: The resulting tensor after division.
        """
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, ...]:
        """
        Computes the gradient of the division operation.

        Args:
            out_grad (Tensor): The gradient of the output tensor.
            node (Tensor): The node in the computational graph where the operation was performed.

        Returns:
            Tuple[Tensor, ...]: The gradient with respect to the tensor.
        """
        return (out_grad / self.scalar, )

def divide_scalar(a: Tensor, scalar: Union[int, float]) -> Tensor:
    """
    Divides a tensor by a scalar.

    Args:
        a (Tensor): The tensor to divide.
        scalar (int, float): The scalar to divide the tensor by.

    Returns:
        Tensor: The resulting tensor after division.

    Example:
        >>> import numpy as np
        >>> a = Tensor(np.array([1, 2, 3]))
        >>> scalar = 2
        >>> result = divide_scalar(a, scalar)
        >>> print(result)
        Tensor([0.5, 1.0, 1.5])
    """
    return DivScalar(scalar)(a)
