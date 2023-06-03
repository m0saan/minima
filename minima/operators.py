# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_operators.ipynb.

# %% auto 0
__all__ = ['EWiseAdd', 'add', 'AddScalar', 'add_scalar', 'EWiseMul', 'multiply', 'MulScalar', 'mul_scalar', 'EWiseDiv', 'divide',
           'DivScalar', 'divide_scalar', 'Negate', 'negate', 'Exp', 'exp', 'ReLU', 'relu', 'PowerScalar',
           'power_scalar', 'Transpose', 'transpose', 'Reshape', 'reshape', 'MatMul', 'matmul', 'Summation', 'summation',
           'BroadcastTo', 'broadcast_to']

# %% ../nbs/01_operators.ipynb 2
"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Operator, Tensor, Value, TensorOp, Tuple, Union
from collections import namedtuple
from typing import NamedTuple
import numpy
import torch

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as ARRAY_API

# %% ../nbs/01_operators.ipynb 8
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

# %% ../nbs/01_operators.ipynb 23
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

# %% ../nbs/01_operators.ipynb 26
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
        a, b = node.children
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

# %% ../nbs/01_operators.ipynb 29
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

# %% ../nbs/01_operators.ipynb 32
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
        a, b = node.children
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


# %% ../nbs/01_operators.ipynb 35
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

# %% ../nbs/01_operators.ipynb 38
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

# %% ../nbs/01_operators.ipynb 41
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
        self.out = ARRAY_API.exp(a)
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

# %% ../nbs/01_operators.ipynb 44
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
        self.out = ARRAY_API.clip(a, a_min=0)
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


# %% ../nbs/01_operators.ipynb 47
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
        return ARRAY_API.power(a, self.scalar)

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

# %% ../nbs/01_operators.ipynb 53
class Transpose(TensorOp):
    """
    Tensor operation class that performs transposition of a tensor along specified axes.
    
    If no axes are specified, it swaps the last two dimensions of the input tensor.

    Example:
        >>> a = Tensor(np.arange(1, 7).reshape(2, 3))
        >>> op = Transpose()
        >>> result = op.compute(a.data)
        >>> print(result)
        array([[1, 4],
               [2, 5],
               [3, 6]])
    """
    def __init__(self, axes: Optional[tuple] = None):
        """
        Initialize the operation with the specified axes.

        Args:
            axes (Optional[tuple]): The pair of axes that should be swapped. If not provided, the last two axes are swapped.
        """
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        """
        Perform the transpose operation.

        Args:
            a (NDArray): The input tensor.

        Returns:
            NDArray: The transposed tensor.
        """

        if self.axes:
            a = a.swapaxes(self.axes[0], self.axes[1])
        else:
            a = a.swapaxes(-2, -1)
        return a

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, ...]:
        """
        Compute the gradient of the transpose operation.

        Args:
            out_grad (Tensor): The gradient of the output tensor.
            node (Tensor): The node in the computational graph where the operation was performed.

        Returns:
            Tuple[Tensor, ...]: The gradient with respect to the input tensor.
        """
        return (transpose(out_grad, axes=self.axes), )

def transpose(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    """
    Perform the transpose operation on the input tensor along the specified axes.
    If no axes are specified, it swaps the last two dimensions of the input tensor.

    Args:
        a (Tensor): The input tensor.
        axes (Optional[tuple]): The pair of axes that should be swapped. If not provided, the last two axes are swapped.

    Returns:
        Tensor: The transposed tensor.

    Example:
        >>> a = Tensor(np.arange(1, 7).reshape(2, 3))
        >>> result = transpose(a)
        >>> print(result)
        Tensor([[1, 4],
                [2, 5],
                [3, 6]])
    """
    return Transpose(axes)(a)


# %% ../nbs/01_operators.ipynb 56
class Reshape(TensorOp):
    """
    Tensor operation class that reshapes a tensor.

    Example:
        >>> a = Tensor([1, 2, 3, 4, 5, 6])
        >>> op = Reshape((2, 3))
        >>> result = op.compute(a)
        >>> print(result)
        Tensor([[1, 2, 3],
                 [4, 5, 6]])
    """
    def __init__(self, shape: Tuple[int, ...]):
        """
        Initialize the operation with the target shape.

        Args:
            shape (Tuple[int, ...]): The desired shape of the output tensor.
        """
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        """
        Perform the reshape operation.

        Args:
            a (NDArray): The input tensor.

        Returns:
            NDArray: The reshaped tensor.
        """
        return ARRAY_API.reshape(a, newshape=self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, ...]:
        """
        Compute the gradient of the reshape operation.

        Args:
            out_grad (Tensor): The gradient of the output tensor.
            node (Tensor): The node in the computational graph where the operation was performed.

        Returns:
            Tuple[Tensor, ...]: The gradient with respect to the input tensor.
        """
        input_shape = node.children[0].shape
        return reshape(out_grad, input_shape), 

def reshape(a: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """
    Reshape the input tensor to the specified shape.

    Args:
        a (Tensor): The input tensor.
        shape (Tuple[int, ...]): The desired shape of the output tensor.

    Returns:
        Tensor: The reshaped tensor.

    Example:
        >>> a = Tensor([1, 2, 3, 4, 5, 6])
        >>> result = reshape(a, (2, 3))
        >>> print(result)
        Tensor([[1, 2, 3],
                 [4, 5, 6]])
    """
    return Reshape(shape)(a)


# %% ../nbs/01_operators.ipynb 67
class MatMul(TensorOp):
    """
    Tensor operation class that performs matrix multiplication.

    Example:
        >>> a = Tensor([[1, 2], [3, 4]])
        >>> b = Tensor([[5, 6], [7, 8]])
        >>> op = MatMul()
        >>> result = op.compute(a, b)
        >>> print(result)
        Tensor([[19, 22],
                 [43, 50]])
    """
    
    
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        """
        Perform the matrix multiplication operation.

        Args:
            a (NDArray): The first input tensor.
            b (NDArray): The second input tensor.

        Returns:
            NDArray: The product of a and b.
        """
        return ARRAY_API.matmul(a, b)

    
    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute the gradient of the matrix multiplication operation.

        Args:
            out_grad (Tensor): The gradient of the output tensor.
            node (Tensor): The node in the computational graph where the operation was performed.

        Returns:
            Tuple[Tensor, Tensor]: The gradients with respect to the input tensors.
        """
        a, b = node.children
        out_shape, a_shape, b_shape = out_grad.shape, a.shape, b.shape
        
        # Compute the gradient with respect to a
        if len(a_shape) == len(out_shape):
            # If a and the output have the same dimensionality, we perform a matrix multiplication
            # between the output gradient and the transpose of b
            grad_wrt_a = matmul(out_grad, transpose(b))
        else:
            # If a has fewer dimensions than the output, we sum over the extra dimensions in the output
            axes_to_sum_over = tuple(range(len(out_shape) - len(a_shape)))
            grad_wrt_a = summation(matmul(out_grad, transpose(b)), axes=axes_to_sum_over)

        # Compute the gradient with respect to b
        if len(b_shape) == len(out_shape):
            # If b and the output have the same dimensionality, we perform a matrix multiplication
            # between the transpose of a and the output gradient
            grad_wrt_b = matmul(transpose(a), out_grad)
        else:
            # If b has fewer dimensions than the output, we sum over the extra dimensions in the output
            axes_to_sum_over = tuple(range(len(out_shape) - len(b_shape)))
            grad_wrt_b = summation(matmul(transpose(a), out_grad), axes=axes_to_sum_over)

        return grad_wrt_a, grad_wrt_b


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Perform matrix multiplication on two tensors.

    Args:
        a (Tensor): The first input tensor.
        b (Tensor): The second input tensor.

    Returns:
        Tensor: The product of a and b.

    Example:
        >>> a = Tensor([[1, 2], [3, 4]])
        >>> b = Tensor([[5, 6], [7, 8]])
        >>> result = matmul(a, b)
        >>> print(result)
        Tensor([[19, 22],
                 [43, 50]])
    """
    return MatMul()(a, b)


# %% ../nbs/01_operators.ipynb 76
class Summation(TensorOp):
    """
    Op to compute the sum of a tensor along specified axes.

    Example:
    >>> a = Tensor([[1, 2, 3], [4, 5, 6]])
    >>> op = Summation(axes=(0,))
    >>> result = op.compute(a)
    >>> print(result)
    Tensor([5, 7, 9])

    Args:
    - axes (tuple, optional): The dimensions to reduce. If `None` (default), reduces all dimensions.

    Methods:
    - compute(a: NDArray) -> NDArray: Computes the sum of `a` along the specified axes.
    - gradient(out_grad: Tensor, node: Tensor) -> Tuple[Tensor]: Computes the gradient of the sum operation.
    """
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        """
        Computes the sum of `a` along the specified axes.

        Args:
        - a: The input tensor.

        Returns:
        The sum of `a` along the specified axes.
        """
        return ARRAY_API.sum(a, self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        """
        Computes the gradient of the sum operation.

        Args:
        - out_grad: The gradient of the output of the operation.
        - node: The node in the computational graph where the operation was performed.

        Returns:
        The gradient with respect to the input.
        """
        # out_grad is the gradient of the output of this operation
        # We need to "undo" the dimensionality reduction performed in the forward pass
        # That's why we create a new shape, replacing the dimensions specified by self.axes with 1

        # Initialize new shape to be the same as the input shape
        new_shape = list(node.children[0].shape)

        # If axes were specified, set those dimensions to 1 in the new shape
        if self.axes:
            for axis in self.axes: new_shape[axis] = 1
            
        else:
            new_shape = [1] * len(new_shape)

        # Reshape out_grad to the new shape
        reshaped_grad = reshape(out_grad, new_shape)

        # Broadcast the reshaped out_grad to match the input shape
        broadcasted_grad = broadcast_to(reshaped_grad, node.children[0].shape)

        # The gradient method needs to return a tuple, even though there's only one input
        return (broadcasted_grad,)


def summation(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    """
    Computes the sum of `a` along the specified axes.

    Args:
    - a: The input tensor.
    - axes (tuple, optional): The dimensions to reduce. If `None` (default), reduces all dimensions.

    Returns:
    The sum of `a` along the specified axes.
    """
    return Summation(axes)(a)


# %% ../nbs/01_operators.ipynb 89
class BroadcastTo(TensorOp):
    """
    Op to broadcast a tensor to a new shape.

    Example:
    >>> a = Tensor([1, 2, 3])
    >>> op = BroadcastTo((3, 3))
    >>> result = op.compute(a)
    >>> print(result)
    Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

    Args:
    - shape (tuple): The new shape to broadcast the input tensor to.

    Methods:
    - compute(a: NDArray) -> NDArray: Broadcasts `a` to the specified shape.
    - gradient(out_grad: Tensor, node: Tensor) -> Tuple[Tensor]: Computes the gradient of the broadcast operation.
    """
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        """
        Broadcasts `a` to the specified shape.

        Args:
        - a: The input tensor.

        Returns:
        The tensor `a` broadcasted to the specified shape.
        """
        return ARRAY_API.broadcast_to(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        """
        Computes the gradient of the broadcast operation.

        Args:
        - out_grad: The gradient of the output of the operation.
        - node: The node in the computational graph where the operation was performed.

        Returns:
        The gradient with respect to the input.
        """
        # First, we need to create a shape that matches the shape of `a` but with ones 
        # prepended to match the length of `self.shape`.
        a_shape = node.children[0].shape
        shape = [1] * (len(self.shape) - len(a_shape)) + list(a_shape)

        # Then, we identify the dimensions along which to sum in the backward pass. 
        # These are the dimensions that were expanded during the broadcast.
        sum_over = tuple([idx for idx in range(len(self.shape)) if self.shape[idx] != shape[idx]])

        # Finally, we reshape the gradient after summing over the appropriate dimensions to match `a`'s shape.
        return reshape(summation(out_grad, sum_over), a_shape)

def broadcast_to(a: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """
    Broadcasts `a` to the specified shape.

    Args:
    - a: The input tensor.
    - shape: The new shape to broadcast the input tensor to.

    Returns:
    The tensor `a` broadcasted to the specified shape.
    """
    return BroadcastTo(shape)(a)

