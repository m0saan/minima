# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_autograd.ipynb.

# %% auto 0
__all__ = ['NDArray', 'LAZY_MODE', 'TENSOR_COUNTER', 'Value', 'Device', 'CPUDevice', 'cpu', 'all_devices', 'Operator', 'TensorOp',
           'Tensor']

# %% ../nbs/00_autograd.ipynb 3
from typing import (
    List,
    Optional,
    Tuple,
    Union,
    Set,
)

import numpy
import numpy as ARRAY_API
import minima as mi
import torch
from graphviz import Digraph

# %% ../nbs/00_autograd.ipynb 20
class Value:
    """
    Represents a node within a computational graph.

    This class encapsulates a single value and its relationships in the graph, making it easy to track and manage the value's dependencies, 
    the operation that produced it, and whether it requires a gradient for backpropagation. It's central to the functioning of automatic 
    differentiation within deep learning frameworks.

    Attributes:
        op (Operator)
        _prev (Set['Value']) 
        cached_data (NDArray)
        requires_grad (bool)
    """
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        return out



# %% ../nbs/00_autograd.ipynb 47
class Value:
    """
    Represents a node within a computational graph.

    This class encapsulates a single value and its relationships in the graph, making it easy to track and manage the value's dependencies, 
    the operation that produced it, and whether it requires a gradient for backpropagation. It's central to the functioning of automatic 
    differentiation within deep learning frameworks.

    Attributes:
        op (Operator)
        _prev (Set['Value']) 
        cached_data (NDArray)
        requires_grad (bool)
    """
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

# %% ../nbs/00_autograd.ipynb 67
class Value:
    """
    Represents a node within a computational graph.

    This class encapsulates a single value and its relationships in the graph, making it easy to track and manage the value's dependencies, 
    the operation that produced it, and whether it requires a gradient for backpropagation. It's central to the functioning of automatic 
    differentiation within deep learning frameworks.

    Attributes:
        op (Operator)
        _prev (Set['Value']) 
        cached_data (NDArray)
        requires_grad (bool)
    """
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

# %% ../nbs/00_autograd.ipynb 71
class Value:
    """
    A class representing a scalar value and its gradient in a computational graph.
    
    Attributes:
    - data (float): the scalar value associated with this node
    - grad (float): the gradient of the output of the computational graph w.r.t. this node's value
    - label (str): a label for this node, used for debugging and visualization purposes
    - _op (str): a string representation of the operation that produced this node in the computational graph
    - _prev (set of Value objects): the set of nodes that contributed to the computation of this node
    - _backward (function): a function that computes the gradients of this node w.r.t. its inputs
    
    Methods:
    - __init__(self, data, children=(), op='', label=''): Initializes a Value object with the given data, children, op, and label
    - __repr__(self): Returns a string representation of this Value object
    - __add__(self, other): Implements the addition operation between two Value objects
    - __mul__(self, other): Implements the multiplication operation between two Value objects
    - item(self): Returns the scalar value associated with this Value object
    - tanh(self): Applies the hyperbolic tangent function to this Value object and returns a new Value object
    """
    
    def __init__(
        self,
        data,
        children=(),
        op='',
        label=''
        ):
        """
        Initializes a Value object with the given data, children, op, and label.
        
        Args:
        - data (float): the scalar value associated with this node
        - children (tuple of Value objects): the nodes that contributed to the computation of this node
        - op (str): a string representation of the operation that produced this node in the computational graph
        - label (str): a label for this node, used for debugging and visualization purposes
        """
        
        self.data = data
        self._prev = set(children)
        self._op = op
        self.grad = 0.0
        self.label = label
        self._backward = lambda: None
        
    def __repr__(self):
        """
        Returns a string representation of this Value object.
        
        Returns:
        - str: a string representation of this Value object
        """
        return f"Value({self.data})"
    
    def __add__(self, other):
        """
        Implements the addition operation between two Value objects.
        
        Args:
        - other (Value): the other Value object to add to this one
        
        Returns:
        - Value: a new Value object representing the sum of this Value object and the other one
        """
        
        other = Value(other) if not isinstance(other, Value) else other
        
        out  = Value(self.data + other.data, children=(self, other), op='+')
        
        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
            
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        """
        Implements the addition operation between two Value objects.
        
        Args:
        - other (Value): the other Value object to add to this one
        
        Returns:
        - Value: a new Value object representing the sum of this Value object and the other one
        """
        return self.__add__(other)
    
    def __sub__(self, other):
        """
        Implements the subtraction operation between two Value objects.
        
        Args:
        - other (Value): the other Value object to subtract from this one
        
        Returns:
        - Value: a new Value object representing the difference between this Value object and the other one
        """
        return self + (-other)
    
    def __rsub__(self, other):
        """
        Implements the subtraction operation between two Value objects.
        
        Args:
        - other (Value): the other Value object to subtract from this one
        
        Returns:
        - Value: a new Value object representing the difference between this Value object and the other one
        """
        return self.__neg__().__add__(other)
    
    def __mul__(self, other):
        """
        Implements the multiplication operation between two Value objects.
        
        Args:
        - other (Value): the other Value object to multiply with this one
        
        Returns:
        - Value: a new Value object representing the product of this Value object and the other one
        """
        
        other = Value(other) if not isinstance(other, Value) else other
        out =  Value(self.data * other.data, children=(self, other), op='*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        """
        Implements the multiplication operation between two Value objects.
        
        Args:
        - other (Value): the other Value object to multiply with this one
        
        Returns:
        - Value: a new Value object representing the product of this Value object and the other one
        """
        return self.__mul__(other)
    
    def __neg__(self):
        """
        Implements the negation operation on this Value object.
        
        Returns:
        - Value: a new Value object representing the negation of this Value object
        """
        return self * -1
    
    
    def __pow__(self, other):
        """
        Implements the power operation between this Value object and another Value object or a scalar.
        
        Args:
        - other (Value or float): the other Value object or scalar to raise this Value object to
        
        Returns:
        - Value: a new Value object representing the power of this Value object and the other one
        """
        assert isinstance(other, (float, int)), "other must be a scalar"
        
        out = Value(self.data ** other, children=(self, ), op='**')
        
        def _backward(): self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward
        
        return out
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def exp(self):
        """
        Compute the exponential of a Value object's data attribute.

        Returns:
        - out: A new Value object that represents the exponential of the original Value object.
            This object stores a reference to the original Value object as a child node.

        Comments:
        - The exponential function is computed using the math.exp() function.
        - The backward pass is defined as a closure function _backward(), which computes the gradient of the original
        Value object using the product rule of differentiation and adds it to the gradient of the output object.
        - The _backward() function is assigned as an attribute to the output object for later use during backpropagation.
        """
        
        x = math.exp(self.data)
        out = Value(x, children=(self,), op='exp')
        
        def _backward(): self.grad += x * out.grad # x = exp(self.data) so x' = x (derivative of exp(x) is exp(x))
        out._backward = _backward
        
        return out
    
    def tanh(self):
        """
        Applies the hyperbolic tangent function to the data of this `Value` object and returns a new `Value` object 
        with the resulting data. This operation is an element-wise operation.
        """
        out = Value(torch.tanh(torch.tensor(self.data)), children=(self,), op='tanh')
        def _backward(): self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        
        return out
    
    def relu(self):
        """
        Applies the rectified linear unit function to the data of this `Value` object and returns a new `Value` object 
        with the resulting data. This operation is an element-wise operation.
        """
        out = Value(max(0, self.data), children=(self,), op='relu')
        
        def _backward(): self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        
        return out
    
    def item(self):
        """
        Return the scalar value being stored in the current Value as a Python float.
        
        Args:
            None
        
        Returns:
            float: The scalar value being stored in the current Value as a Python float.
        
        """
        return self.data
    
    
    def backward(self) -> None:
        """
        Performs backpropagation by computing the gradients of all variables with respect to the loss.

        This method sets the gradient of the computational graph's output variable to 1.0 and then computes
        the gradients of all variables in the graph by performing a reverse topological sort and calling the
        `_backward` method on each variable in the resulting order.

        Returns:
            None
        """
        # Set the gradient of the output variable to 1.0 to start the backpropagation process.
        self.grad = 1.0

        # Perform a reverse topological sort to determine the order in which to compute gradients.
        for v in self._topological_sort():
            # Call the `_backward` method on each variable in the resulting order to compute its gradient.
            v._backward()

        
    def _topological_sort(self):
        """
        Given a node in a computational graph, returns a list of all nodes in the graph sorted in topological order.

        Args:
            node: A node in a computational graph (i.e self).

        Returns:
            A list of all nodes in the graph sorted in topological order.
        """
        
        
        visited = set()
        topo = []

        def build_topo(node):
            visited.add(node)
            for child in node._prev:
                if child not in visited:
                    build_topo(child)
            topo.append(node)

        build_topo(self)
        topo.reverse()
        return topo    
        


# %% ../nbs/00_autograd.ipynb 72
NDArray = numpy.ndarray
LAZY_MODE = False
TENSOR_COUNTER = 0

# %% ../nbs/00_autograd.ipynb 73
class Device:
    """Indicates the device supporting an NDArray."""


class CPUDevice(Device):
    """Represents data that sits in CPU"""

    def __repr__(self):
        return "needle.cpu()"

    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, CPUDevice)

    def enabled(self):
        return True

def cpu():
    """Return cpu device"""
    return CPUDevice()

def all_devices():
    """return a list of all available devices"""
    return [cpu()]

# %% ../nbs/00_autograd.ipynb 74
class Operator:
    
    def __call__(self, *args):
        raise NotImplementedError()
        
    def compute(self, *args: Tuple[NDArray]):
        raise NotImplementedError()
        
    def gradient(self, out_grad: 'Value', node: 'Value') -> Union['Value', Tuple['Value']]:
        raise NotImplementedError()

# %% ../nbs/00_autograd.ipynb 75
class TensorOp(Operator):
    """ Op class specialized to output tensors, will be alternate subclasses for other structures """

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)

# %% ../nbs/00_autograd.ipynb 77
class Tensor(Value):
    """A value in the computational graph."""

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                data = array.realize_data()
            else:
                # fall back, copy through numpy conversion
                data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else cpu()
            data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(None, set(), data=data, requires_grad=requires_grad, )
        
    def __repr__(self):
        return "minima.Tensor(" + str(self.realize_data()) + ")"

    def __str__(self):
        return self.realize_data().__str__()
        
    def _init(
        self,
        op: Optional[Operator],
        children: Set["Tensor"],
        *,
        num_outputs: int = 1,
        data: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(child.requires_grad for child in children)
        self._op = op
        self.data = data
        self.children = children
        self.num_outputs = num_outputs
        self.requires_grad = requires_grad
        
    def realize_data(self):
        if self.data is None:
            self.data = self._op.compute(*[child.realize_data() for child in self.children])
        return self.data
    
    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        if ARRAY_API is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return ARRAY_API.array(numpy_array, device=device, dtype=dtype)
    
    @staticmethod
    def make_from_op(op: Operator, children: Tuple["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, children)
        if not LAZY_MODE:
            tensor.realize_data()
        return tensor
    
    @property
    def shape(self):
        return self.realize_data().shape

    @property
    def dtype(self):
        return self.realize_data().dtype
    
    def __add__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """
        Implements the addition operation between two Tensors or a Tensor and a scalar.

        Args:
        - other (Tensor or scalar): the other Tensor or scalar to add to this one

        Returns:
        - Tensor: a new Tensor object representing the sum of this Tensor and the other one
        """
        if isinstance(other, Tensor):
            # Ensure both tensors have the same shape for addition
            if self.shape != other.shape:
                raise AssertionError(f"Tensors must be of the same shape for addition. Got {self.shape} and {other.shape}.")

            return mi.operators.EWiseAdd()(self, other)

        elif isinstance(other, (int, float)):
            return mi.operators.AddScalar(other)(self)

        else:
            raise ValueError(f"Unsupported operand type for +: '{type(self).__name__}' and '{type(other).__name__}'")
            
    def __mul__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """
        Implements the multiplication operation between two Tensors or a Tensor and a scalar.

        Args:
        - other (Tensor or scalar): the other Tensor or scalar to multiply with this one

        Returns:
        - Tensor: a new Tensor object representing the product of this Tensor and the other one
        """
        if isinstance(other, Tensor):
            # Ensure both tensors have the same shape for multiplication
            if self.shape != other.shape:
                raise AssertionError(f"Tensors must be of the same shape for multiplication. Got {self.shape} and {other.shape}.")

            return mi.operators.EWiseMul()(self, other)

        elif isinstance(other, (int, float)):
            return mi.operators.MulScalar(other)(self)

        else:
            raise ValueError(f"Unsupported operand type for *: '{type(self).__name__}' and '{type(other).__name__}'")

