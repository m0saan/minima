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
numpy.set_printoptions(precision=6, linewidth=160)
# from graphviz import Digraph

# %% ../nbs/00_autograd.ipynb 70
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
        
        self._data = data
        self.children = set(children)
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
        return f"Value({self._data})"
    
    def __add__(self, other):
        """
        Implements the addition operation between two Value objects.
        
        Args:
        - other (Value): the other Value object to add to this one
        
        Returns:
        - Value: a new Value object representing the sum of this Value object and the other one
        """
        
        other = Value(other) if not isinstance(other, Value) else other
        
        out  = Value(self._data + other._data, children=(self, other), op='+')
        
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
        out =  Value(self._data * other._data, children=(self, other), op='*')
        
        def _backward():
            self.grad += other._data * out.grad
            other.grad += self._data * out.grad
            
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
        
        out = Value(self._data ** other, children=(self, ), op='**')
        
        def _backward(): self.grad += other * self._data ** (other - 1) * out.grad
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
        
        x = math.exp(self._data)
        out = Value(x, children=(self,), op='exp')
        
        def _backward(): self.grad += x * out.grad # x = exp(self._data) so x' = x (derivative of exp(x) is exp(x))
        out._backward = _backward
        
        return out
    
    def tanh(self):
        """
        Applies the hyperbolic tangent function to the data of this `Value` object and returns a new `Value` object 
        with the resulting data. This operation is an element-wise operation.
        """
        out = Value(torch.tanh(torch.tensor(self._data)), children=(self,), op='tanh')
        def _backward(): self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        
        return out
    
    def relu(self):
        """
        Applies the rectified linear unit function to the data of this `Value` object and returns a new `Value` object 
        with the resulting data. This operation is an element-wise operation.
        """
        out = Value(max(0, self._data), children=(self,), op='relu')
        
        def _backward(): self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        
        return out
    
    @property
    def data(self):
        """
        Returns a tensor that shares the data with the current tensor but is detached from the computational graph.

        Example:
        >>> t = Tensor([1, 2, 3], requires_grad=True)
        >>> print(t.data)
        Tensor([1, 2, 3])
        """
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    
    def item(self):
        """
        Return the scalar value being stored in the current Value as a Python float.
        
        Args:
            None
        
        Returns:
            float: The scalar value being stored in the current Value as a Python float.
        
        """
        return self._data
    
    def is_leaf(self):
        return self._op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1
    
    
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
            for child in node.children:
                if child not in visited:
                    build_topo(child)
            topo.append(node)

        build_topo(self)
        topo.reverse()
        return topo    
        


# %% ../nbs/00_autograd.ipynb 71
NDArray = numpy.ndarray
LAZY_MODE = False
TENSOR_COUNTER = 0

# %% ../nbs/00_autograd.ipynb 72
class Device:
    """Indicates the device supporting an NDArray."""


class CPUDevice(Device):
    """Represents data that sits in CPU"""

    def __repr__(self):
        return "minima.cpu()"

    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, CPUDevice)

    def enabled(self):
        return True
    
    def zeros(self, *shape, dtype="float32"):
        return numpy.zeros(shape, dtype=dtype)

    def ones(self, *shape, dtype="float32"):
        return numpy.ones(shape, dtype=dtype)

    def randn(self, *shape):
        return numpy.random.randn(*shape) 

    def rand(self, *shape):
        return numpy.random.rand(*shape)

    def one_hot(self, n, i, dtype="float32"):
        return numpy.eye(n, dtype=dtype)[i]

def cpu():
    """Return cpu device"""
    return CPUDevice()

def all_devices():
    """return a list of all available devices"""
    return [cpu()]

# %% ../nbs/00_autograd.ipynb 73
class Operator:
    
    def __call__(self, *args):
        raise NotImplementedError()
        
    def compute(self, *args: Tuple[NDArray]):
        raise NotImplementedError()
        
    def gradient(self, out_grad: 'Value', node: 'Value') -> Union['Value', Tuple['Value']]:
        raise NotImplementedError()

# %% ../nbs/00_autograd.ipynb 74
class TensorOp(Operator):
    """ Op class specialized to output tensors, will be alternate subclasses for other structures """

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)

# %% ../nbs/00_autograd.ipynb 75
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
    op: Optional[Operator]  # The operator that produced this node. If the node was initialized from actual data, this is 'None'.
    children: Set['Value']  # The set of values that this value directly depends on. It's the union of the `_next` sets of all the values in `args`.
    cached_data: NDArray           # The actual data for this value. It's `None` for values that aren't yet computed.
    requires_grad: bool     # Specifies whether this node requires a gradient. This is `False` for nodes that don't need gradients.
    
    def compute_cached_data(self):
        """
        If the data of this tensor has not been computed, computes and caches it.
        Otherwise, returns the cached data.

        Returns:
        The actual data of this tensor.
        """

        if self.cached_data is None:
            self.cached_data = self.op.compute(*[child.compute_cached_data() for child in self.children])
        return self.cached_data
    
    def is_leaf(self):
        return self.op is None

# %% ../nbs/00_autograd.ipynb 76
class Tensor(Value):
    """
    A Tensor represents a multidimensional array of values in a computational graph.

    Attributes:
    - data: The actual data of the tensor. It is computed lazily.
    - children: Other tensors that this tensor depends on for computing its value.
    - requires_grad: Whether this tensor needs to compute gradients.

    Methods:
    - compute_cached_data: Computes and returns the actual data for this tensor.
    - shape: Returns the shape of this tensor.
    - dtype: Returns the data type of this tensor.

    Example:
    >>> t1 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> print(t1.shape)
    (2, 2)
    >>> print(t1.dtype)
    float64
    """
    

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        
        """
        Initializes the tensor with given array, device, and data type.

        Args:
        - array: A numeric array-like object (e.g., list, numpy array, or another tensor).
        - device: The device where the tensor should be allocated.
        - dtype: The desired data type for the tensor.
        - requires_grad: Whether the tensor requires gradient computation.

        Returns:
        None.
        """
        
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                data = array.compute_cached_data()
            else:
                # fall back, copy through numpy conversion
                data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else cpu()
            data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(None, (), data=data, requires_grad=requires_grad, )
        
    def __repr__(self):
        return "mi.Tensor(" + str(self.compute_cached_data()) + ")"

    def __str__(self):
        return "mi.Tensor(" + self.compute_cached_data().__str__() + ")"

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        return Tensor(self.cached_data[index])
        
    def __setitem__(self, index, value):
        self.cached_data[index] = value
        
    def _init(
        self,
        op: Optional[Operator],
        children: Set["Tensor"],
        *,
        num_outputs: int = 1,
        data: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        """
        Internal initialization function for the Tensor.

        Args:
        - op: The operator that produces this tensor.
        - children: Set of tensors that this tensor depends on.
        - num_outputs: Number of outputs that the operator produces.
        - data: Actual data of the tensor, computed lazily.
        - requires_grad: Whether this tensor requires gradient computation.

        Returns:
        None.
        """
        
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(child.requires_grad for child in children)
        self.op = op
        self.cached_data = data
        self.children = children
        self.num_outputs = num_outputs
        self.requires_grad = requires_grad
        self.grad: 'Tensor'
    
    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        """
        Converts a numpy array into an array suitable for the given device and data type.

        Args:
        - numpy_array: The numpy array to convert.
        - device: The device where the converted array should be allocated.
        - dtype: The desired data type for the converted array.

        Returns:
        The converted array.
        """

        if ARRAY_API is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return ARRAY_API.array(numpy_array, device=device, dtype=dtype)
    
    @staticmethod
    def make_from_op(op: Operator, children: Tuple["Value"]):
        """
        Creates a new tensor from a given operator and its children.

        Args:
        - op: The operator that produces the tensor.
        - children: The tensors that the operator depends on.

        Returns:
        The newly created tensor.
        """
        
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, children)
        if not LAZY_MODE:
            tensor.compute_cached_data()
        return tensor
    
    def create_detached_tensor(self, data, requires_grad=False) -> 'Tensor':
        """
        Creates a new tensor that shares the data with the current tensor but detaches it from the computational graph.

        Args:
        - data: The data for the new tensor. It can be any array-like object or a Tensor. 
                 If a Tensor is provided, its underlying data is extracted.
        - requires_grad (optional): Whether the new tensor requires gradient computation. 
                                    The default value is False, meaning that the new tensor will be detached from the computational graph.

        Returns:
        A new Tensor that shares the data with the current tensor but is detached from the computational graph.

        Example:
        >>> t = Tensor([1, 2, 3], requires_grad=True)
        >>> t_detached = t.create_detached_tensor(t.data)
        >>> print(t_detached)
        Tensor([1, 2, 3])
        """
        tensor = Tensor.__new__(Tensor)
        tensor._init(None,
                     set(),
                     data=data if not isinstance(data, Tensor) else data.compute_cached_data(),
                     requires_grad=requires_grad)
        return tensor
        
        
    def detach(self) -> 'Tensor':
        """
        Creates a new tensor that shares the data with the current tensor but is detached from the computational graph.

        Returns:
        A new Tensor that shares the data with the current tensor but is detached from the computational graph.

        Example:
        >>> t = Tensor([1, 2, 3], requires_grad=True)
        >>> t_detached = t.detach()
        >>> print(t_detached)
        Tensor([1, 2, 3])
        """
        return self.create_detached_tensor(self.compute_cached_data())

    @property
    def T(self) -> 'Tensor':
        return mi.operators.transpose(self, self.shape)
    
    def numpy(self):
        """
        Converts the tensor data into a NumPy array.

        Returns:
        The data of the tensor as a NumPy array.

        Example:
        >>> t = Tensor([1, 2, 3])
        >>> np_array = t.numpy()
        >>> print(type(np_array))
        <class 'numpy.ndarray'>
        """
        
        data = self.compute_cached_data()
        if ARRAY_API is numpy: return data
        return data.numpy()  # Data is of type NDArray!

    @property
    def data(self):
        """
        Returns a tensor that shares the data with the current tensor but is detached from the computational graph.

        Example:
        >>> t = Tensor([1, 2, 3], requires_grad=True)
        >>> print(t.data)
        Tensor([1, 2, 3])
        """
        return self.detach()

    @data.setter
    def data(self, value):
        """
        Sets the data of the current tensor to the data of another tensor. The tensors must be of the same dtype.

        Args:
        - value: A tensor whose data is used to set the data of the current tensor.

        Raises:
        - AssertionError: If value is not a tensor or if the dtype of value is not the same as the dtype of the current tensor.

        Example:
        >>> t = Tensor([1, 2, 3], dtype=float)
        >>> t2 = Tensor([4, 5, 6], dtype=float)
        >>> t.data = t2
        >>> print(t.data)
        Tensor([4, 5, 6])
        """
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "The dtype of the given tensor (%s) is not the same as the dtype of the current tensor (%s)." % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.compute_cached_data()

    
    @property
    def shape(self):
        """
        Returns the shape of this tensor.

        Returns:
        A tuple representing the shape of this tensor.
        """
        return self.compute_cached_data().shape

    @property
    def dtype(self):
        """
        Returns the data type of this tensor.

        Returns:
        The data type of this tensor.
        """
        return self.compute_cached_data().dtype
    
    @property
    def device(self):
        """
        Returns the device on which the tensor data is stored.

        Returns:
        The device on which the tensor data is stored. If the data is stored in a NumPy array, returns a CPU device.

        Example:
        >>> t = Tensor([1, 2, 3])
        >>> device = t.device
        >>> print(device)
        cpu
        """
        
        data = self.compute_cached_data()
        if ARRAY_API is numpy: return cpu()
        return data.device
    
    def backward(self, out_grad: Optional['Tensor']=None) -> None:
        """
        computes the backward gradient for a given tensor.

        Args:
            output_grad: A tensor that stores the gradients for back propagation.
                Default value is None, which initializes the tensor with ones.
        """
        self.grad = out_grad if out_grad is not None else Tensor(ARRAY_API.ones(self.shape))
        
        node_to_output_grads_list: Dict[Tensor, Tensor] = {}
        node_to_output_grads_list[self] = self.grad

        def topological_sort(t) -> List['Tensor']:
            """
            Given a node in a computational graph, this function returns a list of all nodes in the graph sorted 
            in topological order.

            Args:
                self: A node in a computational graph.

            Returns:
                A list of all nodes in the graph sorted in topological order.
            """
            
            
            visited = set()
            reverse_topo_order = []

            def build_topo(node):
                visited.add(node)
                for child in node.children:
                    if child not in visited:
                        build_topo(child)
                reverse_topo_order.append(node)

            build_topo(t)
            reverse_topo_order.reverse()
            return reverse_topo_order    

        for node in topological_sort(self):
            node.grad = node_to_output_grads_list[node]
            # compute grad of current node w.r.t. output node
            # propagate grad to inputs
            if not node.is_leaf():
                for in_node, grad in zip(node.children, node.op.gradient(node.grad, node)):
                    if in_node not in node_to_output_grads_list:
                        node_to_output_grads_list[in_node] = grad
                    else:
                        node_to_output_grads_list[in_node] += grad

    
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
            return mi.operators.AddScalar(scalar=other)(self)

        else:
            raise ValueError(f"Unsupported operand type for +: '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __sub__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """
        Implements the subtraction operation between two Tensors or a Tensor and a scalar.

        Args:
        - other (Tensor or scalar): the other Tensor or scalar to subtract from this one

        Returns:
        - Tensor: a new Tensor object representing the difference between this Tensor and the other one

        Raises:
        - AssertionError: If the two Tensors don't have the same shape
        - ValueError: If the other operand is neither a Tensor nor a scalar
        """
        if isinstance(other, Tensor):
            # Ensure both tensors have the same shape for subtraction
            if self.shape != other.shape:
                raise AssertionError(f"Tensors must be of the same shape for subtraction. Got {self.shape} and {other.shape}.")

            return mi.operators.EWiseAdd()(self, mi.operators.negate(other))

        elif isinstance(other, (int, float)):
            return mi.operators.AddScalar(scalar=-other)(self)

        else:
            raise ValueError(f"Unsupported operand type for -: '{type(self).__name__}' and '{type(other).__name__}'")


            
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
            return mi.operators.MulScalar(scalar=other)(self)

        else:
            raise ValueError(f"Unsupported operand type for *: '{type(self).__name__}' and '{type(other).__name__}'")
            
    def __pow__(self, other):
        
        if isinstance(other, Tensor):
            raise NotImplementedError()        
        if isinstance(other, (int, float)):
            return mi.operators.PowerScalar(scalar=other)(self)
        else:
            raise ValueError(f"Unsupported operand type for ^: '{type(self).__name__}' and '{type(other).__name__}'")

    def __truediv__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """
        Implements the division operation between two Tensors or a Tensor and a scalar.

        Args:
        - other (Tensor or scalar): the other Tensor or scalar to divide to this one

        Returns:
        - Tensor: a new Tensor object representing the result of division of this Tensor and the other one
        """
        if isinstance(other, Tensor):
            # Ensure both tensors have the same shape for addition
            if self.shape != other.shape:
                raise AssertionError(f"Tensors must be of the same shape for addition. Got {self.shape} and {other.shape}.")

            return mi.operators.EWiseDiv()(self, other)

        elif isinstance(other, (int, float)):
            return mi.operators.DivScalar(scalar=other)(self)

        else:
            raise ValueError(f"Unsupported operand type for /: '{type(self).__name__}' and '{type(other).__name__}'")

    
    def __rtruediv__(self, other): # other / self
        """
        Implements the right division operation between a scalar or a Tensor and this Tensor.

        Args:
        - other (Tensor or scalar): the other Tensor or scalar to divide by this one

        Returns:
        - Tensor: a new Tensor object representing the result of the division

        Example:
        - If the method is called as `other.__rtruediv__(self)`, this corresponds to `other / self` in usual operations.
        """
        return self.__pow__(-1).__mul__(other)
        # other * self**-1

    def __matmul__(self, other):
        """
        Implements the matrix multiplication operation between this Tensor and another Tensor.

        Args:
        - other (Tensor): the other Tensor to multiply with this one

        Returns:
        - Tensor: a new Tensor object representing the result of the matrix multiplication

        Example:
        - If the method is called as `self.__matmul__(other)`, this corresponds to `self @ other` in usual operations.
        """
        return mi.operators.MatMul()(self, other)
    
    
    def matmul(self, other):
        return mi.operators.MatMul()(self, other)

    def sum(self, axes=None):
        return mi.operators.Summation(axes)(self)

    def broadcast_to(self, shape):
        return mi.operators.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return mi.operators.Reshape(shape)(self)

    def __neg__(self):
        return mi.operators.Negate()(self)

    def transpose(self, axes=None):
        return mi.operators.Transpose(axes)(self)
    
    def exp(self) -> 'Tensor':
        return mi.operators.Exp()(self)
        
    def item(self):
        return self.compute_cached_data().item()

    def argmax(self, axis=None, keepdims=None):
        return Tensor(ARRAY_API.argmax(self.compute_cached_data(), axis=axis, keepdims=keepdims))

    @staticmethod
    def accuracy(preds, yb):
       
        assert preds.shape == yb.shape
        correct_predictions = Tensor(preds.compute_cached_data() == yb.compute_cached_data()).sum()
        return correct_predictions / preds.shape[0]

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__
