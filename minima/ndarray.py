# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/06_ndarray.ipynb.

# %% auto 0
__all__ = ['BackendDevice', 'cpu_numpy', 'default_device', 'NDArray']

# %% ../nbs/06_ndarray.ipynb 2
import math
import numpy as np
from . import ndarray_backend_numpy
from typing import Optional, Sequence, Tuple, Union, Callable, Any
from .utility import *
# from . import ndarray_backend_cpu

# %% ../nbs/06_ndarray.ipynb 3
class BackendDevice:
    """A backend device, wraps the implementation module."""

    def __init__(self, name, mod):
        self.name = name
        self.mod = mod

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f"(type='{self.name}')"

    def __getattr__(self, name):
        return getattr(self.mod, name)

    def enabled(self):
        return self.mod is not None

    def randn(self, *shape, dtype="float32"):
        return NDArray(numpy.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape, dtype="float32"):
        return NDArray(numpy.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n, i, dtype="float32"):
        return NDArray(numpy.eye(n, dtype=dtype)[i], device=self)

    def empty(self, shape, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape, fill_value, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr

def cpu_numpy():
    """Return numpy device"""
    return BackendDevice('cpu_numpy', ndarray_backend_numpy)

def default_device():
    return cpu_numpy()

# %% ../nbs/06_ndarray.ipynb 4
class NDArray:

    """
    NDArray represents a n-dimensional array with operations that can be performed on multiple devices. 
    This class is an abstraction over numpy and other backend devices, providing a unified interface to interact with arrays.

    Use cases of this class include numerical operations, scientific computing, and machine learning.

    Parameters
    ----------
    value : NDArray or np.ndarray or other array-like structures
        The array-like structure to be transformed into NDArray.
    device : Optional[BackendDevice]
        The device on which the array computations should be performed. 
        If None, the default device is used.

    Attributes
    ----------
    _shape : tuple
        The shape of the array.
    _strides : tuple
        The strides of the array.
    _offset : int
        The offset in the underlying buffer.
    _device : BackendDevice
        The device on which the array computations are performed.
    _handle : Buffer
        The underlying buffer that holds the data.
    """
    
    def __init__(
        self,
        value: Union['NDArray', np.ndarray, Sequence], # The value on which to create the NDArray from
        device: Optional[BackendDevice] = None # The device on which the array computations are performed.
    ) -> None:
        """
        Constructs a new NDArray instance from an existing `NDArray`, numpy array, or a Python sequence. 
        This array can be used to perform high-performance computations on the specified device.

        Parameters
        ----------
        value : Union[NDArray, np.ndarray, Sequence]
            The value to create the NDArray from. If it's an NDArray, it is deep-copied to the new NDArray. 
            If it's a numpy array, it's copied to a new NDArray. If it's a Python sequence, it's converted to 
            a numpy array and then copied to a new NDArray.

        device : Optional[BackendDevice]
            The device on which the array computations are performed. Defaults to the device of the input value 
            if it's an NDArray, or to the default device otherwise.
        """
        
        if isinstance(value, NDArray): # copy of existing NDArray
            if device is None: device = value._device
            self._init(value.to(device) + 0.0)
        elif isinstance(value, np.ndarray): # copy of existing np array
            device = device if device is not None else default_device()
            array = self.make(value.shape, device=device)
            array._device.from_numpy(np.ascontiguousarray(value), array._handle)
            self._init(array)
        else:
            array = NDArray(np.array(value), device=device)
            self._init(array)

    def _init(self, other) -> None:
        """
        A private method that initializes the new NDArray with the values, shape, strides, offset, device, 
        and handle of another NDArray.
    
        Parameters
        ----------
        other : NDArray
            The NDArray to initialize from.
        """
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle

    @staticmethod
    def make(
        shape: Sequence[int], # The shape of the new array.
        strides: Optional[Sequence[int]] = None, # The strides of the new array. If None, compact strides are computed.
        device: Optional[BackendDevice] = None, # The device on which the new array computations should be performed. If None, the default device is used.
        offset: Optional[int] = None, # The offset in the underlying buffer of the new array. If None, it defaults to 0.
        handle: Optional[Any] = None # The underlying buffer that should hold the data. If None, a new buffer is allocated.
    ) -> 'NDArray':
        """
        Constructs a new NDArray with the specified shape, strides, device, offset, and handle.

        Parameters
        ----------
        shape : Sequence[int]
            The shape of the new array.
        strides : Optional[Sequence[int]]
            The strides of the new array. If None, compact strides are computed.
        device : Optional[BackendDevice]
            The device on which the new array computations should be performed. If None, the default device is used.
        offset : Optional[int]
            The offset in the underlying buffer of the new array. If None, it defaults to 0.
        handle : Optional[Buffer]
            The underlying buffer that should hold the data. If None, a new buffer is allocated.

        Returns
        -------
        NDArray
            A new NDArray instance.
        """
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = NDArray.compact_strides(shape) if strides is None else strides
        array._device = default_device() if device is None else device
        array._offset = offset
        array._handle = array._device.Array(prod(shape)) if handle is None else handle
        return array

    @staticmethod
    def compact_strides(shape) -> Tuple:
        res = [1] + [prod(shape[-i:]) for i in range(1, len(shape))]
        return tuple(res[::-1])

    def _is_compact(self) -> bool:
        return self._strides == self.compact_strides(self._shape) and prod(self._shape) == self._handle.size

    def compact(self) -> 'NDArray':
        """
        Returns a compact version of this array. If the array is already compact, it returns itself.

        Returns
        -------
        NDArray
            The compact version of this array.
        """
        if self._is_compact():
            return self
        out = NDArray.make(shape=self._shape, device=self._device)
        self._device.compact(self._handle, out._handle, self._shape, self._strides, self._offset)
        return out
        
    def as_strided(self, shape, strides) -> 'NDArray':
        assert len(shape) == len(strides)
        return NDArray.make(shape=shape, strides=strides, handle=self._handle)

    def flat(self) -> 'NDArray':
        return self.reshape((self.size, ))

    def to(self, device: BackendDevice) -> 'NDArray':
        """
        Transfers this array to the specified device.

        Parameters
        ----------
        device : BackendDevice
            The device to which this array should be transferred.

        Returns
        -------
        NDArray
            This array after it has been transferred to `device`.
        """
        return self if device == self._device else NDArray(self.numpy(), device=device)
        
    def numpy(self) -> np.ndarray:
        """
        Returns a numpy representation of this array.

        Returns
        -------
        np.ndarray
            A numpy array that has the same data as this array.
        """
        return self._device.to_numpy(self._handle, self._shape, self._strides, self._offset)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def strides(self) -> Tuple[int, ...]:
        return self._strides

    @property
    def device(self) -> BackendDevice:
        return self._device

    @property
    def dtype(self) -> str:
        # only support float32 for now
        return "float32"

    @property
    def ndim(self) -> int:
        """ Return number of dimensions. """
        return len(self._shape)

    @property
    def size(self) -> int:
        return prod(self._shape)

    def __repr__(self) -> str:
        return "NDArray(" + self.numpy().__str__() + f", device={self._device})"

    def __str__(self) -> str:
        return self.numpy().__str__()

    def fill(self, val) -> 'NDArray':
        return self._device.fill(self._handle, val)

    ### Elementwise functions

    def log(self):
        """
        Computes the natural logarithm element-wise for the NDArray. 
    
        Returns
        -------
        NDArray
            A new NDArray with the natural logarithm applied element-wise. The shape of the returned array matches
            the original NDArray.
        """
        out = NDArray.make(self._shape, device=self._device)
        self._device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self):
        """
        Computes the exponential function element-wise for the NDArray.
    
        Returns
        -------
        NDArray
            A new NDArray with the exponential function applied element-wise. The shape of the returned array matches
            the original NDArray.
        """
        
        out = NDArray.make(self._shape, device=self._device)
        self._device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        """
        Computes the hyperbolic tangent element-wise for the NDArray.
    
        Returns
        -------
        NDArray
            A new NDArray with the hyperbolic tangent applied element-wise. The shape of the returned array matches
            the original NDArray.
        """
        
        out = NDArray.make(self._shape, device=self._device)
        self._device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    def reshape(self, new_shape):
        """
        Reshape the matrix without copying memory.  This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array.
        Raises:
            ValueError if product of current shape is not equal to the product
            of the new shape, or if the matrix is not compact.
        Args:
            new_shape (tuple): new shape of the array
        Returns:
            NDArray : reshaped array; this will point to the same memory as the original NDArray.
        """
        
        # TODO: support reshaping of non-compact arrays.
        
        if prod(new_shape) != prod(self._shape):
            raise ValueError("Invalid reshape")
        strides = self._strides if not NDArray.compact_strides(new_shape) == self._strides else NDArray.compact_strides(new_shape)
        return self.as_strided(shape=new_shape, strides=strides)


    def permute(self, new_axes):
        """
        Permute order of the dimensions.  new_axes describes a permutation of the
        existing axes, so e.g.:
          - If we have an array with dimension "BHWC" then .permute((0,3,1,2))
            would convert this to "BCHW" order.
          - For a 2D array, .permute((1,0)) would transpose the array.
        Like reshape, this operation should not copy memory, but achieves the
        permuting by just adjusting the shape/strides of the array.  That is,
        it returns a new array that has the dimensions permuted as desired, but
        which points to the same memory as the original array.
        Args:
            new_axes (tuple): permutation order of the dimensions
        Returns:
            NDarray : new NDArray object with permuted dimensions, pointing
            to the same memory as the original NDArray (i.e., just shape and
            strides changed).
        """
        
        new_shape = tuple(self._shape[i] for i in new_axes)
        new_strides = tuple(self._strides[i] for i in new_axes)
        return self.as_strided(shape=new_shape, strides=new_strides)

    def broadcast_to(self, new_shape):
        """
        Broadcast an array to a new shape.  new_shape's elements must be the
        same as the original shape, except for dimensions in the self where
        the size = 1 (which can then be broadcast to any size).  As with the
        previous calls, this will not copy memory, and just achieves
        broadcasting by manipulating the strides.
        Raises:
            assertion error if new_shape[i] != shape[i] for all i where
            shape[i] != 1
        Args:
            new_shape (tuple): shape to broadcast to
        Returns:
            NDArray: the new NDArray object with the new broadcast shape; should
            point to the same memory as the original array.
        """
        
        for old_shape_i, new_shape_i in zip(self._shape, new_shape):
            if old_shape_i != 1:
                assert new_shape_i == old_shape_i
        new_strides = tuple(0 if old_shape_i == 1 else stride_i for old_shape_i, stride_i in zip(self._shape, self._strides))
        return self.as_strided(shape=new_shape, strides=new_strides)

    def _ewise_or_scalar(self, other: Union['NDArray', float], ewise_fn: Callable, scalr_fn: Callable) -> 'NDArray':
        """
        This private method applies an element-wise function (`ewise_fn`) to two `NDArray` instances, or a scalar function (`scalr_fn`) 
        to this `NDArray` and a scalar value. It returns a new `NDArray` instance with the results.
    
        Parameters
        ----------
        other : Union[NDArray, float]
            The second operand for the operation. It can be either another `NDArray` (for element-wise operations) or a scalar 
            (for scalar operations).
    
        ewise_fn : Callable
            A function to apply element-wise if `other` is an `NDArray`. This function should take two `NDArray` handles and 
            output a handle.
    
        scalr_fn : Callable
            A function to apply if `other` is a scalar. This function should take an `NDArray` handle and a scalar, and 
            output a handle.
    
        Returns
        -------
        NDArray
            A new `NDArray` instance with the results of the operation.
    
        Raises
        ------
        AssertionError
            If `other` is an `NDArray` but does not have the same shape as `self`.
        """
        out = NDArray.make(shape=self._shape, device=self._device)
        if isinstance(other, NDArray):
            assert self._shape == other._shape, f'operands could not be added together with shapes {self._shape} {other._shape}'
            ewise_fn(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalr_fn(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other: Union['NDArray', float]) -> 'NDArray':
        """
        Performs element-wise addition between this array and `other`. If `other` is not an NDArray, it is treated as a scalar.

        Parameters
        ----------
        other : NDArray or scalar
            The other operand in the addition.

        Returns
        -------
        NDArray
            The result of the addition.

        Raises
        ------
        AssertionError
            If `other` is an NDArray and does not have the same shape as this array.
        """
        return self._ewise_or_scalar(other, ewise_fn=self._device.ewise_add, scalr_fn=self._device.scalar_add)

    def __sub__(self, other) -> 'NDArray':
        """
        Implements the subtract operation. This method performs element-wise subtraction between two NDArrays
        or an NDArray and a scalar.
    
        Parameters
        ----------
        other : NDArray or scalar
            The array or scalar to subtract from the current NDArray.
    
        Returns
        -------
        NDArray
            The resultant NDArray after performing subtraction.
        """
        return self + (-other)

    def __rsub__(self, other) -> 'NDArray':
        """
        Implements the reverse subtract operation. This is used when the NDArray is on the right side of a subtraction.
    
        Parameters
        ----------
        other : scalar
            The scalar to subtract the NDArray from.
    
        Returns
        -------
        NDArray
            The resultant NDArray after performing subtraction.
        """
        
        return (-self) + other

    def __mul__(self, other) -> 'NDArray':
        """
        Implements the multiply operation. This method performs element-wise multiplication between two NDArrays
        or an NDArray and a scalar.
    
        Parameters
        ----------
        other : NDArray or scalar
            The array or scalar to multiply with the current NDArray.
    
        Returns
        -------
        NDArray
            The resultant NDArray after performing multiplication.
        """
        return self._ewise_or_scalar(other, ewise_fn=self._device.ewise_mul, scalr_fn=self._device.scalar_mul)

    def __truediv__(self,  other) -> 'NDArray':
        """
        Implements the true divide operation. This method performs element-wise division between two NDArrays
        or an NDArray and a scalar.
    
        Parameters
        ----------
        other : NDArray or scalar
            The array or scalar to divide the current NDArray by.
    
        Returns
        -------
        NDArray
            The resultant NDArray after performing division.
        """
        return self._ewise_or_scalar(other, ewise_fn=self._device.ewise_div, scalr_fn=self._device.scalar_div)

    def __neg__(self):
        """
        Implements the negation operation. This method performs element-wise negation for self(NDArray).

        Parameters
        ----------
        self : NDArray
            The array to negate.
    
        Returns
        -------
        NDArray
            The resultant NDArray after performing negation.
        """
        
        return self * (-1)

    def __pow__(self, scalar) -> 'NDArray':
        out = NDArray.make(self._shape, self._device)
        self._device.scalr_power(self.compact()._handle, scalar, out._handle)
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def maximum(self, other):
        return self.ewise_or_scalar(other, self._device.ewise_maximum, self._device.scalar_maximum)

    def __eq__(self, other):
        return self.ewise_or_scalar(other, self._device.ewise_eq, self._device.scalar_eq)

    def __ge__(self, other):
        return self.ewise_or_scalar(other, self._device.ewise_ge, self._device.scalar_ge)

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)
        

    def process_slice(self, sl, dim):
        """ Convert a slice to an explicit start/stop/step """
        start, stop, step = sl.start, sl.stop, sl.step
        if start == None:
            start = 0
        if start < 0:
            start = self._shape[dim]
        if stop == None:
            stop = self._shape[dim]
        if stop < 0:
            stop = self._shape[dim] + stop
        if step == None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

    def __getitem__(self, idxs):
        """
        Implements the get item operation to access elements or sub-arrays of our NDArray instance. 
        This method supports slicing and integer-based access similar to NumPy. It returns a new NDArray
        object that represents a view into the original array without copying memory.
    
        Raises
        ------
        AssertionError
            If a slice has negative size or step, or if the number of slices is not equal to the number of dimensions.
    
        Parameters
        ----------
        idxs : tuple
            A tuple of slice or integer elements corresponding to the subset of the matrix to get.
    
        Returns
        -------
        NDArray
            A new NDArray object corresponding to the selected subset of elements. This should not copy memory but 
            just manipulate the shape/strides/offset of the new array, referencing the same array as the original one.
        """

        # handle singleton as tuple, everything as slices
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(idxs) == self.ndim, "Need indexes equal to number of dimensions"
        
        shape = []
        for i in idxs:
            d = i.stop - i.start
            dim_size = d // i.step + d % i.step
            shape.append(dim_size)
            
        offset = sum(idx.start * stride for idx, stride in zip(idxs, self._strides))
        strides = tuple(idx.step * stride for idx, stride in zip(idxs, self._strides)) # Corrected line -> haha was FUN!!
        return NDArray.make(shape, strides=strides, device=self._device, handle=self._handle, offset=offset)

    def __setitem__(self, idxs, other):
        """
        Implements the set item operation to modify elements or sub-arrays of our NDArray instance. 
        This method supports slicing and integer-based access similar to NumPy. It modifies the original NDArray
        in place.
        -> uses same semantics as __getitem__().
    
        Parameters
        ----------
        idxs : tuple
            A tuple of slice or integer elements corresponding to the subset of the matrix to set.
    
        other : NDArray or scalar
            The array or scalar value to set into the specified subset of the matrix. If `other` is an NDArray,
            its shape should match the shape of the subset defined by `idxs`.
    
        Raises
        ------
        AssertionError
            If `other` is an NDArray and its shape does not match the shape of the subset defined by `idxs`.
        """
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert prod(view._shape) == prod(other._shape)
            self._device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view._shape,
                view._strides,
                view._offset,
            )
        else:
            self._device.scalar_setitem(
                other,
                view._handle,
                view._shape,
                view._strides,
                view._offset,
            )

    def reduce_view_out(self, axis):
        """
        Prepares and returns a view of the array and an output array, set up 
        for performing reduction functions.
        """
        if axis is None:
            # Reshape the array into 1D if we're reducing over all axes.
            new_shape = (1,) * (self.ndim - 1) + (prod(self._shape),)
            view = self.reshape(new_shape)
    
            # Prepare an output array with one element for each dimension.
            output_shape = (1,) * self.ndim
            out = NDArray.make(output_shape, device=self._device)
        else:
            # If we're reducing over a specific axis, bring that axis to the end.
            permute_order = tuple(a for a in range(self.ndim) if a != axis) + (axis,)
            view = self.permute(permute_order)
    
            # Prepare an output array with the same shape as the original, 
            # but with 1 in place of the reduction axis.
            output_shape = tuple(1 if i == axis else s for i, s in enumerate(self._shape))
            out = NDArray.make(output_shape, device=self._device)
        return view, out

    def reduce(self, operation, axis=None):
        """
        Performs a reduction operation ('sum' or 'max') over the given axis, 
        or over the entire array if no axis is provided.
        """
        # Prepare the view and output array.
        view, out = self.reduce_view_out(axis)
        
        # Perform the operation.
        if operation == 'sum':
            self._device.reduce_sum(view.compact()._handle, out._handle, view._shape[-1])
        elif operation == 'max':
            self._device.reduce_max(view.compact()._handle, out._handle, view._shape[-1])
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
        return out
    
    def sum(self, axis=None):
        """Performs a sum operation over the given axis, or over the entire array if no axis is provided."""
        return self.reduce('sum', axis)
    
    def max(self, axis=None):
        """Finds the maximum value over the given axis, or over the entire array if no axis is provided."""
        return self.reduce('max', axis)

    def __matmul__(self, other):
        """Perform matrix multiplication of two arrays."""
    
        # Ensuring the arrays are 2D and have matching dimensions for multiplication
        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]
    
        # Retrieve the dimensions of the two matrices
        m, n, p = self.shape[0], self.shape[1], other.shape[1]
    
        def tile_matrix(matrix, tile_size):
            """Function to tile a matrix based on a given tile size."""
            return matrix.as_strided(
                (matrix.shape[0] // tile_size, matrix.shape[1] // tile_size, tile_size, tile_size),
                (matrix.shape[1] * tile_size, tile_size, self.shape[1], 1),
            )
    
        if hasattr(self.device, "matmul_tiled") and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):
            # The device supports tiled multiplication and the dimensions of the matrices are divisible by the tile size
    
            # Determine the tile size
            tile_size = self.device.__tile_size__
    
            # Tile the matrices
            tiled_self = tile_matrix(self.compact(), tile_size).compact()
            tiled_other = tile_matrix(other.compact(), tile_size).compact()
    
            # Create an output array for the result
            output = NDArray.make((tiled_self.shape[0], tiled_other.shape[1], tile_size, tile_size), device=self.device)
    
            # Perform the tiled matrix multiplication
            self.device.matmul_tiled(tiled_self._handle, tiled_other._handle, output._handle, m, n, p)
    
            # Rearrange and reshape the output to get the final result
            return output.permute((0, 2, 1, 3)).compact().reshape((m, p))

        else:
            # The device does not support tiled multiplication or the dimensions of the matrices are not divisible by the tile size
    
            # Create an output array for the result
            output = NDArray.make((m, p), device=self.device)
    
            # Perform regular matrix multiplication
            self.device.matmul(self.compact()._handle, other.compact()._handle, output._handle, m, n, p)
    
            return output


