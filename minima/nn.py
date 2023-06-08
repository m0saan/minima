# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_nn.ipynb.

# %% auto 0
__all__ = ['Parameter', 'Module', 'Sequential', 'Linear', 'Flatten', 'ReLU', 'CrossEntropyLoss', 'LayerNorm1d', 'BatchNorm1d',
           'Dropout', 'Residual', 'Identity']

# %% ../nbs/03_nn.ipynb 2
from typing import List, Callable, Any, Tuple
from .autograd import Tensor
from . import operators
import minima.init as init
import numpy as np
import minima as mi
import torch

# %% ../nbs/03_nn.ipynb 3
class Parameter(Tensor):
    """
    A kind of Tensor that is to be considered a module parameter.

    Parameters are `Tensor` subclasses, that have a very special property when used with
    `Module` s - when they're assigned as Module attributes they are automatically added
    to the list of its parameters, and will appear in `Module.parameters()` iterator.
    Another difference is that parameters can't be volatile and that they require gradient by default.
    """

# %% ../nbs/03_nn.ipynb 4
def _unpack_params(value: object) -> List[Tensor]:
    """
    Unpack parameters from different Python objects.

    This function takes an object of type `Parameter`, `Module`, `dict`, `list`, or `tuple` and 
    recursively extracts any contained `Parameter` instances, returning them as a list. For other 
    object types, it returns an empty list.

    Args:
        value (object): The input object which could be of type `Parameter`, `Module`, `dict`, 
                        `list`, `tuple`, or any other type.

    Returns:
        List[Tensor]: A list containing all the `Parameter` instances found within the input object.
                      If no `Parameter` instances are found, an empty list is returned.
                      
    Example:
        module = nn.Module(...)
        params = _unpack_params(module)
        print(params)  # Prints list of `Parameter` instances contained in `module`.
    """
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return list(value.parameters())
    elif isinstance(value, dict):
        return [item for v in value.values() for item in _unpack_params(v)]
    elif isinstance(value, (list, tuple)):
        return [item for v in value for item in _unpack_params(v)]
    return []

# %% ../nbs/03_nn.ipynb 5
def _child_modules(value: object) -> List["Module"]:
    """
    Recursively unpack child modules from different Python objects.

    This function takes an object of type `Module`, `dict`, `list`, or `tuple` and 
    recursively extracts any contained `Module` instances, returning them as a list. 
    For other object types, it returns an empty list.

    Args:
        value (object): The input object which could be of type `Module`, `dict`, 
                        `list`, `tuple`, or any other type.

    Returns:
        List[Module]: A list containing all the `Module` instances found within 
                      the input object. If no `Module` instances are found, 
                      an empty list is returned.

    Example:
        class MyModule(Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(20, 20)
                self.layer2 = nn.Linear(20, 20)
        
        my_module = MyModule()
        children = _child_modules(my_module)
        print(children)  # Prints list of `Module` instances contained in `my_module`.
    """
    if isinstance(value, Module):
        return [value] + _child_modules(value.__dict__)
    elif isinstance(value, dict):
        return [item for v in value.values() for item in _child_modules(v)]
    elif isinstance(value, (list, tuple)):
        return [item for v in value for item in _child_modules(v)]
    else:
        return []

# %% ../nbs/03_nn.ipynb 6
class Module:
    """
    Base class for all neural network modules in Minima.

    Your models should also subclass this class. Subclasses should define a `forward` method.

    Attributes:
    - `training` (bool): Module is initialized in training mode by default. Use `eval()` to switch it to evaluation mode.

    Methods:
    - `parameters()`: Returns a list of all `Parameter` instances in the module.
    - `_children()`: Returns a list of all child `Module` instances.
    - `eval()`: Switches the module and all its children to evaluation mode.
    - `train()`: Switches the module and all its children back to training mode.
    - `__call__()`: The call method, which simply calls the `forward` method, must be defined by all subclasses.
    """
    
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Parameter]:
        """
        Returns a list of all `Parameter` instances in the module.
        This is done by unpacking the parameters from the module's dictionary.
        """
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        """
        Returns a list of all child `Module` instances in the module.
        This is done by unpacking the modules from the module's dictionary.
        """
        return _child_modules(self.__dict__)

    def eval(self):
        """
        Switches the module and all its child modules to evaluation mode.
        """
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        """
        Switches the module and all its child modules to training mode.
        """
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        """
        Defines the call method for the module.
        This method simply calls the forward method and must be overridden by all subclasses.
        """
        return self.forward(*args, **kwargs)

# %% ../nbs/03_nn.ipynb 7
class Sequential(Module):
    """
    A sequential container in Minima.

    Modules will be added to it in the order they are passed in the constructor.
    A `Sequential` module contains a sequence of child modules stored in the order they were added. 
    Each module is applied in order to the input to produce the output.

    The `Sequential` class makes it easy to build networks where the output of one layer is the input to the next.

    Attributes:
    - `modules` (tuple of `Module`): The sequence of child modules to apply.

    Methods:
    - `forward(x: Tensor) -> Tensor`: Passes the input through all the child modules in sequential order.
    """
    def __init__(
        self,
        *modules # The sequence of child modules to apply. Each argument should be an instance of `Module`.
    ):
        """
        Initializes a new `Sequential` instance.
        
        Args:
            *modules: The sequence of child modules to apply. Each argument should be an instance of `Module`.
        """
        super().__init__()
        self.modules = modules
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass for the sequential module.
        
        Passes the input through all the child modules in the order they were added.

        Args:
            x (Tensor): The input tensor.
        
        Returns:
            Tensor: The output tensor.
        """
        for module in self.modules:
            x = module(x)
        return x


# %% ../nbs/03_nn.ipynb 8
class Linear(Module):
    """
    A class representing a fully connected (linear) layer in a neural network.
    This class inherits from the `Module` class.

    Attributes:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        device (str): The device to store the Parameters on (defaults to None, which means CPU).
        dtype (str): The data type of the Parameters (defaults to 'float32').
        weight (Parameter): The weight parameters of the layer.
        bias (Parameter): The bias parameters of the layer, or None if bias=False.

    Methods:
        forward(X: Tensor) -> Tensor: Compute the forward pass of the layer.
    """
    
    def __init__(
        self,
        in_features, # The number of input features.
        out_features,# The number of output features.
        bias=True, # Whether or not to include a bias term. Default is True.
        device=None, # The device to store the Parameters on. Default is None, which means CPU.
        dtype="float32" # The data type of the Parameters. Default is 'float32'.
    ):
        """
        Initialize the layer with given input/output feature sizes and, optionally, bias, device, and dtype.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): Whether or not to include a bias term. Default is True.
            device (str, optional): The device to store the Parameters on. Default is None, which means CPU.
            dtype (str, optional): The data type of the Parameters. Default is 'float32'.
        """
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weight = Parameter(init.kaiming_uniform(fan_in=in_features, fan_out=out_features, device=device, dtype=dtype))
        self.bias = (Parameter(init.kaiming_uniform(fan_in=out_features, fan_out=1, device=device, dtype=dtype)).reshape((1, out_features))
                     if bias else None)
        
    def __repr__(self) -> str:
        return f'Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})'
            
    def forward(self, X: Tensor) -> Tensor:
        """
        Compute the forward pass of the layer.

        This function applies the linear transformation to the input tensor X, 
        i.e., performs the matrix multiplication of X and the weight tensor, 
        and then adds the bias tensor (if bias is not None).

        Args:
            X (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        
        out = X @ self.weight
        out = out + self.bias.broadcast_to(out.shape) if self.bias else out
        return out

# %% ../nbs/03_nn.ipynb 13
class Flatten(Module):
    """
    A `Flatten` module in Minima.

    This module flattens an input tensor into a 2D matrix, typically for transitioning from convolutional layers to linear layers within a neural network model.

    Methods:
    - `forward(X: Tensor) -> Tensor`: Flattens the input tensor.
    """
    
    def forward(self, X: Tensor) -> Tensor:
        """
        Defines the forward pass for the Flatten module.
        
        This method flattens an input tensor along all dimensions except the batch dimension.

        Args:
            X (Tensor): The input tensor. It is expected to have at least two dimensions.

        Returns:
            Tensor: The output tensor, which is a 2D tensor with the same number of elements as the input tensor.
        """
        return X.reshape((X.shape[0], -1))


# %% ../nbs/03_nn.ipynb 14
class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return operators.relu(x)

# %% ../nbs/03_nn.ipynb 24
class CrossEntropyLoss(Module):
    """
    Cross-entropy loss module in Minima.

    This module computes the Cross Entropy Loss between the input logits and the target classes. 
    It's useful in classification tasks where the model outputs probabilities for each class.

    Methods:
    - `forward(input: Tensor, target: Tensor) -> Tensor`: Calculates the cross-entropy loss between the input (logits) and the target (class indices).

    Example:
    ```python
    model = Sequential(
        Linear(10, 20),
        ReLU(),
        Linear(20, 10),
    )
    loss_fn = CrossEntropyLoss()
    output = model(input_tensor)  # compute model output
    loss = loss_fn(output, target_tensor)  # compute loss
    ```
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Computes the Cross Entropy Loss between the input logits and the target class indices.

        Args:
            input (Tensor): The input tensor. The logits, typically of shape (batch_size, num_classes).
            target (Tensor): The target tensor. The correct class indices, typically of shape (batch_size).

        Returns:
            Tensor: A single tensor that is the average cross-entropy loss.
        """
        log_sum_exp_logits = ops.logsumexp(input, axes=(1, )).sum()
        true_class_logits_sum = (input * init.one_hot(input.shape[1], target)).sum()
        return (log_sum_exp_logits - true_class_logits_sum) / input.shape[0]


# %% ../nbs/03_nn.ipynb 33
class LayerNorm1d(Module):
    """
    1D Layer normalization module in Minima.

    Applies layer normalization over a 1D input. The mean and standard deviation are computed over the last dimension.

    Attributes:
    - `dim` (int): The dimension of the input feature space.
    - `eps` (float): A small constant for numerical stability.
    - `weight` (Parameter): The learnable weights of the module of size 'dim', initialized with ones.
    - `bias` (Parameter): The learnable bias of the module of size 'dim', initialized with zeros.

    Methods:
    - `forward(x: Tensor) -> Tensor`: Applies layer normalization to the input tensor.

    """
    def __init__(
        self,
        dim: int, # The dimension of the input feature space.
        eps=1e-5, # A small constant for numerical stability. Default is 1e-5.
        device=None, # The desired device of returned tensor. If None, uses the current device for the default tensor type. Default is None.
        dtype="float32" # The desired data type of returned tensor. If None, uses the default data type. Default is "float32".
    ):
        """
        Initializes a new `LayerNorm1d` instance.
        
        Args:
            dim: The dimension of the input feature space.
            eps: A small constant for numerical stability. Default is 1e-5.
            device: The desired device of returned tensor. If None, uses the current device for the default tensor type. Default is None.
            dtype: The desired data type of returned tensor. If None, uses the default data type. Default is "float32".
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the layer normalization over the input.

        Args:
            x (Tensor): The input tensor of shape (batch_size, num_features).

        Returns:
            Tensor: The output tensor after applying layer normalization.
        """
        bs, fs = x.shape
        axes = (-1,)
        mean = x.sum(axes=axes).reshape((bs, 1)) / fs
        x_centered = x - mean.broadcast_to(x.shape)
        std = ((x_centered ** 2).sum(axes=axes).reshape((bs, 1)) / fs + self.eps) ** 0.5
        x_normed = x_centered / std.broadcast_to(x.shape)
        return self.weight.broadcast_to(x.shape) * x_normed + self.bias.broadcast_to(x.shape)


# %% ../nbs/03_nn.ipynb 36
class BatchNorm1d(Module):
    """
    1D Batch normalization module in Minima.

    This module applies Batch Normalization over a 1D input as described in the paper "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" by Ioffe and Szegedy.

    Attributes:
    - `dim` (int): The dimension of the input feature space.
    - `eps` (float): A small constant added to the denominator for numerical stability.
    - `momentum` (float): The value used for the running_mean and running_var computation.
    - `weight` (Parameter): The learnable scale factor of the module of size 'dim', initialized with ones.
    - `bias` (Parameter): The learnable offset of the module of size 'dim', initialized with zeros.
    - `running_mean` (Tensor): The running mean. Represents the mean of the features over batches. Initialized with zeros.
    - `running_std` (Tensor): The running standard deviation. Represents the standard deviation of the features over batches. Initialized with ones.

    Methods:
    - `update_stats(x: Tensor) -> Tuple[Tensor, Tensor]`: Calculates the mean and standard deviation of the input tensor.
    - `forward(x: Tensor) -> Tensor`: Applies batch normalization to the input tensor.

    Example:
    ```python
    batch_norm = BatchNorm1d(dim=512)
    output = batch_norm(input_tensor)  # Apply batch normalization
    ```

    """
    def __init__(
        self,
        dim: int,
        eps=1e-5,
        momentum=0.1,
        device=None,
        dtype="float32"
    ):
        """
        Initializes a new `BatchNorm1d` instance.
        
        Args:
            dim: The dimension of the input feature space.
            eps: A small constant for numerical stability. Default is 1e-5.
            momentum: The value used for the running_mean and running_var computation. Default is 0.1.
            device: The desired device of returned tensor. If None, uses the current device for the default tensor type. Default is None.
            dtype: The desired data type of returned tensor. If None, uses the default data type. Default is "float32".
        """
        
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        
        self.running_mean = Tensor(init.zeros(dim, device=device, dtype=dtype))
        self.running_std = Tensor(init.ones(dim, device=device, dtype=dtype))
        
    def update_stats(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Updates the running mean and running variance of the input tensor.
        
        Parameters:
        ----------
        x : Tensor
            Input tensor.
        
        Returns:
        ----------
        Tuple[Tensor, Tensor]
            Mean and variance of the input tensor.
        """

        bs, fs = x.shape
        axes=(0,)
        mean = x.sum(axes=axes) / bs
        x_centered = x - mean.broadcast_to(x.shape)
        std = ((x_centered ** 2).sum(axes=axes) / bs)
        self.running_mean = self.momentum * mean.data  + (1 - self.momentum) * self.running_mean
        self.running_std = self.momentum * std.data + (1 - self.momentum) * self.running_std
        return mean,std

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward propagation of the batch normalization layer.
        
        Applies the batch normalization to the input tensor.
        
        Parameters:
        ----------
        x : Tensor
            Input tensor.
        
        Returns:
        ----------
        Tensor
            Output tensor after applying batch normalization.
        """
        
        if self.training:
            mean, std = self.update_stats(x)
        else:
            mean, std = self.running_mean, self.running_std
        x_normed = (x - mean.broadcast_to(x.shape)) / (std.broadcast_to(x.shape) + self.eps) ** .5
        return self.weight.broadcast_to(x.shape) * x_normed + self.bias.broadcast_to(x.shape)

# %% ../nbs/03_nn.ipynb 37
class Dropout(Module):
    """
    Dropout Layer for a Neural Network.
    
    This class represents a dropout layer in a neural network, which is a simple 
    and effective regularization technique.
    During training, it randomly zeroes out some of the elements of the input tensor
    with probability p using samples from a Bernoulli distribution.
    
    Parameters:
    ----------
    p: float, optional, default = 0.5
        Probability of an element to be zeroed. Default: 0.5.
    """
    
    def __init__(self, p = 0.5):
        """
        Initializes the Dropout layer with the specified probability.
        
        Parameters:
        ----------
        p : float
            Probability of an element to be zeroed.
        """
        
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward propagation of the dropout layer.
        
        If the layer is in training mode, it applies dropout to the input tensor. 
        If the layer is in evaluation mode, it returns the input tensor as is.
        
        Parameters:
        ----------
        x : Tensor
            Input tensor.
        
        Returns:
        ----------
        Tensor
            Output tensor after applying dropout.
        """
        
        binary_mask = init.randb(*x.shape, p=self.p)
        if self.training:
            return (binary_mask * x) / (1 - self.p)
        return x


# %% ../nbs/03_nn.ipynb 38
class Residual(Module):
    """
    Residual Layer for a Neural Network.
    
    This class represents a residual layer in a neural network, which is a technique that helps to overcome
    the problem of vanishing and exploding gradients in deep neural networks. It achieves this by allowing
    gradients to pass through layers directly (via an identity shortcut connection) without any modification.

    Parameters:
    ----------
    fn: Module
        The function to be applied to the input tensor.
    """
    
    def __init__(self, fn: Module):
        """
        Initializes the Residual layer with the specified function.
        
        Parameters:
        ----------
        fn : Module
            The function to be applied to the input tensor.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward propagation of the residual layer.
        
        Applies the function to the input tensor and then adds the result to the original input tensor.
        
        Parameters:
        ----------
        x : Tensor
            Input tensor.
        
        Returns:
        ----------
        Tensor
            Output tensor after applying the function and adding the result to the original input.
        """
        return x + self.fn(x)

# %% ../nbs/03_nn.ipynb 39
class Identity(Module):
    def forward(self, x):
        return x
