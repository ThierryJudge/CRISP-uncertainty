from functools import wraps
from typing import Any, Callable, Dict, Mapping, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from vital.utils.format.native import Item
from vital.utils.format.native import prefix as prefix_fn
from vital.utils.format.native import squeeze as squeeze_fn


def _has_method(o: object, name: str) -> bool:
    return callable(getattr(o, name, None))


def prefix(
    prefix: str, exclude: Union[str, Sequence[str]] = None
) -> Callable[[Callable[..., Mapping[str, Any]]], Callable[..., Dict[str, Any]]]:
    """Decorator for functions that return a mapping with string keys, to add a prefix to the keys.

    Args:
        prefix: Prefix to add to the current keys in the mapping.
        exclude: Keys to exclude from the prefix addition. These will remain unchanged in the new mapping.

    Returns:
        Function where the keys of the mapping returned have been prepended with `prefix`, except for the keys listed in
        `exclude`, that are left as-is.
    """

    def prefix_decorator(fn: Callable[..., Mapping[str, Any]]) -> Callable[..., Dict[str, Any]]:
        @wraps(fn)
        def prefix_wrapper(self, *args, **kwargs):
            return prefix_fn(fn(self, *args, **kwargs), prefix, exclude=exclude)

        return prefix_wrapper

    return prefix_decorator


def squeeze(fn: Callable[..., Sequence[Item]]) -> Callable[..., Union[Item, Sequence[Item]]]:
    """Decorator for functions that return sequences of possibly one item, where we would want the lone item directly.

    Args:
        fn: Function that returns a sequence.

    Returns:
        Function that returns a single item from the sequence, or the sequence itself it has more than one item.
    """

    @wraps(fn)
    def unpack_single_item_return(*args, **kwargs):
        return squeeze_fn(fn(*args, **kwargs))

    return unpack_single_item_return


def batch_function(item_ndim: int) -> Callable:
    """Decorator that allows to compute item-by-item measures on a batch of items by looping over the batch.

    Args:
        item_ndim: Dimensionality of the items expected by the function. Allows to detect if the function's input is a
            single item, or a batch of items.

    Returns:
        Function that accepts batch of items as well as individual items.
    """

    def batch_function_decorator(func: Callable) -> Callable:
        @wraps(func)
        def _loop_function_on_batch(*args, **kwargs):
            if _has_method(args[0], func.__name__):
                # If `func` is a method, pass over the implicit `self` as first argument
                self_or_empty, args = args[0:1], args[1:]
            else:
                self_or_empty = ()
            data, *args = args
            if not isinstance(data, np.ndarray):
                raise ValueError(
                    f"The first argument provided to '{func.__name__}' was not the input data, as a numpy array, "
                    f"expected by the function. The first argument of the function was rather of type '{type(data)}'."
                )
            if data.ndim == item_ndim:  # If the input data is a single item
                result = np.array(func(*self_or_empty, data, *args, **kwargs))
                if result.ndim == 0:  # If the function's output is a single number, add a dim of 1 for consistency
                    result = result[None]
            elif data.ndim == (item_ndim + 1):  # If the input data is a batch of items
                result = np.array([func(*self_or_empty, item, *args, **kwargs) for item in data])
            else:
                raise RuntimeError(
                    f"Couldn't apply '{func.__name__}', either in batch or one-shot, over the input data. The use of"
                    f"the `batch_function` decorator allows '{func.__name__}' to accept {item_ndim}D (item) or "
                    f"{item_ndim+1}D (batch) input data. The input data passed to '{func.__name__}' was of shape: "
                    f"{data.shape}."
                )
            return result

        return _loop_function_on_batch

    return batch_function_decorator


def auto_cast_data(func: Callable) -> Callable:
    """Decorator to allow functions relying on numpy arrays to accept other input data types.

    Args:
        func: Function for which to automatically convert the first argument to a numpy array.

    Returns:
        Function that accepts input data types other than numpy arrays by converting between them and numpy arrays.
    """
    cast_types = [Tensor]
    dtypes = [np.ndarray, *cast_types]

    @wraps(func)
    def _call_func_with_cast_data(data, *args, **kwargs):
        dtype = type(data)
        if dtype not in dtypes:
            raise ValueError(
                f"Decorator 'auto_cast_data' used by function '{func.__name__}' does not support casting inputs of "
                f"type '{dtype}' to numpy arrays. Either provide the implementation for casting to numpy arrays "
                f"from '{cast_types}' in 'auto_cast_data' decorator, or manually convert the input of '{func.__name__}'"
                f"to one of the following supported types: {dtypes}."
            )
        if dtype == Tensor:
            data_device = data.device
            data = data.detach().cpu().numpy()
        result = func(data, *args, **kwargs)
        if dtype == Tensor:
            result = torch.tensor(result, device=data_device)
        return result

    return _call_func_with_cast_data


def auto_move_data(fn: Callable) -> Callable:
    """Decorator for ``LightningModule`` methods for which input args should be moved to the correct device.

    It as no effect if applied to a method of an object that is not an instance of ``LightningModule`` and is typically
    applied to ``__call__`` or ``forward``.

    Args:
        fn: A LightningModule method for which the arguments should be moved to the device the parameters are on.

    Example::

        # directly in the source code
        class LitModel(LightningModule):

            @auto_move_data
            def forward(self, x):
                return x

        # or outside
        LitModel.forward = auto_move_data(LitModel.forward)

        model = LitModel()
        model = model.to('cuda')
        model(torch.zeros(1, 3))

        # input gets moved to device
        # tensor([[0., 0., 0.]], device='cuda:0')

    """

    @wraps(fn)
    def auto_transfer_args(self, *args, **kwargs):
        from pytorch_lightning.core.lightning import LightningModule

        if not isinstance(self, LightningModule):
            return fn(self, *args, **kwargs)

        args, kwargs = self.transfer_batch_to_device((args, kwargs))
        return fn(self, *args, **kwargs)

    return auto_transfer_args
