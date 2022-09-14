import numpy as np

available_reductions = {
    "max": lambda x: np.nanmax(x, axis=tuple(range(1, x.ndim))),
    "min": lambda x: np.nanmin(x, axis=tuple(range(1, x.ndim))),
    "mean": lambda x: np.nanmean(x, axis=tuple(range(1, x.ndim))),
    "sum": lambda x: np.nansum(x, axis=tuple(range(1, x.ndim))),
}
