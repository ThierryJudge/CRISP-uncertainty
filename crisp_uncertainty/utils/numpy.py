import numpy as np


def prob_to_categorical(y: np.ndarray, channel_axis: int = None, threshold=0.5) -> np.ndarray:
    """Converts probability map to categorical map.

    Args:
        y: probability map
        channel_axis: axis of channels
        threshold: threshold for binary conversion.

    Returns:
        categorical data.
    """
    if channel_axis is not None:
        return y.argmax(axis=channel_axis)
    if y.ndim == 4:  # Contains batch dimension
        if y.shape[1] > 1:
            return y.argmax(axis=1)
        else:
            return (y > threshold).astype(np.int8).squeeze(axis=1)
    elif y.ndim == 3:  # Does not contain batch dimension
        if y.shape[0] > 1:
            return y.argmax(axis=0)
        else:
            return (y > threshold).astype(np.int8).squeeze()
    else:
        raise ValueError(f"prob_to_categorical - Dimensions not supported {y.shape}")
