import math

import numpy as np

from jax import random


def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255.0 - 0.5) / 0.5
    return img


def numpy_collate(batch):
    # We need to stack the batch elements as numpy arrays
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def init_func(x):
    return lambda rng, shape, dtype: random.uniform(
        rng,
        shape=shape,
        minval=-1 / np.sqrt(x.shape[1]),
        maxval=1 / np.sqrt(x.shape[1]),
        dtype=dtype,
    )


def xavier_init(key, shape, dtype):
    bound = math.sqrt(6) / math.sqrt(shape[0] + shape[1])
    return random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def expand_mask(mask):
    assert (
        mask.ndim > 2
    ), "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask
