from typing import Tuple, Union

import numpy as np


def to_tuple(np_array: np.ndarray) -> Union[Tuple, np.ndarray]:
    try:
        return tuple(to_tuple(element) for element in np_array)
    except TypeError:
        return np_array
