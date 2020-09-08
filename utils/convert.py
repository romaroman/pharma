from typing import Tuple, Union

import numpy as np


def to_tuple(np_array: np.ndarray) -> Union[Tuple, np.ndarray]:
    try:
        return tuple(to_tuple(element) for element in np_array)
    except TypeError:
        return np_array


def swap_dimensions(shape: Union[Tuple[int, int, int], Tuple[int, int]]) -> Tuple[int, int]:
    return shape[:2][::-1]


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
