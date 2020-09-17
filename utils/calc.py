import cv2 as cv

from typing import Tuple
import numpy as np


def get_contour_center(contour: np.ndarray) -> Tuple[int, int]:
    moments = cv.moments(contour)

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return cx, cy


def calc_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    return np.linalg.norm(np.asarray(point1) - np.asarray(point2))
