import cv2 as cv

from typing import Tuple
import numpy as np


def get_contour_center(contour: np.ndarray) -> Tuple[int, int]:
    moments = cv.moments(contour)

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return cx, cy


def calc_points_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    return np.linalg.norm(np.asarray(point1) - np.asarray(point2))


# TODO: implement
def calc_rrects_distance(coords1: np.ndarray, coords2: np.ndarray) -> float:
    distances = {}

    for idx1, point1 in enumerate(coords1, start=0):
        for idx2, point2 in enumerate(coords2, start=0):
            distances[(idx1, idx2)] = calc_points_distance(point1, point2)

    return 0.0
