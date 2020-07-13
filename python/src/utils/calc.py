import cv2 as cv

from typing import List, Tuple
import numpy as np
import math


def get_contour_center(contour: np.ndarray) -> Tuple[int, int]:
    M = cv.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY


def calc_points_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    # return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point1[1]) ** 2)
    return np.linalg.norm(np.asarray(point1) - np.asarray(point2))


def calc_rrects_distance(coords1: np.ndarray, coords2: np.ndarray) -> float:
    distances = {}

    for idx1, point1 in enumerate(coords1, start=0):
        for idx2, point2 in enumerate(coords2, start=0):
            distances[(idx1, idx2)] = calc_points_distance(point1, point2)

    return 0.0
