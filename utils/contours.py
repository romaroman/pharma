import cv2 as cv
import numpy as np


def approximate_contour(contour: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    epsilon_arc = epsilon * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon_arc, True)
    return approx
