import cv2.cv2 as cv
import numpy as np
from copy import deepcopy

from typing import Tuple


def clear_borders(image_bw: np.ndarray) -> np.ndarray:

    def is_contour_valid(contour) -> bool:
        for point in contour:
            point = point[0]
            if point[0] <= 25 or point[1] == 25:
                return True

        return False

    contours, _ = cv.findContours(image_bw, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours_to_delete = list(filter(is_contour_valid, contours))

    image_contour_mask = np.full(image_bw.shape, 255, dtype=np.uint8)
    contour_thickness = int(max(image_bw.shape) * 0.015)
    cv.drawContours(image_contour_mask, contours_to_delete, -1, 0, contour_thickness)

    image_cleared = cv.bitwise_and(image_bw, image_contour_mask)

    return image_cleared


def std_filter(image_gray: np.ndarray, kernel_size: int) -> np.ndarray:
    image_gray = image_gray / 255.0
    h = np.ones((kernel_size, kernel_size))
    n = h.sum()
    n1 = n - 1
    c1 = cv.filter2D(image_gray ** 2, -1, h / n1, borderType=cv.BORDER_REFLECT)
    c2 = cv.filter2D(image_gray, -1, h, borderType=cv.BORDER_REFLECT) ** 2 / (n * n1)
    image_filtered = np.sqrt(np.maximum(c1 - c2, 0))
    return (255 * image_filtered).astype(np.uint8)


def prepare_image_gray(image: np.ndarray) -> np.ndarray:
    if image.shape[2] == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image


def morph_line(length: int, angle: int) -> np.ndarray:

    length_2 = length * 2

    element = np.zeros((length_2, length_2), dtype=np.uint8)
    element[int(length / 2):int(length * 1.5), length] = 255

    mat = cv.getRotationMatrix2D((length, length), angle, 1)
    element_rotated = cv.warpAffine(element, mat, (length_2, length_2), flags=cv.INTER_NEAREST)

    contours, _ = cv.findContours(element_rotated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv.boundingRect(contours[0])

    element_cropped = element_rotated[y:y+h, x:x+w]

    return element_cropped


def fill_holes(image_bw: np.ndarray) -> np.ndarray:
    contours, _ = cv.findContours(image_bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    image_filled = np.zeros_like(image_bw)
    cv.drawContours(image_filled, contours, -1, 255, -1)
    return image_filled


def find_magnitude_and_angle(image_std_filtered: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sobel_x = cv.Sobel(image_std_filtered, cv.CV_64F, 1, 0)  # Find x and y gradients
    sobel_y = cv.Sobel(image_std_filtered, cv.CV_64F, 0, 1)

    magnitude = np.sqrt(sobel_x ** 2.0 + sobel_y ** 2.0)
    angle = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)

    _, image_bw = cv.threshold(image_std_filtered, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)

    image_bw = image_bw / 255

    magnitude = magnitude * image_bw
    angle = angle * image_bw

    magnitude = magnitude / magnitude.max()

    return magnitude, angle


def filter_long_edges(image_ind: np.ndarray, image_binary_mask: np.ndarray, nbins: int = 7) -> np.ndarray:
    image_new_ind = np.zeros_like(image_ind)

    for j in range(1, nbins + 1):
        image_current_bin = deepcopy(image_ind)
        image_current_bin[image_current_bin != j] = 0
        image_current_bin[image_current_bin != 0] = 1
        image_current_bin = image_current_bin * image_binary_mask

        image_current_bin = cv.dilate(image_current_bin, kernel=np.ones((14, 14)))

        contours, _ = cv.findContours(image_current_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        valid_contours = list(filter(lambda c: cv.contourArea(c) > 300, contours))

        image_bw = np.zeros_like(image_current_bin)
        cv.drawContours(image_bw, valid_contours, -1, 1, -1)

        image_current_bin = deepcopy(image_ind)
        image_current_bin[image_current_bin != j] = 0

        image_new_ind = image_new_ind + image_bw * image_current_bin

    return image_new_ind
