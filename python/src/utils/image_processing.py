import cv2 as cv
import numpy as np
from utils.io import show_image_as_plot


def clear_borders(image_bw: np.ndarray) -> np.ndarray:

    def is_contour_valid(contour) -> bool:
        for point in contour:
            point = point[0]
            if point[0] == 0 or point[1] == 0:
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
