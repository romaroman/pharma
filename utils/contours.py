import cv2 as cv
import numpy as np

from .convert import to_tuple


def approximate_contour(contour: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    epsilon_arc = epsilon * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon_arc, True)
    return approx


def get_mask_by_contour(image_ref: np.ndarray, contour: np.ndarray) -> np.ndarray:
    image_mask = np.zeros_like(image_ref, dtype=np.uint8)
    return cv.drawContours(image_mask, [contour], -1, 255, -1)


def crop_image_by_contour(
        image_in: np.ndarray,
        contour: np.ndarray,
        roughly_by_brect: bool
) -> np.ndarray:
    x, y, w, h = cv.boundingRect(contour)

    if roughly_by_brect:
        return image_in[y:y + h, x:x + w]
    else:
        image_mask = get_mask_by_contour(image_in[:, :, 1] if len(image_in.shape) == 3 else image_in, contour)
        return cv.copyTo(image_in, image_mask)[y:y + h, x:x + w]


def get_brect_contour(contour: np.ndarray) -> np.ndarray:
    x, y, w, h = cv.boundingRect(contour)
    return np.asarray([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], dtype=np.int32).reshape(-1, 1, 2)


def contour_intersect(contour_ref: np.ndarray, contour_ver: np.ndarray, edges_only: bool = False) -> bool:
    for pt in contour_ver:
        return (edges_only and (cv.pointPolygonTest(contour_ref, to_tuple(pt[0]), True) == 0)) or \
               (not(edges_only) and (cv.pointPolygonTest(contour_ref, to_tuple(pt[0]), True) >= 0))


def perspective_transform_contour(contour: np.ndarray, homo: np.ndarray) -> np.ndarray:
    return cv.perspectiveTransform(contour.astype(np.float32), homo).astype(np.int32)
