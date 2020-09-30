from typing import Tuple, Union

import cv2 as cv
import numpy as np

try:
    sift = cv.SIFT.create()
    mser = cv.MSER.create(_max_area=int(10e3), _min_area=50, _max_variation=0.3, _min_diversity=0.1)
except:
    sift = cv.x2features.SIFT_create()
    mser = cv.x2features.MSER_create(_max_area=10e3, _min_area=50, _max_variation=0.3, _min_diversity=0.1)


def to_rgb(image_1c: np.ndarray) -> np.ndarray:
    return np.stack((image_1c,) * 3, axis=-1)


def to_gray(image_3c: np.ndarray) -> np.ndarray:
    return cv.cvtColor(image_3c, cv.COLOR_BGR2GRAY)


def std_filter(image_gray: np.ndarray, kernel_size: int) -> np.ndarray:
    image_gray = image_gray / 255.0
    h = np.ones((kernel_size, kernel_size))
    n = h.sum()
    n1 = n - 1

    c1 = cv.filter2D(image_gray ** 2, -1, h / n1, borderType=cv.BORDER_REFLECT)
    c2 = cv.filter2D(image_gray, -1, h, borderType=cv.BORDER_REFLECT) ** 2 / (n * n1)
    image_filtered = np.sqrt(np.maximum(c1 - c2, 0))

    return (255 * image_filtered).astype(np.uint8)


def morph_line(length: int, angle: int) -> np.ndarray:
    if length < 5:
        length = 5

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
    cv.erode(image_filled, np.ones((3, 3)))

    return image_filled


def find_magnitude_and_angle(image_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    image_sobel_x = cv.Sobel(image_gray, cv.CV_64F, 1, 0)
    image_sobel_y = cv.Sobel(image_gray, cv.CV_64F, 0, 1)

    image_magnitude = np.sqrt(image_sobel_x ** 2.0 + image_sobel_y ** 2.0)
    image_angle = np.arctan2(image_sobel_y, image_sobel_x) * (180 / np.pi)

    image_magnitude = image_magnitude / image_magnitude.max()

    return image_magnitude, image_angle


def apply_watershed(image_rgb: np.ndarray, image_bw: np.ndarray) -> np.ndarray:
    image_opened = cv.morphologyEx(image_bw, cv.MORPH_OPEN, np.ones((3, 3)), iterations=2)

    image_background = cv.dilate(image_opened, np.ones((3, 3)), iterations=3)

    image_distance = cv.distanceTransform(image_opened, cv.DIST_L2, 5)
    thresh_value, image_foreground = cv.threshold(image_distance, 0.7 * image_distance.mean(), 255, 0)
    image_foreground = image_foreground.astype(np.uint8)

    image_unknown = cv.subtract(image_background, image_foreground)

    markers_amount, image_markers = cv.connectedComponents(image_foreground)

    image_markers += 1
    image_markers[image_unknown == 255] = 0
    image_markers = cv.watershed(image_rgb, image_markers)

    return image_markers


def MSER(image_gray: np.ndarray) -> np.ndarray:
    global mser

    regions, boxes = mser.detectRegions(image_gray)
    hulls = [cv.convexHull(r.reshape(-1, 1, 2)) for r in regions]
    mask = cv.drawContours(np.zeros_like(image_gray), hulls, -1, 255, -1)

    return mask


def find_homography_matrix(image_ref: np.ndarray, image_ver: np.ndarray) -> Union[np.ndarray, None]:
    global sift

    MIN_MATCH_COUNT = 10

    keypoints_ref, descriptors_ref = sift.detectAndCompute(image_ref, None)
    keypoints_ver, descriptors_ver = sift.detectAndCompute(image_ver, None)

    flann = cv.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
    matches = flann.knnMatch(descriptors_ref, descriptors_ver, k=2)

    valid_matches = list()
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            valid_matches.append(m)

    if len(valid_matches) < MIN_MATCH_COUNT:
        return None
    else:
        points_ref = np.float32([keypoints_ref[m.queryIdx].pt for m in valid_matches]).reshape(-1, 1, 2)
        points_ver = np.float32([keypoints_ver[m.trainIdx].pt for m in valid_matches]).reshape(-1, 1, 2)

        homo_mat, _ = cv.findHomography(points_ref, points_ver, cv.RANSAC, 5.0)
        return homo_mat


def scale_image(image: np.ndarray, scale: float) -> np.ndarray:
    return cv.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))


def thresh(image_gray: np.ndarray, thresh_adjust: int = -10, otsu: bool = False) -> np.ndarray:
    thresh_value, image_otsu = cv.threshold(image_gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    if otsu:
        return image_otsu

    if thresh_value + thresh_adjust <= 0:
        thresh_adjust = - thresh_value + 1

    _, image_bw = cv.threshold(image_gray, thresh_value + thresh_adjust, 255, cv.THRESH_BINARY)
    return image_bw
