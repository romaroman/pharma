from typing import Tuple, Union

import cv2 as cv
import numpy as np


def to_rgb(image_1c: np.ndarray) -> np.ndarray:
    return np.stack((image_1c,) * 3, axis=-1)


def to_gray(image_3c: np.ndarray) -> np.ndarray:
    return cv.cvtColor(image_3c, cv.COLOR_BGR2GRAY)


def clear_borders(image_bw: np.ndarray) -> np.ndarray:

    def is_contour_invalid(contour) -> bool:
        amount_of_border_pixels_to_omit = 25

        for point in contour:
            point = point[0]
            if point[0] <= amount_of_border_pixels_to_omit or point[1] == amount_of_border_pixels_to_omit:
                return True

        return False

    contours, _ = cv.findContours(image_bw, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours_to_delete = list(filter(is_contour_invalid, contours))

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

    return image_filled


def find_magnitude_and_angle(image_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    image_sobel_x = cv.Sobel(image_gray, cv.CV_64F, 1, 0)  # Find x and y gradients
    image_sobel_y = cv.Sobel(image_gray, cv.CV_64F, 0, 1)

    image_magnitude = np.sqrt(image_sobel_x ** 2.0 + image_sobel_y ** 2.0)
    image_angle = np.arctan2(image_sobel_y, image_sobel_x) * (180 / np.pi)

    image_magnitude = image_magnitude / image_magnitude.max()

    return image_magnitude, image_angle


def filter_small_edges(image_ind: np.ndarray, image_binary_mask: np.ndarray, nbins: int = 7) -> np.ndarray:
    image_new_ind = np.zeros_like(image_ind)

    for j in range(1, nbins + 1):
        image_current_bin = np.copy(image_ind)
        image_current_bin[image_current_bin != j] = 0
        image_current_bin[image_current_bin != 0] = 1
        image_current_bin = image_current_bin * image_binary_mask

        image_current_bin = cv.dilate(image_current_bin, kernel=np.ones((14, 14)))

        contours, _ = cv.findContours(image_current_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        valid_contours = list(filter(lambda c: cv.contourArea(c) > 300, contours))

        image_bw = np.zeros_like(image_current_bin)
        cv.drawContours(image_bw, valid_contours, -1, 1, -1)

        image_current_bin = np.copy(image_ind)
        image_current_bin[image_current_bin != j] = 0

        image_new_ind = image_new_ind + image_bw * image_current_bin

    return image_new_ind


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


def MSER(image_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mser = cv.MSER_create()

    image_visualization = image_gray.copy()

    regions, _ = mser.detectRegions(image_gray)

    hulls = [cv.convexHull(r.reshape(-1, 1, 2)) for r in regions]
    cv.polylines(image_visualization, hulls, 1, (0, 255, 0))

    image_mask = np.zeros_like(image_gray)

    cv.drawContours(image_mask, hulls, -1, 255, -1)

    image_text_only = cv.bitwise_and(image_gray, image_gray, mask=image_mask)

    return image_mask, image_text_only, image_visualization


def find_homography(image_ref: np.ndarray, image_ver: np.ndarray) -> Union[np.ndarray, None]:
    sift = cv.xfeatures2d.SIFT_create()

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


def scale(image: np.ndarray, scale: float) -> np.ndarray:
    return cv.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))


def random_forest_edge_detection(image: np.ndarray) -> np.ndarray:
    detector = cv.ximgproc_StructuredEdgeDetection()

    edges = detector.detectEdges(image)
    orientation = detector.computeOrientation(edges)
    edges_supressed = detector.edgesNms(edges, orientation, r=2, s=0, m=1, isParallel=True)

    return edges_supressed
