from typing import Tuple, Union

import skimage
import cv2 as cv
import numpy as np
import skimage.measure

from common import config

import utils


def mscale(
        obj: Union[int, float, Tuple[int, int], np.ndarray],
        down: bool = True
) -> Union[int, float, Tuple[int, int], np.ndarray]:
    if config.segmentation.scale_factor == 1.0:
        return obj

    scale_f = float(config.segmentation.scale_factor if down else 1 / config.segmentation.scale_factor)

    if type(obj) is int:
        return int(obj * scale_f)
    elif type(obj) is float:
        return obj * scale_f
    elif type(obj) is tuple:
        return int(obj[0] * scale_f), int(obj[1] * scale_f)
    elif type(obj) is np.ndarray:
        return utils.scale_image(obj, scale_f)


def prepare_image(image: np.ndarray, scale_factor: float) -> np.ndarray:
    m = int(1 / scale_factor)
    h, w = image.shape[:2]
    return image[:h - h % m, :w - w % m]


def extract_edges(
        image_gray: np.ndarray,
        image_mask: np.ndarray = None,
        nbins: int = 7,
        post_morph: bool = False
) -> np.ndarray:

    def remove_long_edges(image_ind: np.ndarray, image_binary_mask: np.ndarray, nbins: int = 7) -> np.ndarray:

        def is_edge_valid(contour: np.ndarray) -> bool:
            area = cv.contourArea(contour)
            return area < mscale(50) ** 2

        image_new_ind = np.zeros_like(image_ind)

        for j in range(1, nbins + 1):
            image_current_bin = np.copy(image_ind)
            image_current_bin[image_current_bin != j] = 0
            image_current_bin[image_current_bin != 0] = 1
            image_current_bin = image_current_bin * image_binary_mask

            image_current_bin = cv.dilate(image_current_bin, kernel=np.ones(mscale((3, 3))))

            contours, _ = cv.findContours(image_current_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            valid_contours = list(filter(lambda c: is_edge_valid(c), contours))

            image_bw = np.zeros_like(image_current_bin)
            cv.drawContours(image_bw, valid_contours, -1, 1, -1)

            image_current_bin = np.copy(image_ind)
            image_current_bin[image_current_bin != j] = 0

            image_new_ind = image_new_ind + image_bw * image_current_bin

        return image_new_ind

    if image_mask is None:
        image_mask = np.full(image_gray.shape, 255)

    image_magnitude, image_direction = utils.find_magnitude_and_angle(image_gray)

    image_mask_from_threshold = (image_mask / 255).astype(np.uint8)

    image_magnitude = image_magnitude * image_mask_from_threshold
    image_direction = image_direction * image_mask_from_threshold

    image_ind = (np.ceil((image_direction + 180) / (360 / (nbins - 1))) + 1).astype(np.uint8)

    threshold = 0.025
    image_binary_mask = (image_magnitude > threshold).astype(np.uint8) * 255

    image_ind_filtered = remove_long_edges(image_ind, image_binary_mask, nbins)

    image_edges = (image_binary_mask * image_ind_filtered / nbins).astype(np.uint8)
    image_edges[image_edges != 0] = 255

    if post_morph:
        image_edges = cv.morphologyEx(image_edges, cv.MORPH_OPEN, kernel=np.ones((3, 3)))
         # image_edges = cv.erode(image_edges, kernel=np.ones((3, 3)))

    return image_edges


def filter_enclosed_contours(image_bw: np.ndarray) -> np.ndarray:
    contours, hierarchy = cv.findContours(image_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return image_bw

    occurrences = dict()
    children_areas = dict()
    children_lengths = dict()

    for index, contour_info in enumerate(hierarchy[0], start=0):
        parent_contour = contour_info[3]
        contour_area = cv.contourArea(contours[index])

        if parent_contour != -1 and contour_area > mscale(25):
            contour_length = cv.arcLength(contours[index], False)

            if parent_contour in occurrences.keys():
                occurrences[parent_contour] += 1
                children_areas[parent_contour].append(contour_area)
                children_lengths[parent_contour].append(contour_length)
            else:
                occurrences[parent_contour] = 1
                children_areas[parent_contour] = [contour_area]
                children_lengths[parent_contour] = [contour_length]

    image_mask_contours = np.zeros_like(image_bw)

    contours_to_delete = list()
    for index, occurrence in occurrences.items():
        if occurrence > 5:
            current_contour = contours[index]

            children_area = np.sum(children_areas[index])
            contour_area = cv.contourArea(current_contour)
            area_ratio = contour_area / children_area

            children_length = np.sum(children_lengths[index])
            length_ratio = cv.arcLength(current_contour, False) / children_length

            _, shape, _ = cv.minAreaRect(current_contour)

            contour_mask = np.zeros_like(image_mask_contours)
            cv.drawContours(contour_mask, [current_contour], -1, 255, -1)
            contour_blob = cv.bitwise_and(image_bw, image_bw, mask=contour_mask)

            pixel_amount_inside_contour = cv.countNonZero(contour_blob)
            solidity_ratio = abs(1 - pixel_amount_inside_contour / children_area)

            if solidity_ratio < 0.2 and length_ratio < 0.5 and area_ratio > 1.5:
                contours_to_delete.append(current_contour)

    cv.drawContours(image_mask_contours, contours_to_delete, -1, 255, mscale(20))
    image_filtered = cv.bitwise_and(image_bw, image_bw, mask=~image_mask_contours)

    return image_filtered


def filter_non_text_blobs(image_bw: np.ndarray) -> np.ndarray:

    def is_prop_valid(prop: skimage.measure._regionprops._RegionProperties) -> bool:
        return prop.solidity > 0.25 and prop.area > mscale(15) ** 2

    _, image_labeled = cv.connectedComponents(image_bw)
    props = skimage.measure.regionprops(image_labeled)

    valid_props = list(filter(is_prop_valid, props))
    valid_labels = np.asarray([prop.label for prop in valid_props])

    image_filtered = (np.isin(image_labeled, valid_labels) > 0).astype(np.uint8) * 255

    return image_filtered


def filter_contours(image: np.ndarray, key) -> np.ndarray:
    contours, _ = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    valid_contours = list(filter(key, contours))
    return cv.drawContours(np.zeros_like(image), valid_contours, -1, 255, -1)


def apply_line_morphology(
        image_bw: np.ndarray,
        line_length: Union[int, None] = None,
        key: str = 'min'
) -> np.ndarray:
    if not line_length:
        line_length = max(image_bw.shape) / 10

    line_rotation_angles = [i for i in range(-90, 90, 5)]

    pixel_amounts = list()
    for line_rotation_angle in line_rotation_angles:
        strel_line_rotated = utils.morph_line(line_length, line_rotation_angle)
        image_linearly_morphed = cv.dilate(image_bw, strel_line_rotated)
        pixel_amounts.append(cv.countNonZero(image_linearly_morphed))

    if key == 'min':
        k = np.asarray(pixel_amounts).argmin()
    elif key == 'max':
        k = np.asarray(pixel_amounts).argmax()

    strel_line_rotated = utils.morph_line(int(line_length / 2), line_rotation_angles[k])
    image_linearly_morphed = cv.dilate(image_bw, strel_line_rotated)

    return image_linearly_morphed


def apply_line_morphology_simplified(
        image_bw: np.ndarray,
        angle: int,
        line_length: int
) -> Tuple[int, np.ndarray]:
    strel_line_rotated = utils.morph_line(line_length, angle)
    image_linearly_morphed = cv.dilate(image_bw, strel_line_rotated)

    return image_linearly_morphed


def apply_rectangular_segmentation(image_bw: np.ndarray, axis: int = 0) -> np.ndarray:
    h, w = image_bw.shape

    def find_lines(image_roi, axis):

        if axis == 0:
            limit, _ = image_roi.shape
        elif axis == 1:
            _, limit = image_roi.shape
        else:
            raise AttributeError

        histogram = cv.reduce(image_roi, 1 - axis, cv.REDUCE_AVG).reshape(-1)

        threshold = 3
        uppers = [y for y in range(limit - 1) if histogram[y] <= threshold < histogram[y + 1]]
        lowers = [y for y in range(limit - 1) if histogram[y] > threshold >= histogram[y + 1]]

        return uppers, lowers

    horizontal_upper, horizontal_lower = find_lines(image_bw, axis)
    horizontal_upper_i, horizontal_lower_i = find_lines(~image_bw, axis)

    horizontal_upper.extend(horizontal_upper_i)
    horizontal_lower.extend(horizontal_lower_i)

    horizontal_rectangles = np.asarray(
        [[(0, y1), (w, y2)] for y1, y2 in zip(horizontal_upper, horizontal_lower)], dtype=np.int32
    )

    return horizontal_rectangles


def clear_borders(image_bw: np.ndarray) -> np.ndarray:

    def is_contour_invalid(contour) -> bool:
        amount_of_border_pixels_to_omit = mscale(25)

        if len(contour) >= 4 and abs(1 - cv.contourArea(contour) / image_bw.size) < 0.5:
            for point in contour:
                point = point[0]
                if point[0] <= amount_of_border_pixels_to_omit or point[1] == amount_of_border_pixels_to_omit:
                    return True

        return False

    contour_thickness = int(max(image_bw.shape) * 0.02)

    image_bwd = cv.dilate(image_bw, np.ones(mscale((7, 7))))
    contours, _ = cv.findContours(image_bwd, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    contours_to_delete = list(map(lambda p: utils.approximate_contour(p, 0.01), filter(is_contour_invalid, contours)))

    if not contours_to_delete:
        return image_bw

    image_mask_to_delete = cv.drawContours(np.zeros_like(image_bw), contours_to_delete, -1, 255, contour_thickness)
    image_result = cv.bitwise_and(~image_mask_to_delete, image_bw)

    return clear_borders(image_result)
