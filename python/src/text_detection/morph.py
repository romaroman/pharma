import cv2 as cv
import numpy as np
from typing import NoReturn, List, Tuple, Dict
import skimage

import utils
from text_detection.types import PreprocessMethod


class Morph:

    @classmethod
    def find_package_mask_and_angle(cls, image_std_filtered, ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        general_rotation = -90

        is_mask_partial = False

        thresh_value, _ = cv.threshold(image_std_filtered, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        _, image_bw = cv.threshold(image_std_filtered, thresh_value - 10, 255, cv.THRESH_BINARY)

        image_dilated = cv.dilate(image_bw, kernel=np.ones((9, 9)))

        contours, _ = cv.findContours(image_dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = np.asarray(contours)
        max_area_contour = max(contours, key=lambda x: cv.contourArea(x))
        points = cv.boxPoints(cv.minAreaRect(max_area_contour)).astype(np.int0)

        mask_to_image_area_ratio = cv.contourArea(points) / image_bw.size

        image_mask = np.zeros_like(image_bw)
        if mask_to_image_area_ratio < 0.3:
            image_mask = np.full(image_bw.shape, 1, dtype=np.uint8)
            general_rotation = int(np.mean(
                [x for x in [int(cv.minAreaRect(x)[2]) for x in sorted(contours, key=lambda x: cv.contourArea(x))[::-1]] if
                 x % 90 != 0]))
        else:
            cv.drawContours(image_mask, [points], -1, 1, -1)
            is_mask_partial = True

        return image_mask, image_bw, general_rotation, is_mask_partial

    @classmethod
    def apply_basic_morphology(cls, image_bw) -> np.ndarray:
        image_dilated = cv.dilate(image_bw, kernel=np.ones((5, 5)))

        image_cleared = utils.clear_borders(image_dilated)

        image_preprocessed = utils.fill_holes(image_cleared)

        return image_preprocessed

    @classmethod
    def extract_edges(cls, image_std_filtered, image_bw, preprocess_method) -> NoReturn:
        image_magnitude, image_angle = utils.find_magnitude_and_angle(image_std_filtered)

        image_mask_from_threshold = (image_bw / 255).astype(np.uint8)

        image_magnitude = image_magnitude * image_mask_from_threshold
        image_angle = image_angle * image_mask_from_threshold

        nbins = 7
        image_ind = (np.ceil((image_angle + 180) / (360 / (nbins - 1))) + 1).astype(np.uint8)

        threshold = 0.075
        image_binary_mask = (image_magnitude > threshold).astype(np.uint8) * 255

        if preprocess_method == PreprocessMethod.EdgeExtraction:
            image_preprocessed = image_binary_mask * image_ind / nbins
        elif preprocess_method == PreprocessMethod.EdgeExtractionAndFiltration:
            image_ind = utils.filter_small_edges(image_ind, image_binary_mask, nbins)

            image_preprocessed = (image_binary_mask * image_ind / nbins).astype(np.uint8)
            image_preprocessed[image_preprocessed != 0] = 255
            image_preprocessed = cv.dilate(image_preprocessed, kernel=np.ones((3, 3)))

        return image_preprocessed

    @classmethod
    def filter_enclosed_contours(cls, image_cleared_borders) -> NoReturn:
        contours, hierarchy = cv.findContours(image_cleared_borders, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

        occurences = {}
        children_areas = {}
        for contour_index, contour_info in enumerate(hierarchy[0], start=0):
            parent_contour = contour_info[3]
            doesnt_have_any_child = contour_info[2] == -1
            contour_area = cv.contourArea(contours[contour_index])
            if contour_area > 20 and parent_contour != -1 and doesnt_have_any_child:
                if parent_contour in occurences.keys():
                    occurences[parent_contour] += 1
                    children_areas[parent_contour].append(contour_area)
                else:
                    occurences[parent_contour] = 1
                    children_areas[parent_contour] = [contour_area]

        image_mask_contours = np.zeros_like(image_cleared_borders)

        contours_to_delete_indices = []
        for index, occ in occurences.items():
            if occ > 5:
                children_area = np.sum(children_areas[index])
                ratio = cv.contourArea(contours[index]) / children_area
                if ratio < 1.5:
                    contours_to_delete_indices.append(index)

        # contours_to_delete_indices = [contour_index for contour_index, amount in occurences.items() if amount > 3]
        contours_to_delete = list(contours[i] for i in contours_to_delete_indices)

        cv.drawContours(image_mask_contours, contours_to_delete, -1, 255, 4)

        return image_mask_contours

    @classmethod
    def filter_non_text_blobs(cls, image_filled) -> NoReturn:
        def is_prop_valid(prop: skimage.measure._regionprops._RegionProperties) -> bool:
            if prop.minor_axis_length > 20:
                if prop.solidity > 0.33:
                    if prop.area > 500:
                        return True

            return False

        _, image_labeled = cv.connectedComponents(image_filled)
        props = skimage.measure.regionprops(image_labeled, coordinates='rc')

        valid_props = list(filter(is_prop_valid, props))
        valid_labels = np.asarray([prop.label for prop in valid_props])

        image_filtered = (np.isin(image_labeled, valid_labels) > 0).astype(np.uint8) * 255

        return image_filtered

    @classmethod
    def apply_line_morphology(cls, image_filtered, scale: float = 1) -> NoReturn:
        line_length = int(150 * scale)

        line_rotation_angles = [i for i in range(-90, 90, 5)]

        pixel_amounts = []
        for line_rotation_angle in line_rotation_angles:
            strel_line_rotated = utils.morph_line(line_length, line_rotation_angle)
            image_linearly_morphed = cv.dilate(image_filtered, strel_line_rotated)

            pixel_amounts.append(cv.countNonZero(image_linearly_morphed))

        k = np.asarray(pixel_amounts).argmin()

        strel_line_rotated = utils.morph_line(int(line_length / 2), line_rotation_angles[k])
        image_linearly_morphed = cv.dilate(image_filtered, strel_line_rotated)

        strel_line_rotated = utils.morph_line(int(line_length / 4), line_rotation_angles[k])
        image_linearly_morphed = cv.erode(image_linearly_morphed, strel_line_rotated)

        image_linearly_morphed = cv.morphologyEx(image_linearly_morphed, cv.MORPH_OPEN, kernel=np.ones((7, 7)))

        return image_linearly_morphed
