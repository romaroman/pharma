import cv2 as cv
import numpy as np
from typing import Tuple
import skimage
import skimage.measure

import utils
from text_detection.types import PreprocessMethod


class Morph:

    class TunableVariables:

        otsu_threshold_adjustment: int = -10



    @classmethod
    def find_package_mask(cls, image_std_filtered: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        valid_mask_to_image_area_ratio = 0.3
        valid_minrect_contour_area_ratio = 0.2

        is_mask_partial = False

        thresh_value, _ = cv.threshold(image_std_filtered, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        thresh_value += cls.TunableVariables.otsu_threshold_adjustment
        _, image_bw = cv.threshold(image_std_filtered, thresh_value, 255, cv.THRESH_BINARY)

        image_dilated = cv.dilate(image_bw, kernel=np.ones((9, 9)))

        contours, _ = cv.findContours(image_dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = np.asarray(contours)
        max_area_contour = max(contours, key=lambda x: cv.contourArea(x))
        points = cv.boxPoints(cv.minAreaRect(max_area_contour)).astype(np.int0)

        minrect_contour_area_ratio = abs(1 - cv.contourArea(points) / cv.contourArea(max_area_contour))
        mask_to_image_area_ratio = cv.contourArea(points) / image_bw.size

        image_mask = np.zeros_like(image_bw)
        if mask_to_image_area_ratio < valid_mask_to_image_area_ratio or\
                minrect_contour_area_ratio > valid_minrect_contour_area_ratio:

            image_mask = np.full(image_bw.shape, 1, dtype=np.uint8)

        else:
            cv.drawContours(image_mask, [points], -1, 1, -1)
            is_mask_partial = True

        return image_mask, image_bw, is_mask_partial

    @classmethod
    def extract_edges(cls, image_std_filtered: np.ndarray, image_bw: np.ndarray,
                      nbins: int = 7, post_dilate: bool = True) -> np.ndarray:

        image_magnitude, image_angle = utils.find_magnitude_and_angle(image_std_filtered)

        image_mask_from_threshold = (image_bw / 255).astype(np.uint8)

        image_magnitude = image_magnitude * image_mask_from_threshold
        image_angle = image_angle * image_mask_from_threshold

        image_ind = (np.ceil((image_angle + 180) / (360 / (nbins - 1))) + 1).astype(np.uint8)

        threshold = 0.075
        image_binary_mask = (image_magnitude > threshold).astype(np.uint8) * 255

        image_preprocessed = np.zeros_like(image_binary_mask)
        image_ind = utils.filter_small_edges(image_ind, image_binary_mask, nbins)

        image_preprocessed = (image_binary_mask * image_ind / nbins).astype(np.uint8)
        image_preprocessed[image_preprocessed != 0] = 255
        if post_dilate:
            image_preprocessed = cv.dilate(image_preprocessed, kernel=np.ones((3, 3)))

        return image_preprocessed

    @classmethod
    def filter_enclosed_contours(cls, image_cleared_borders: np.ndarray) -> np.ndarray:
        contours, hierarchy = cv.findContours(image_cleared_borders, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        occurrences = {}
        children_areas = {}
        children_lengths = {}
        for index, contour_info in enumerate(hierarchy[0], start=0):
            parent_contour = contour_info[3]
            contour_area = cv.contourArea(contours[index])

            if parent_contour != -1 and contour_area > 20:
                contour_length = cv.arcLength(contours[index], False)

                if parent_contour in occurrences.keys():
                    occurrences[parent_contour] += 1
                    children_areas[parent_contour].append(contour_area)
                    children_lengths[parent_contour].append(contour_length)
                else:
                    occurrences[parent_contour] = 1
                    children_areas[parent_contour] = [contour_area]
                    children_lengths[parent_contour] = [contour_length]

        image_mask_contours = np.zeros_like(image_cleared_borders)

        contours_to_delete = []
        for index, occurrence in occurrences.items():
            if occurrence > 5:
                current_contour = contours[index]

                children_area = np.sum(children_areas[index])
                contour_area = cv.contourArea(current_contour)
                area_ratio = contour_area / children_area

                children_length = np.sum(children_lengths[index])
                length_ratio = cv.arcLength(current_contour, False) / children_length

                _, shape, _ = cv.minAreaRect(current_contour)
                area_c_ratio = shape[0] * shape[1] / contour_area

                contour_mask = np.zeros_like(image_mask_contours)
                cv.drawContours(contour_mask, [current_contour], -1, 255, -1)
                contour_blob = cv.bitwise_and(image_cleared_borders, image_cleared_borders, mask=contour_mask)

                pixel_amount_inside_contour = cv.countNonZero(contour_blob)
                solidity_ratio = abs(1 - pixel_amount_inside_contour / children_area)

                if solidity_ratio < 0.2 and length_ratio < 0.5 and area_ratio > 1.5:
                    contours_to_delete.append(current_contour)

                    # cx, cy = utils.get_contour_center(current_contour)
                    # cv.putText(image_mask_contours, f"{index}, {children_area:.2f} : {area_ratio:.2f}", (cx - 60, cy),
                    #            cv.FONT_HERSHEY_COMPLEX, 0.5, 255, 1)
                    # cv.putText(image_mask_contours, f"     {children_length:.2f} : {length_ratio:.2f}",
                    #            (cx - 60, cy + 20), cv.FONT_HERSHEY_COMPLEX, 0.5, 255, 1)
                    # cv.putText(image_mask_contours, f"     {area_c_ratio:.2f}", (cx - 60, cy + 40),
                    #            cv.FONT_HERSHEY_COMPLEX, 0.5, 255, 1)
                # print(pixel_amount_inside_contour, )
                # contours_to_delete.append(current_contour)
                # print(f"{index}, {children_area:.2f} : {area_ratio:.2f}")
                # print(f"     {children_length:.2f} : {length_ratio:.2f}")
                # print(f"     {area_c_ratio:.2f}")

        cv.drawContours(image_mask_contours, contours_to_delete, -1, 255, 20)
        image_filtered = cv.bitwise_and(image_cleared_borders, image_cleared_borders, mask=~image_mask_contours)

        return image_filtered

    @classmethod
    def filter_non_text_blobs(cls, image_filled: np.ndarray) -> np.ndarray:

        def is_prop_valid(prop: skimage.measure._regionprops._RegionProperties) -> bool:
            if prop.minor_axis_length > 20:
                if prop.solidity > 0.25:
                    if prop.area > 300:
                        return True

            return False

        _, image_labeled = cv.connectedComponents(image_filled)
        props = skimage.measure.regionprops(image_labeled, coordinates='rc')

        valid_props = list(filter(is_prop_valid, props))
        valid_labels = np.asarray([prop.label for prop in valid_props])

        image_filtered = (np.isin(image_labeled, valid_labels) > 0).astype(np.uint8) * 255

        return image_filtered

    @classmethod
    def apply_line_morphology(cls, image_filtered: np.ndarray, line_length: int = None, scale: float = 1) -> np.ndarray:
        if not line_length:
            line_length = int(max(image_filtered.shape) / 10 * scale)

        line_rotation_angles = [i for i in range(-90, 90, 5)]

        pixel_amounts = []
        for line_rotation_angle in line_rotation_angles:
            strel_line_rotated = utils.morph_line(line_length, line_rotation_angle)
            image_linearly_morphed = cv.dilate(image_filtered, strel_line_rotated)

            pixel_amounts.append(cv.countNonZero(image_linearly_morphed))

        k = np.asarray(pixel_amounts).argmin()

        strel_line_rotated = utils.morph_line(int(line_length / 2), line_rotation_angles[k])
        image_linearly_morphed = cv.dilate(image_filtered, strel_line_rotated)

        # strel_line_rotated = utils.morph_line(int(line_length / 4), line_rotation_angles[k])
        # image_linearly_morphed = cv.erode(image_linearly_morphed, strel_line_rotated)
        #
        # image_linearly_morphed = cv.morphologyEx(image_linearly_morphed, cv.MORPH_OPEN, kernel=np.ones((7, 7)))

        return image_linearly_morphed

    @classmethod
    def apply_line_morphology_simplified(cls, image_filtered: np.ndarray, angle: int, line_length: int) -> np.ndarray:

        line_rotation_angles = [i for i in range(angle, angle + 180, 5)]

        pixel_amounts = []
        for line_rotation_angle in line_rotation_angles:
            strel_line_rotated = utils.morph_line(line_length, line_rotation_angle)
            image_linearly_morphed = cv.dilate(image_filtered, strel_line_rotated)

            pixel_amounts.append(cv.countNonZero(image_linearly_morphed))

        k = np.asarray(pixel_amounts).argmin()

        strel_line_rotated = utils.morph_line(line_length, line_rotation_angles[k])
        image_linearly_morphed = cv.dilate(image_filtered, strel_line_rotated)

        return image_linearly_morphed
