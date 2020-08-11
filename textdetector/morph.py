from typing import Tuple, Union

import skimage
import skimage.measure
import cv2 as cv
import numpy as np

import utils


class Morph:

    class TunableVariables:

        otsu_threshold_adjustment: int = -10

    @classmethod
    def align_package_to_corners(cls, image_rgb: np.ndarray) -> Tuple[bool, Union[np.ndarray, None]]:
        image_gray = cv.cvtColor(image_rgb, cv.COLOR_BGR2GRAY)

        threshold, _ = cv.threshold(image_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        _, image_bw = cv.threshold(image_gray, threshold - 30, 255, cv.THRESH_BINARY)

        image_closed = cv.morphologyEx(image_bw, cv.MORPH_CLOSE, (15, 15))
        image_dilated = cv.dilate(image_closed, (10, 10))

        image_filled = utils.fill_holes(image_dilated)
        image_filled = cv.erode(image_filled, np.ones((10, 10)))

        contours, _ = cv.findContours(image_filled, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        contour = max(contours, key=lambda x: cv.contourArea(x))

        if cv.contourArea(contour) / image_filled.size < 0.3:
            return False, image_rgb

        image_aligned = cls.crop_and_align_contour(image_rgb, contour)

        return True, image_aligned

    @classmethod
    def find_package_mask(cls, image_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        valid_mask_to_image_area_ratio = 0.3
        valid_minrect_contour_area_ratio = 0.2

        is_mask_partial = False

        thresh_value, _ = cv.threshold(image_gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        thresh_value += cls.TunableVariables.otsu_threshold_adjustment
        _, image_bw = cv.threshold(image_gray, thresh_value, 255, cv.THRESH_BINARY)

        image_dilated = cv.dilate(image_bw, kernel=np.ones((9, 9)))

        contours, _ = cv.findContours(image_dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = np.asarray(contours)
        max_area_contour = max(contours, key=lambda x: cv.contourArea(x))
        points = cv.boxPoints(cv.minAreaRect(max_area_contour)).astype(np.int0)

        minrect_contour_area_ratio = abs(1 - cv.contourArea(points) / cv.contourArea(max_area_contour))
        mask_to_image_area_ratio = cv.contourArea(points) / image_bw.size

        image_mask = np.zeros_like(image_bw)
        if mask_to_image_area_ratio < valid_mask_to_image_area_ratio or \
                minrect_contour_area_ratio > valid_minrect_contour_area_ratio:

            image_mask = np.full(image_bw.shape, 1, dtype=np.uint8)

        else:
            cv.drawContours(image_mask, [points], -1, 1, -1)
            is_mask_partial = True

        return image_mask, image_bw, is_mask_partial

    @classmethod
    def extract_edges(
            cls,
            image_gray: np.ndarray,
            image_mask: np.ndarray = None,
            nbins: int = 7,
            post_morph: bool = False
    ) -> np.ndarray:
        if image_mask is None:
            image_mask = np.full(image_gray.shape, 255)

        image_magnitude, image_angle = utils.find_magnitude_and_angle(image_gray)

        image_mask_from_threshold = (image_mask / 255).astype(np.uint8)

        image_magnitude = image_magnitude * image_mask_from_threshold
        image_angle = image_angle * image_mask_from_threshold

        image_ind = (np.ceil((image_angle + 180) / (360 / (nbins - 1))) + 1).astype(np.uint8)

        threshold = 0.075
        image_binary_mask = (image_magnitude > threshold).astype(np.uint8) * 255

        image_ind = utils.filter_small_edges(image_ind, image_binary_mask, nbins)

        image_edges = (image_binary_mask * image_ind / nbins).astype(np.uint8)
        image_edges[image_edges != 0] = 255

        if post_morph:
            image_edges = cv.morphologyEx(image_edges, cv.MORPH_CLOSE, kernel=np.ones((3, 3)))

        return image_edges

    @classmethod
    def filter_enclosed_contours(cls, image_bw: np.ndarray) -> np.ndarray:
        contours, hierarchy = cv.findContours(image_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if hierarchy is None:
            return image_bw

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

        image_mask_contours = np.zeros_like(image_bw)

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

                contour_mask = np.zeros_like(image_mask_contours)
                cv.drawContours(contour_mask, [current_contour], -1, 255, -1)
                contour_blob = cv.bitwise_and(image_bw, image_bw, mask=contour_mask)

                pixel_amount_inside_contour = cv.countNonZero(contour_blob)
                solidity_ratio = abs(1 - pixel_amount_inside_contour / children_area)

                if solidity_ratio < 0.2 and length_ratio < 0.5 and area_ratio > 1.5:
                    contours_to_delete.append(current_contour)

        cv.drawContours(image_mask_contours, contours_to_delete, -1, 255, 20)
        image_filtered = cv.bitwise_and(image_bw, image_bw, mask=~image_mask_contours)

        return image_filtered

    @classmethod
    def filter_non_text_blobs(cls, image_bw: np.ndarray) -> np.ndarray:

        def is_prop_valid(prop: skimage.measure._regionprops._RegionProperties) -> bool:
            return prop.solidity > 0.25 and prop.area > 200

        _, image_labeled = cv.connectedComponents(image_bw)
        props = skimage.measure.regionprops(image_labeled, coordinates='rc')

        valid_props = list(filter(is_prop_valid, props))
        valid_labels = np.asarray([prop.label for prop in valid_props])

        image_filtered = (np.isin(image_labeled, valid_labels) > 0).astype(np.uint8) * 255

        return image_filtered

    @classmethod
    def apply_line_morphology(
            cls,
            image_bw: np.ndarray,
            line_length: Union[int, None] = None,
            scale: float = 1,
            key: str = 'min'
    ) -> Tuple[int, np.ndarray]:

        if not line_length:
            line_length = int(max(image_bw.shape) / 10 * scale)

        line_rotation_angles = [i for i in range(-90, 90, 5)]

        pixel_amounts = []
        for line_rotation_angle in line_rotation_angles:
            strel_line_rotated = utils.morph_line(line_length, line_rotation_angle)
            image_linearly_morphed = cv.dilate(image_bw, strel_line_rotated)

            pixel_amounts.append(cv.countNonZero(image_linearly_morphed))

        if key == 'min':
            k = np.asarray(pixel_amounts).argmin()
        elif key == 'max':
            k = np.asarray(pixel_amounts).argmax()

        final_angle = line_rotation_angles[k]

        strel_line_rotated = utils.morph_line(int(line_length / 2), final_angle)
        image_linearly_morphed = cv.dilate(image_bw, strel_line_rotated)

        return final_angle, image_linearly_morphed

    @classmethod
    def apply_line_morphology_simplified(
            cls,
            image_bw: np.ndarray,
            angle: int,
            line_length: int
    ) -> Tuple[int, np.ndarray]:

        strel_line_rotated = utils.morph_line(line_length, angle)
        image_linearly_morphed = cv.dilate(image_bw, strel_line_rotated)

        return image_linearly_morphed

    @classmethod
    def split_lines(
            cls,
            image_bw: np.ndarray,
            axis: int = 0
    ) -> np.ndarray:

        H, W = image_bw.shape

        def find_lines(image_roi, axis):

            if axis == 0:
                limit, _ = image_roi.shape
            elif axis == 1:
                limit, _ = image_roi.shape
            else:
                raise AttributeError

            hist = cv.reduce(image_roi, 1 - axis, cv.REDUCE_AVG).reshape(-1)

            th = 3
            uppers = [y for y in range(limit - 1) if hist[y] <= th < hist[y + 1]]
            lowers = [y for y in range(limit - 1) if hist[y] > th >= hist[y + 1]]

            return uppers, lowers

        horizontal_upper, horizontal_lower = find_lines(image_bw, axis)
        horizontal_upper_i, horizontal_lower_i = find_lines(~image_bw, axis)

        horizontal_upper.extend(horizontal_upper_i)
        horizontal_lower.extend(horizontal_lower_i)

        horizontal_rectangles = np.asarray([[(0, y1), (W, y2)] for y1, y2 in zip(horizontal_upper, horizontal_lower)],
                                           dtype=np.int0)

        return horizontal_rectangles

    @classmethod
    def crop_and_align_contour(
            cls,
            image: np.ndarray,
            contour: np.ndarray
    ) -> np.ndarray:

        rrect = cv.minAreaRect(contour)
        box = cv.boxPoints(rrect)
        box = np.asarray(box, dtype=np.int0)

        W = rrect[1][0]
        H = rrect[1][1]

        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1, x2, y1, y2 = min(Xs), max(Xs), min(Ys), max(Ys)

        angle = rrect[2]
        if angle < -45:
            angle += 90

        # Center of rectangle in source image
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Size of the upright rectangle bounding the rotated rectangle
        size = (x2 - x1, y2 - y1)
        M = cv.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)

        # Cropped upright rectangle
        cropped = cv.getRectSubPix(image, size, center)
        cropped = cv.warpAffine(cropped, M, size)

        # Final cropped & rotated rectangle
        hw_img = cv.getRectSubPix(cropped, (int(H), int(W)), (size[0] / 2, size[1] / 2))
        wh_img = cv.getRectSubPix(cropped, (int(W), int(H)), (size[0] / 2, size[1] / 2))

        return hw_img if cv.countNonZero(hw_img[:, :, 0]) > cv.countNonZero(wh_img[:, :, 0]) else wh_img
