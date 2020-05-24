import cv2 as cv
import numpy as np
import skimage.measure

import pathlib
import os

import time
from copy import deepcopy
from typing import List, Tuple, NoReturn
from enum import Enum, auto

import utils
import logging
from text_detection.file_info import get_file_info


_rrect = List[Tuple[Tuple[float, float], Tuple[float, float], float]]
logger = utils.get_logger(__name__, logging.DEBUG)

display = utils.show_image_as_plot


class DetectTextRegion:

    class PreprocessMethod(Enum):

        def __str__(self):
            return str(self.value)

        BasicMorphology = auto(),
        EdgeExtraction = auto(),
        EdgeExtractionAndFiltration = auto()

    def __init__(self, image_path: str, preprocess_method: PreprocessMethod):
        self.timestamp = time.time()

        self.filename = pathlib.Path(image_path).stem
        self.preprocess_method = preprocess_method

        self.image_orig: np.ndarray = cv.imread(image_path)
        self.image_gray: np.ndarray = cv.cvtColor(self.image_orig, cv.COLOR_RGB2GRAY)

        self.is_mask_partial: bool = False
        self.image_mask: np.ndarray = np.zeros_like(self.image_gray)

        self.morph_start_angle: int = -90

        self.image_bw: np.ndarray = np.zeros_like(self.image_gray)
        self.image_std_filtered: np.ndarray = utils.std_filter(self.image_gray, 5)

        self.image_visualization: np.ndarray = np.zeros((self.image_gray.shape[0] * 4, self.image_gray.shape[1] * 4))

        self.image_preprocessed: np.ndarray = np.zeros_like(self.image_gray)

        self.image_filled: np.ndarray = np.zeros_like(self.image_gray)
        self.image_cleared_borders: np.ndarray = np.zeros_like(self.image_gray)
        self.image_filtered: np.ndarray = np.zeros_like(self.image_gray)
        self.image_linearly_morphed: np.ndarray = np.zeros_like(self.image_gray)
        self.image_labeled: np.ndarray = np.zeros_like(self.image_gray)

        self.image_text_masks: np.ndarray = np.zeros_like(self.image_gray)
        self.image_text_regions: np.ndarray = deepcopy(self.image_orig)

        self.rrects: List[_rrect] = list()
        self.coordinates: List[np.ndarray] = list()
        self.coordinates_ravel: List[np.ndarray] = list()

        self._update_timestamp("CLASS INITIALIZATION")

    def detect_text_regions(self) -> NoReturn:
        self._find_package_mask()

        if self.is_mask_partial:
            self._apply_mask()

        self._update_timestamp("MASK OPERATIONS")

        self._preprocess()
        self._update_timestamp("PREPROCESS")

        self.image_cleared_borders = utils.clear_borders(self.image_preprocessed)

        self.image_filled = utils.fill_holes(self.image_cleared_borders)

        self._filter_non_text_blobs()
        self._update_timestamp("BORDER, FILLING, FILTERING")

        self._apply_line_morphology()
        self._update_timestamp("LINE MORPHOLOGY")

        self._find_text_regions()
        self._update_timestamp("FIND TEXT REGIONS")

        self._draw_text_regions_and_mask()

        self._create_visualization()

        logger.info(f'SUCCESS {self.filename}, found {len(self.rrects)} regions')

    def write_results(self, base_folder: str, database: str) -> NoReturn:
        text_coord_dst_folder = base_folder + database + f"/python/text_coords"
        text_regions_dst_folder = base_folder + database + f"/python/text_regions"
        text_masks_dst_folder = base_folder + database + f"/python/text_masks"
        visualization_dst_folder = base_folder + database + f"/python/visualizations"

        os.makedirs(text_coord_dst_folder, exist_ok=True)
        os.makedirs(text_regions_dst_folder, exist_ok=True)
        os.makedirs(text_masks_dst_folder, exist_ok=True)
        os.makedirs(visualization_dst_folder, exist_ok=True)

        text_coord_dst_path = text_coord_dst_folder + f"/{self.filename}.csv"
        text_regions_dst_path = text_regions_dst_folder + f"/{self.filename}.png"
        text_masks_dst_path = text_masks_dst_folder + f"/{self.filename}.png"
        visualization_dst_path = visualization_dst_folder + f"/{self.filename}.png"

        np.savetxt(text_coord_dst_path, self.coordinates_ravel, delimiter=",", fmt='%i')
        cv.imwrite(text_regions_dst_path, self.image_text_regions)
        cv.imwrite(text_masks_dst_path, self.image_text_masks)
        cv.imwrite(visualization_dst_path, self.image_visualization)

    def _find_package_mask(self):
        _, self.image_bw = cv.threshold(self.image_std_filtered, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

        image_dilated = cv.dilate(self.image_bw, kernel=np.ones((9, 9)))

        contours, _ = cv.findContours(image_dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = np.asarray(contours)
        max_area_contour = max(contours, key=lambda x: cv.contourArea(x))
        points = cv.boxPoints(cv.minAreaRect(max_area_contour)).astype(np.int0)

        mask_to_image_area_ratio = cv.contourArea(points) / self.image_gray.size

        if mask_to_image_area_ratio < 0.3:
            self.image_mask = np.full(self.image_bw.shape, 1, dtype=np.uint8)
        else:
            cv.drawContours(self.image_mask, [points], -1, 1, -1)
            self.is_mask_partial = True

        self.morph_start_angle = int(np.mean(
            [x for x in [int(cv.minAreaRect(x)[2]) for x in sorted(contours, key=lambda x: cv.contourArea(x))[::-1]] if
             x % 45 != 0]))

    def _apply_mask(self):
        self.image_gray = self.image_gray * self.image_mask
        self.image_orig = self.image_orig * utils.to_rgb(self.image_mask)

        self.image_bw = self.image_bw * self.image_mask
        self.image_std_filtered = self.image_std_filtered * self.image_mask

    def _preprocess(self):
        if self.preprocess_method == self.PreprocessMethod.BasicMorphology:
            self._apply_basic_morphology()
        else:
            self._extract_edges()

    def _apply_basic_morphology(self) -> NoReturn:
        image_dilated = cv.dilate(self.image_bw, kernel=np.ones((5, 5)))

        image_cleared = utils.clear_borders(image_dilated)

        self.image_preprocessed = utils.fill_holes(image_cleared)

    def _extract_edges(self) -> NoReturn:
        image_magnitude, image_angle = utils.find_magnitude_and_angle(self.image_std_filtered)

        nbins = 7
        image_ind = (np.ceil((image_angle + 180) / (360 / (nbins - 1))) + 1).astype(np.uint8)

        threshold = 0.075
        image_binary_mask = (image_magnitude > threshold).astype(np.uint8) * 255

        if self.preprocess_method == self.PreprocessMethod.EdgeExtraction:
            self.image_preprocessed = image_binary_mask * image_ind / nbins
        elif self.preprocess_method == self.PreprocessMethod.EdgeExtractionAndFiltration:
            # image_ind = utils.filter_long_edges(image_ind, image_binary_mask, nbins)

            self.image_preprocessed = (image_binary_mask * image_ind / nbins).astype(np.uint8)
            self.image_preprocessed[self.image_preprocessed != 0] = 255
            self.image_preprocessed = cv.dilate(self.image_preprocessed, kernel=np.ones((8, 8))).astype(np.uint8)

    def _filter_non_text_blobs(self) -> NoReturn:
        def is_prop_valid(prop: skimage.measure._regionprops._RegionProperties) -> bool:
            if prop.minor_axis_length > 20:
                if prop.solidity > 0.33:
                    if prop.area > 500:
                        return True

            return False

        _, image_labeled = cv.connectedComponents(self.image_filled)
        props = skimage.measure.regionprops(image_labeled, coordinates='rc')

        valid_props = list(filter(is_prop_valid, props))
        valid_labels = np.asarray([prop.label for prop in valid_props])

        self.image_filtered = (np.isin(image_labeled, valid_labels) > 0).astype(np.uint8) * 255

    def _apply_line_morphology(self, scale: float = 1) -> NoReturn:
        line_length = int(150 * scale)

        line_rotation_angles = [i for i in range(self.morph_start_angle, 90, 5)]

        pixel_amounts = []
        for line_rotation_angle in line_rotation_angles:
            strel_line_rotated = utils.morph_line(line_length, line_rotation_angle)
            self.image_linearly_morphed = cv.dilate(self.image_filtered, strel_line_rotated)

            pixel_amounts.append(cv.countNonZero(self.image_linearly_morphed))

        k = np.asarray(pixel_amounts).argmin()

        strel_line_rotated = utils.morph_line(int(line_length / 6), line_rotation_angles[k])
        self.image_linearly_morphed = cv.dilate(self.image_filtered, strel_line_rotated)

        print(self.morph_start_angle, line_rotation_angles[k])

    def _find_text_regions(self) -> NoReturn:
        # self.image_linearly_morphed = self.image_filtered
        self.image_linearly_morphed = cv.morphologyEx(self.image_linearly_morphed, cv.MORPH_OPEN, kernel=np.ones((5, 5)))

        _, self.image_labeled = cv.connectedComponents(self.image_linearly_morphed)

        for k in range(1, self.image_labeled.max() + 1):
            image_blob = self.image_labeled[self.image_labeled == k]
            contours, _ = cv.findContours(image_blob, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            rrect = cv.minAreaRect(contours[0])
            points = np.int0(cv.boxPoints(rrect))

            self.rrects.append(rrect)
            self.coordinates.append(points)
            self.coordinates_ravel.append(np.transpose(points).ravel())

    def _draw_text_regions_and_mask(self, color: int = 255) -> NoReturn:
        for coord in self.coordinates:
            cv.drawContours(self.image_text_regions, [coord], -1, (color, 0, 0), 2)
            cv.drawContours(self.image_text_masks, [coord], -1, color, -1)

    def _create_visualization(self) -> NoReturn:

        first_row = np.hstack([
            self.image_orig,
            utils.to_rgb(self.image_mask * 255),
            utils.to_rgb(self.image_preprocessed),
            utils.to_rgb(self.image_cleared_borders)
        ])

        second_row = np.hstack([
            utils.to_rgb(self.image_filled),
            utils.to_rgb(self.image_filtered),
            utils.to_rgb(self.image_linearly_morphed),
            utils.to_rgb(self.image_labeled * int(255 / self.image_labeled.max())),

        ])

        third_row = np.hstack([
            utils.to_rgb(self.image_text_masks),
            self.image_text_regions,
            np.zeros_like(self.image_text_regions),
            np.zeros_like(self.image_text_regions)
        ])

        # fourth_row = np.hstack([
        #
        # ])

        self.image_visualization = np.vstack([first_row, second_row, third_row])

    def _update_timestamp(self, message: str):
        logger.debug(f"{message} --- {(time.time() - self.timestamp)} sec ---")
        self.timestamp = time.time()


def main() -> NoReturn:
    base_folder = "D:/pharmapack/"
    database = "Enrollment"
    src_folder = base_folder + database + "/cropped/"

    images_path = utils.get_images_list(src_folder)

    for image_path in images_path[50::25]:
        file_info = get_file_info(image_path, database)

        # if file_info.angle != 360:
        #     continue

        detect = DetectTextRegion(image_path, DetectTextRegion.PreprocessMethod.EdgeExtractionAndFiltration)

        detect.detect_text_regions()
        display(detect.image_visualization)
        detect.write_results(base_folder, database)


if __name__ == '__main__':
    main()
