import cv2 as cv
import numpy as np
import skimage.measure

import pathlib
import time
import os
import logging

from copy import deepcopy
from typing import List, Tuple, NoReturn

import utils
from utils import display
from text_detection.morph import Morph
from text_detection.types import PreprocessMethod, _rrect


logger = utils.get_logger(__name__, logging.DEBUG)


class Detect:

    class Flags:

        def __init__(self, visualize: bool, current_iteration: int = 1, last_iteration: int = 2):
            self.visualize: bool = visualize
            self.current_iteration: int = current_iteration
            self.last_iteration: int = last_iteration

    def __init__(self, image_orig: np.ndarray, preprocess_method: PreprocessMethod, flags: Flags):
        self.image_orig: np.ndarray = image_orig
        self.preprocess_method = preprocess_method
        self.flags = flags

        self.timestamp = time.time()

        self.image_gray: np.ndarray = cv.cvtColor(self.image_orig, cv.COLOR_BGR2GRAY)

        self.is_mask_partial: bool = False
        self.image_mask: np.ndarray = np.zeros_like(self.image_gray)

        self.morph_start_angle: int = -90

        self.image_bw: np.ndarray = np.zeros_like(self.image_gray)
        self.image_std_filtered: np.ndarray = utils.std_filter(self.image_gray, 5)

        self.image_visualization: np.ndarray = np.zeros((self.image_gray.shape[0] * 4, self.image_gray.shape[1] * 4))

        self.image_preprocessed: np.ndarray = np.zeros_like(self.image_gray)

        self.image_filled: np.ndarray = np.zeros_like(self.image_gray)

        self.image_cleared_borders: np.ndarray = np.zeros_like(self.image_gray)
        self.image_cleared_enclosed: np.ndarray = np.zeros_like(self.image_gray)

        self.image_filtered: np.ndarray = np.zeros_like(self.image_gray)

        self.image_segmented: np.ndarray = np.zeros_like(self.image_gray)

        self.image_linearly_morphed: np.ndarray = np.zeros_like(self.image_gray)
        self.image_labeled: np.ndarray = np.zeros_like(self.image_gray)

        self.image_text_masks: np.ndarray = np.zeros_like(self.image_gray)
        self.image_text_regions: np.ndarray = deepcopy(self.image_orig)

        self.rrects: List[_rrect] = list()
        self.coordinates: List[np.ndarray] = list()
        self.coordinates_ravel: List[np.ndarray] = list()

        self.text_regions: List[np.ndarray] = list()

        self._update_timestamp("CLASS INITIALIZATION")

    def detect_text_regions(self) -> NoReturn:

        self.image_mask, self.image_bw, self.general_rotation, self.is_mask_partial = \
            Morph.find_package_mask_and_angle(self.image_std_filtered)

        if self.is_mask_partial:
            self._apply_mask()

        self._update_timestamp("MASK OPERATIONS")

        self._preprocess()
        self._update_timestamp("PREPROCESS")

        self.image_cleared_borders = utils.clear_borders(self.image_preprocessed)

        self.image_cleared_enclosed = Morph.filter_enclosed_contours(self.image_cleared_borders)

        self.image_filled = utils.fill_holes(self.image_cleared_borders)

        self.image_filtered = Morph.filter_non_text_blobs(self.image_filled)
        self._update_timestamp("BORDER, FILLING, FILTERING")

        self.image_segmented = utils.apply_watershed(self.image_orig, self.image_filtered)

        self.image_linearly_morphed = Morph.apply_line_morphology(self.image_filtered, scale=1)

        self._update_timestamp("LINE MORPHOLOGY")

        self._find_text_regions()
        self._update_timestamp("FIND TEXT REGIONS")

        if self.flags.current_iteration != self.flags.last_iteration:
            self._split_text_regions()

        self._draw_text_regions_and_mask()

        if self.flags.visualize:
            self._create_visualization()

    def detect_text_regions_second(self) -> NoReturn:
        display(self.image_orig)

        self._preprocess()
        self._update_timestamp("PREPROCESS")

        self.image_cleared_borders = utils.clear_borders(self.image_preprocessed)

        self.image_cleared_enclosed = Morph.filter_enclosed_contours(self.image_cleared_borders)

        self.image_filled = utils.fill_holes(self.image_cleared_borders)

        self.image_filtered = Morph.filter_non_text_blobs(self.image_filled)
        self._update_timestamp("BORDER, FILLING, FILTERING")


    def _apply_mask(self) -> NoReturn:
        self.image_gray = self.image_gray * self.image_mask
        self.image_orig = self.image_orig * utils.to_rgb(self.image_mask)

        self.image_bw = self.image_bw * self.image_mask
        self.image_std_filtered = self.image_std_filtered * self.image_mask

    def _preprocess(self) -> NoReturn:
        if self.preprocess_method == PreprocessMethod.BasicMorphology:
            self.image_preprocessed = Morph.apply_basic_morphology(self.image_bw)
        else:
            self.image_preprocessed = Morph.extract_edges(self.image_std_filtered, self.image_bw, self.preprocess_method)

    def _find_text_regions(self) -> NoReturn:
        _, self.image_labeled = cv.connectedComponents(self.image_linearly_morphed)

        for k in range(1, self.image_labeled.max() + 1):
            image_blob = (self.image_labeled == k).astype(np.uint8) * 255
            contours, _ = cv.findContours(image_blob, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            rrect = cv.minAreaRect(contours[0])
            points = np.int0(cv.boxPoints(rrect))
            x, y, w, h = cv.boundingRect(contours[0])
            image_text_region = cv.copyTo(self.image_orig, image_blob)

            image_text_region = image_text_region[y:y+h, x:x+w, :]

            self.text_regions.append(image_text_region)
            self.rrects.append(rrect)
            self.coordinates.append(points)
            self.coordinates_ravel.append(np.transpose(points).ravel())

    def _split_text_regions(self) -> NoReturn:
        for text_region in self.text_regions:
            upd_flags = deepcopy(self.flags)
            upd_flags.current_iteration += 1
            detect = Detect(text_region, self.preprocess_method, upd_flags)
            detect.detect_text_regions_second()

    def _draw_text_regions_and_mask(self, color: int = 255) -> NoReturn:
        for coord in self.coordinates:
            cv.drawContours(self.image_text_regions, [coord], -1, (color, 0, 0), 2)
            cv.drawContours(self.image_text_masks, [coord], -1, color, -1)

    def _create_visualization(self) -> NoReturn:
        def add_text(image: np.ndarray, text: str) -> np.ndarray:
            return cv.putText(image, text, (50, 50), cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 1)

        first_row = np.hstack([
            add_text(self.image_orig, "orig"),
            add_text(utils.to_rgb(self.image_mask * 255), "mask"),
            add_text(utils.to_rgb(self.image_preprocessed), "preproc, edges"),
            add_text(utils.to_rgb(self.image_cleared_borders), "clear borders")
        ])

        second_row = np.hstack([
            add_text(utils.to_rgb(self.image_filled), "filled"),
            add_text(utils.to_rgb(self.image_filtered), "filtered blobs"),
            add_text(utils.to_rgb(self.image_linearly_morphed), "morphed"),
            add_text(utils.to_rgb(self.image_labeled * int(255 / self.image_labeled.max())), "labeled"),

        ])

        third_row = np.hstack([
            add_text(utils.to_rgb(self.image_text_masks), "text masks"),
            add_text(self.image_text_regions, "text regions"),
            add_text(np.zeros_like(self.image_text_regions), "empty"),
            add_text(np.zeros_like(self.image_text_regions), "empty")
        ])

        # fourth_row = np.hstack([
        #
        # ])

        self.image_visualization = np.vstack([first_row, second_row, third_row])

    def _update_timestamp(self, message: str):
        logger.debug(f"{message} --- {(time.time() - self.timestamp)} sec ---")
        self.timestamp = time.time()
