import cv2 as cv
import numpy as np
import abc

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


class DetectTextRegion:

    class Flags:

        def __init__(self, visualize: bool, time_profiling: bool):
            self.visualize: bool = visualize
            self.time_profiling: bool = time_profiling

    def __init__(self, image_orig: np.ndarray, preprocess_method: PreprocessMethod, flags: Flags):
        self.image_orig: np.ndarray = image_orig
        self.preprocess_method = preprocess_method
        self.flags = flags

        self.timestamp = time.time()

        self.image_gray: np.ndarray = cv.cvtColor(self.image_orig, cv.COLOR_BGR2GRAY)

        self.is_mask_partial: bool = False
        self.image_mask: np.ndarray = np.zeros_like(self.image_gray)

        self.image_bw: np.ndarray = np.zeros_like(self.image_gray)
        self.image_std_filtered: np.ndarray = utils.std_filter(self.image_gray, 5)

        self.image_visualization: np.ndarray = np.zeros((self.image_gray.shape[0] * 4, self.image_gray.shape[1] * 4))

        self.image_preprocessed: np.ndarray = np.zeros_like(self.image_gray)
        self.image_filled: np.ndarray = np.zeros_like(self.image_gray)
        self.image_cleared_borders: np.ndarray = np.zeros_like(self.image_gray)
        self.image_filtered: np.ndarray = np.zeros_like(self.image_gray)
        self.image_segmented: np.ndarray = np.zeros_like(self.image_gray)

        self.image_text_linearly_morphed: np.ndarray = np.zeros_like(self.image_gray)
        self.image_text_labeled: np.ndarray = np.zeros_like(self.image_gray)
        self.image_text_masks: np.ndarray = np.zeros_like(self.image_gray)
        self.image_text_regions: np.ndarray = deepcopy(self.image_orig)

        self.image_word_linearly_morphed: np.ndarray = np.zeros_like(self.image_gray)
        self.image_word_labeled: np.ndarray = np.zeros_like(self.image_gray)
        self.image_word_masks: np.ndarray = np.zeros_like(self.image_gray)
        self.image_word_regions: np.ndarray = deepcopy(self.image_orig)

        self.text_regions: List[TextRegion] = list()
        self.word_regions: List[TextRegion] = list()

        self._update_timestamp("CLASS INITIALIZATION")

    def detect_text_regions(self) -> NoReturn:

        self.image_mask, self.image_bw, self.is_mask_partial = Morph.find_package_mask(self.image_std_filtered)

        if self.is_mask_partial:
            self._apply_mask()

        self._update_timestamp("MASK OPERATIONS")

        self.image_preprocessed = Morph.extract_edges(self.image_std_filtered, self.image_bw)
        self._update_timestamp("EXTRACT EDGES")

        self.image_cleared_borders = utils.clear_borders(self.image_preprocessed)

        self.image_filled = utils.fill_holes(self.image_cleared_borders)

        self.image_filtered = Morph.filter_non_text_blobs(self.image_filled)
        self._update_timestamp("BORDER, FILLING, FILTERING")

        self.image_segmented = utils.apply_watershed(self.image_orig, self.image_filtered)

        self.image_text_linearly_morphed = Morph.apply_line_morphology(self.image_filtered, scale=1)

        self._update_timestamp("LINE MORPHOLOGY")

        self._find_text_regions()
        self._update_timestamp("FIND TEXT REGIONS")
        self._draw_text_regions_and_mask()

        self._detect_words_2()

        self._find_word_regions()

        self._draw_word_regions_and_mask()

        if self.flags.visualize:
            self._create_visualization()

    def _apply_mask(self) -> NoReturn:
        self.image_gray = self.image_gray * self.image_mask
        self.image_orig = self.image_orig * utils.to_rgb(self.image_mask)

        self.image_bw = self.image_bw * self.image_mask
        self.image_std_filtered = self.image_std_filtered * self.image_mask

    def _find_text_regions(self) -> NoReturn:
        _, self.image_text_labeled = cv.connectedComponents(self.image_text_linearly_morphed)

        for k in range(1, self.image_text_labeled.max() + 1):
            image_blob = (self.image_text_labeled == k).astype(np.uint8) * 255

            text_region = TextRegion(self, image_blob)
            self.text_regions.append(text_region)

    def _find_word_regions(self) -> NoReturn:
        _, self.image_word_labeled = cv.connectedComponents(self.image_word_linearly_morphed)

        for k in range(1, self.image_word_labeled.max() + 1):
            image_blob = (self.image_word_labeled == k).astype(np.uint8) * 255

            word_region = TextRegion(self, image_blob)

            if word_region.contour_area > 400:
                self.word_regions.append(word_region)

    def _detect_words(self) -> NoReturn:

        for text_region in self.text_regions:

            edges = Morph.extract_edges(text_region.image_std_filtered, text_region.image_bw, post_dilate=False)
            edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((3, 3)))
            filtered = Morph.filter_enclosed_contours(edges)
            filled = utils.fill_holes(filtered)

            contours, _ = cv.findContours(filled, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            contours = list(filter(lambda x: cv.contourArea(x) > 200, contours))

            # variance = np.var([cv.contourArea(contour) for contour in contours])
            # if variance > 10000:
            #     continue

            points = [cv.boxPoints(cv.minAreaRect(contour)) for contour in contours if len(contour) > 4]


            distances = dict()
            for idx1, points1 in enumerate(points, start=0):
                for idx2, points2 in enumerate(points[idx1 + 1:], start=idx1 + 1):

                    min_distance = 1000000
                    for point1 in points1:
                        for point2 in points2:
                            distance = utils.calc_points_distance(point1, point2)
                            if distance < min_distance:
                                min_distance = distance

                    if idx1 in distances.keys():
                        if distances[idx1] > min_distance:
                            distances[idx1] = min_distance
                    else:
                        distances[idx1] = min_distance

            if len(contours) < 2:
                mean_distance = 25
                rotation_angle = -90
            else:
                try:
                    rotation_angle = int(np.mean([x for x in [int(cv.minAreaRect(x)[2]) for x in contours if len(x) > 3] if x % 90 != 0]))
                except:
                    rotation_angle = -90
                mean_distance = max(int(1.5 * np.mean(np.asarray([v for _, v in distances.items()]))), 10)

            rrects = [np.int0(cv.convexHull(contour)) for contour in contours]
            sss = cv.drawContours(np.zeros_like(filled), rrects, -1, 255, -1)

            morphed = Morph.apply_line_morphology_simplified(sss, rotation_angle, mean_distance)

            x, y, w, h = text_region.brect
            self.image_word_linearly_morphed[y:y + h, x:x + w] = \
                cv.bitwise_xor(morphed, self.image_word_linearly_morphed[y:y + h, x:x + w])

    def _detect_words_2(self) -> NoReturn:

        for text_region in self.text_regions:

            edges = Morph.extract_edges(text_region.image_std_filtered, text_region.image_bw, post_dilate=False)
            edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((3, 3)))
            filtered = Morph.filter_enclosed_contours(edges)
            filled = utils.fill_holes(filtered)
            utils.display(filled)
            contours, _ = cv.findContours(filled, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            contours = list(filter(lambda x: cv.contourArea(x) > 200 or len(x) > 4, contours))

            def extract_extreme(c):
                left = tuple(c[c[:, :, 0].argmin()][0])
                right = tuple(c[c[:, :, 0].argmax()][0])
                top = tuple(c[c[:, :, 1].argmin()][0])
                bottom = tuple(c[c[:, :, 1].argmax()][0])
                return [left, right, top, bottom]

            points = [cv.convexHull(contour) for contour in contours if len(contour) > 4]

            distances = dict()
            for idx1, points1 in enumerate(points, start=0):
                for idx2, points2 in enumerate(points[idx1 + 1:], start=idx1 + 1):

                    min_distance = 1000000
                    for point1 in points1:
                        for point2 in points2:
                            distance = np.linalg.norm(point1[0] - point2[0])
                            # print(f"{point1[0]=} {point2[0]=} {distance=}")
                            if distance < min_distance:
                                min_distance = distance

                    if idx1 in distances.keys():
                        if distances[idx1][0] > min_distance:
                            distances[idx1] = (min_distance, idx2)
                    else:
                        distances[idx1] = (min_distance, idx2)


            # TODO Combine contours
            if len(contours) < 2:
                mean_distance = 25
                rotation_angle = -90
            else:
                try:
                    rotation_angle = int(np.mean([x for x in [int(cv.minAreaRect(x)[2]) for x in contours if len(x) > 3] if x % 90 != 0]))
                except:
                    rotation_angle = -90
                mean_distance = np.ceil(np.mean(np.asarray([v for _, v in distances.items()])))

            to_morph = cv.drawContours(np.zeros_like(filled), contours, -1, 255, -1)

            morphed = Morph.apply_line_morphology_simplified(to_morph, rotation_angle, mean_distance)
            utils.display(morphed)
            x, y, w, h = text_region.brect
            self.image_word_linearly_morphed[y:y + h, x:x + w] = \
                cv.bitwise_xor(morphed, self.image_word_linearly_morphed[y:y + h, x:x + w])



    def _draw_text_regions_and_mask(self, color: int = 255) -> NoReturn:
        for text_region in self.text_regions:
            cv.drawContours(self.image_text_regions, [text_region.coordinates], -1, (color, 0, 0), 2)
            cv.drawContours(self.image_text_masks, [text_region.coordinates], -1, color, -1)

    def _draw_word_regions_and_mask(self, color: int = 255) -> NoReturn:
        for word_region in self.word_regions:
            cv.drawContours(self.image_word_regions, [word_region.coordinates], -1, (color, 0, 0), 2)
            cv.drawContours(self.image_word_masks, [word_region.coordinates], -1, color, -1)

    def _create_visualization(self) -> NoReturn:
        def add_text(image: np.ndarray, text: str) -> np.ndarray:
            return cv.putText(image, text, (50, 50), cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 1)

        first_row = np.hstack([
            add_text(self.image_orig, "orig"),
            add_text(utils.to_rgb(self.image_mask * 255), "mask"),
            add_text(utils.to_rgb(self.image_preprocessed), "edges"),
            add_text(utils.to_rgb(self.image_cleared_borders), "clear borders")
        ])

        second_row = np.hstack([
            add_text(utils.to_rgb(self.image_filled), "filled"),
            add_text(utils.to_rgb(self.image_filtered), "filtered blobs"),
            add_text(utils.to_rgb(self.image_text_linearly_morphed), "morphed text"),
            add_text(utils.to_rgb(self.image_text_labeled * int(255 / self.image_text_labeled.max())), "labeled text"),

        ])

        third_row = np.hstack([
            add_text(utils.to_rgb(self.image_text_masks), "text masks"),
            add_text(self.image_text_regions, "text regions"),
            add_text(utils.to_rgb(np.zeros_like(self.image_word_linearly_morphed)), "empty"),
            add_text(utils.to_rgb(np.zeros_like(self.image_text_labeled)), "empty")
        ])

        fourth_row = np.hstack([
            add_text(utils.to_rgb(self.image_word_linearly_morphed), "morphed words"),
            add_text(utils.to_rgb(self.image_word_labeled * int(255 / self.image_word_labeled.max())), "labeled words"),
            add_text(utils.to_rgb(self.image_word_masks), "word masks"),
            add_text(self.image_word_regions, "word regions"),
        ])

        self.image_visualization = np.vstack([first_row, second_row, third_row, fourth_row])

    def _update_timestamp(self, message: str):
        logger.debug(f"{message} --- {(time.time() - self.timestamp)} sec ---")
        self.timestamp = time.time()


class TextRegion:

    def __init__(self, detect_text_region: DetectTextRegion, image_blob: np.ndarray):

        self.image_blob = image_blob
        self.contour = cv.findContours(image_blob, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0][0]

        self.rrect = cv.minAreaRect(self.contour)
        self.coordinates: np.ndarray = np.int0(cv.boxPoints(self.rrect))
        self.brect = cv.boundingRect(self.contour)

        self.rrect_area = self.rrect[1][0] * self.rrect[1][1]
        self.contour_area = cv.contourArea(self.contour)


#         self.coordinates_ravel.append(np.transpose(points).ravel())

        self.image_orig = self._crop_by_brect(detect_text_region.image_orig)
        self.image_gray = self._crop_by_brect(detect_text_region.image_gray)
        self.image_preprocessed = self._crop_by_brect(detect_text_region.image_preprocessed)
        self.image_std_filtered = self._crop_by_brect(detect_text_region.image_std_filtered)
        self.image_bw = self._crop_by_brect(detect_text_region.image_bw)

        # TODO continue needed copying images

    def _crop_by_brect(self, image):
        x, y, w, h = self.brect

        if len(image.shape) == 3:
            return cv.copyTo(image, self.image_blob)[y:y + h, x:x + w, :]
        elif len(image.shape) == 2:
            return cv.copyTo(image, self.image_blob)[y:y + h, x:x + w]

