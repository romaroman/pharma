import logging
import itertools
from typing import List, NoReturn, Dict, Tuple, Union

import cv2 as cv
import numpy as np

from textdetector import morph, config
from textdetector.enums import DetectionAlgorithm

import utils


logger = logging.getLogger('detector')


class Detector:

    def __init__(self, image_input: np.ndarray) -> NoReturn:
        self.is_image_aligned, self.image_not_scaled = morph.align_package_to_corners(np.copy(image_input))
        self.image_not_scaled = morph.prepare_image(self.image_not_scaled, config.scale_factor)

        self.image_rgb = morph.mscale(self.image_not_scaled)
        self.image_gray: np.ndarray = cv.cvtColor(self.image_rgb, cv.COLOR_BGR2GRAY)
        self.image_bw = utils.thresh(self.image_gray)

        self.is_mask_partial: bool = False
        self.image_mask: np.ndarray = np.zeros_like(self.image_gray)

        self.image_std_filtered: np.ndarray = utils.std_filter(self.image_gray, 6)

        self.image_edges: np.ndarray = np.zeros_like(self.image_gray)
        self.image_filled: np.ndarray = np.zeros_like(self.image_gray)
        self.image_cleared_borders: np.ndarray = np.zeros_like(self.image_gray)
        self.image_filtered: np.ndarray = np.zeros_like(self.image_gray)
        self.image_morphed: np.ndarray = np.zeros_like(self.image_gray)

        self.results: Dict[str, Tuple[np.ndarray, List['Region']]] = dict()

    def detect(self, algorithms: List[DetectionAlgorithm]) -> NoReturn:
        self.image_edges = morph.extract_edges(self.image_std_filtered, self.image_bw, post_morph=True)
        self.image_cleared_borders = morph.clear_borders(self.image_edges)

        self.image_filled = utils.fill_holes(self.image_cleared_borders)
        self.image_filtered = morph.filter_non_text_blobs(self.image_filled)

        if DetectionAlgorithm.MorphologyIteration1 in algorithms:
            self.image_morphed = morph.apply_line_morphology(self.image_filtered, morph.mscale(30))[1]
            self._save_result(DetectionAlgorithm.MorphologyIteration1.name, self.image_morphed)

        if DetectionAlgorithm.MorphologyIteration2 in algorithms:
            image_iteration2_result = self._detect_words(self.get_result_by_algorithm(DetectionAlgorithm.MorphologyIteration1)[1])
            self._save_result(DetectionAlgorithm.MorphologyIteration2.name, image_iteration2_result)

        if DetectionAlgorithm.LineSegmentation in algorithms:
            image_lines_v = self._process_lines(morph.apply_rectangular_segmentation(self.image_filtered, axis=0))
            image_lines_h = self._process_lines(morph.apply_rectangular_segmentation(self.image_filtered, axis=1))
            self._save_result(DetectionAlgorithm.LineSegmentation.name, cv.bitwise_or(image_lines_h, image_lines_v))

        if DetectionAlgorithm.MSER in algorithms:
            image_MSER_bw = self._get_MSER_mask()
            self._save_result(DetectionAlgorithm.MSER.name, image_MSER_bw)

        if DetectionAlgorithm.MajorVoting in algorithms:
            for algorithm, mask in self._perform_major_voting().items():
                self._save_result(algorithm, mask)

    def to_dict(self) -> Dict[str, Dict[int, Dict[str, Union[int, float]]]]:
        dict_to_dump = dict()

        for algorithm, result in self.results.items():
            _, regions = result

            dict_regions = dict()
            for index, region in enumerate(regions, start=1):
                dict_regions[index] = region.to_dict()

            dict_to_dump[str(algorithm)] = dict_regions

        return dict_to_dump

    def get_result_by_algorithm(self, algorithm: DetectionAlgorithm) -> Tuple[np.ndarray, List["Region"]]:
        return self.results[str(algorithm)]

    @staticmethod
    def get_coordinates_from_regions(text_regions: List['Region']) -> List[np.ndarray]:
        return [text_region.coordinates_ravel for text_region in text_regions]

    def _apply_mask(self) -> NoReturn:
        self.image_gray = self.image_gray * self.image_mask
        self.image_rgb = self.image_rgb * utils.to_rgb(self.image_mask)

        self.image_bw = self.image_bw * self.image_mask
        self.image_std_filtered = self.image_std_filtered * self.image_mask

    def _process_lines(self, lines: np.ndarray) -> np.ndarray:
        image_bw: np.ndarray = np.zeros_like(self.image_gray)

        for line_region in lines:
            try:
                x_min, x_max, y_min, y_max = line_region[0][1], line_region[1][1], line_region[0][0], line_region[1][0]

                if x_min > x_max:
                    x_max, x_min = x_min, x_max

                if x_max - x_min < morph.mscale(15):
                    continue

                image_filtered = self.image_filtered[x_min:x_max, y_min:y_max]

                contours, _ = cv.findContours(image_filtered, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                min_sides = list()
                for contour in contours:
                    x, y, w, h = cv.boundingRect(contour)
                    min_sides.append(min(w, h))

                mean_min_side = np.mean(min_sides)
                morph_line_length = int(mean_min_side / 1.5)

                angle, image_morphed = morph.apply_line_morphology(image_filtered, morph_line_length, key='min')

                image_bw[x_min:x_max, y_min:y_max] = cv.bitwise_xor(image_morphed, image_bw[x_min:x_max, y_min:y_max])
            except:
                pass

        return image_bw

    def _find_regions(self, image_bw: np.ndarray) -> Tuple[List['Region'], np.ndarray]:
        _, image_text_labeled = cv.connectedComponents(image_bw)
        regions = list()

        for k in range(1, image_text_labeled.max() + 1):
            image_blob = (image_text_labeled == k).astype(np.uint8) * 255

            region = Region(self, image_blob)

            if region.contour_area > morph.mscale(25) ** 2:
                regions.append(region)

        return regions, image_text_labeled

    def _detect_words(self, regions: List['Region']) -> np.ndarray:
        image_word_linearly_morphed = np.zeros(self.image_not_scaled.shape[:2], dtype=np.uint8)

        for region in regions:
            edges = region.crop_by_brect(morph.mscale(self.image_edges, down=False))
            edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((3, 3)))

            filtered = morph.filter_enclosed_contours(edges)
            filled = utils.fill_holes(filtered)

            contours, _ = cv.findContours(filled, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours = list(filter(lambda x: cv.contourArea(x) > morph.mscale(200), contours))

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
                mean_distance = morph.mscale(25)
            else:
                mean_distance = max(
                    int(1.5 * np.mean(np.asarray([v for _, v in distances.items()]))),
                    morph.mscale(20)
                )

            _, morphed = morph.apply_line_morphology(filled, mean_distance, key='min')
            x, y, w, h = region.brect

            image_word_linearly_morphed[y:y + h, x:x + w] = cv.bitwise_xor(
                morphed, image_word_linearly_morphed[y:y + h, x:x + w]
            )
        return morph.mscale(image_word_linearly_morphed)

    def _draw_regions_and_mask(
            self,
            regions: List['Region'],
            color: int = 255
    ) -> Tuple[np.ndarray, np.ndarray]:

        image_regions = np.copy(self.image_rgb)
        image_masks = np.zeros_like(self.image_gray)

        for region in regions:
            cv.drawContours(image_regions, [region.coordinates], -1, (color, 0, 0), 2)
            cv.drawContours(image_masks, [region.coordinates], -1, 255, -1)

        return image_regions, image_masks

    def create_visualization(self) -> NoReturn:

        def add_text(image: np.ndarray, text: str) -> np.ndarray:
            return cv.putText(np.copy(image), text, (50, 50), cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 1)

        images = [
            add_text(self.image_rgb, "orig"),
            add_text(self.image_mask * 255, "mask"),
            add_text(self.image_edges, "edges"),
            add_text(self.image_cleared_borders, "clear borders"),
            add_text(self.image_filled, "filled"),
            add_text(self.image_filtered, "filtered blobs"),
        ]

        for blob, result in self.results.items():
            image_mask, _ = result
            images.append(add_text(image_mask, blob))

        for i, image in enumerate(images):
            if len(image.shape) != 3:
                images[i] = utils.to_rgb(image)

        return utils.combine_images(images)

    def _get_MSER_mask(self) -> np.ndarray:
        image_hsv = cv.cvtColor(self.image_rgb, cv.COLOR_BGR2HSV)
        _, image_channel_s, image_channel_v = cv.split(image_hsv)

        image_MSER_s = utils.MSER(image_channel_s)[0]
        image_MSER_v = utils.MSER(image_channel_v)[0]

        image_bw = cv.bitwise_xor(image_MSER_s, image_MSER_v)
        image_bw = utils.fill_holes(image_bw)

        morph.apply_line_morphology(image_bw, morph.mscale(30))

        return image_bw

    def _save_result(self, algorithm: str, image_mask: np.ndarray) -> NoReturn:
        image_mask_orig_size = morph.mscale(image_mask, down=False)
        regions, _ = self._find_regions(image_mask_orig_size)

        image_regions, image_mask = self._draw_regions_and_mask(regions)

        if algorithm in self.results.keys():
            logger.warning(f"{algorithm} result already exists, overwriting...")

        self.results[algorithm] = image_mask, regions

    def _perform_major_voting(self) -> Dict[str, np.ndarray]:
        resulting_masks = dict()

        combinations = list(itertools.combinations(list(self.results.keys()), 3))
        for combination in combinations:
            alg1, alg2, alg3 = combination

            mask1, mask2, mask3 = self.results[alg1][0], self.results[alg2][0], self.results[alg3][0]

            alg12 = cv.bitwise_and(mask1, mask1)
            alg13 = cv.bitwise_and(mask1, mask3)
            alg23 = cv.bitwise_and(mask2, mask3)

            resulting_masks["+".join([alg1, alg2, alg3])] = cv.bitwise_or(cv.bitwise_or(alg12, alg13), alg23)

        return resulting_masks


class Region:

    def __init__(self, detection: Detector, image_blob: np.ndarray) -> NoReturn:
        self.image_blob = np.copy(image_blob)
        self.contour = cv.findContours(image_blob, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0][0]

        self.rrect = cv.minAreaRect(self.contour)
        self.coordinates: np.ndarray = np.asarray(cv.boxPoints(self.rrect), dtype=int)
        self.coordinates_ravel: np.ndarray = np.transpose(self.coordinates).ravel()

        self.brect = cv.boundingRect(self.contour)

        self.rrect_area = self.rrect[1][0] * self.rrect[1][1]
        self.contour_area = cv.contourArea(self.contour)

        self.image_rgb = self.crop_by_brect(detection.image_not_scaled)

    def crop_by_brect(self, image: np.ndarray) -> np.ndarray:
        x, y, w, h = self.brect

        if len(image.shape) == 3:
            return cv.copyTo(image, self.image_blob)[y:y + h, x:x + w, :]
        elif len(image.shape) == 2:
            return cv.copyTo(image, self.image_blob)[y:y + h, x:x + w]

    def coordinates_to_dict(self) -> Dict[str, int]:
        return dict(zip(['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'], self.coordinates_ravel))

    def to_dict(self) -> Dict[str, Union[int, float]]:
        coords = self.coordinates_to_dict()

        for key, value in coords.items():
            coords[key] = int(value)

        return {'angle': round(self.rrect[2], 2), **coords}