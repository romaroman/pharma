import logging
import itertools
from typing import List, NoReturn, Dict

import cv2 as cv
import numpy as np

from textdetector import morph, config
from textdetector.enums import DetectionAlgorithm, ResultMethod

import utils


logger = logging.getLogger('detector')


class Detector:

    def __init__(self, image_input: np.ndarray) -> NoReturn:
        self.is_image_aligned, self.image_not_scaled = morph.align_package_to_corners(np.copy(image_input))
        self.image_not_scaled = morph.prepare_image(self.image_not_scaled, config.scale_factor)

        self.image_rgb = morph.mscale(self.image_not_scaled)
        self.image_gray: np.ndarray = cv.cvtColor(self.image_rgb, cv.COLOR_BGR2GRAY)
        self.image_bw = utils.thresh(self.image_gray)

        self.image_std_filtered: np.ndarray = utils.std_filter(self.image_gray, 6)

        self.image_edges: np.ndarray = np.zeros_like(self.image_gray)
        self.image_filled: np.ndarray = np.zeros_like(self.image_gray)
        self.image_cleared_borders: np.ndarray = np.zeros_like(self.image_gray)
        self.image_filtered: np.ndarray = np.zeros_like(self.image_gray)

        self.results: Dict[str, DetectionResult] = dict()

    def detect(self, algorithms: List[DetectionAlgorithm]) -> NoReturn:
        self.image_edges = morph.extract_edges(self.image_std_filtered, self.image_bw, post_morph=True)
        self.image_cleared_borders = morph.clear_borders(self.image_edges)
        self.image_filled = utils.fill_holes(self.image_cleared_borders)
        self.image_filtered = morph.filter_non_text_blobs(self.image_filled)

        self.results['MI0'] = DetectionResult(self.image_filtered)

        for algorithm in algorithms:
            if algorithm is not DetectionAlgorithm.MajorVoting:
                self.results[algorithm.vs()] = DetectionResult(self._run_algorithm(algorithm))
            else:
                if DetectionAlgorithm.MajorVoting in algorithms:
                    for algorithm, mask in self._perform_major_voting().items():
                        self.results[algorithm] = DetectionResult(mask)

    def _run_algorithm(self, algorithm: DetectionAlgorithm) -> np.ndarray:
        if algorithm is DetectionAlgorithm.MorphologyIteration1:
            return morph.apply_line_morphology(self.image_filtered, morph.mscale(30))[1]
        if algorithm is DetectionAlgorithm.MorphologyIteration2:
            return self._detect_words(
                self.get_result_by_algorithm(DetectionAlgorithm.MorphologyIteration1).get_default_regions()
            )
        if algorithm is DetectionAlgorithm.LineSegmentation:
            image_lines_v = self._process_lines(morph.apply_rectangular_segmentation(self.image_filtered, axis=0))
            image_lines_h = self._process_lines(morph.apply_rectangular_segmentation(self.image_filtered, axis=1))
            return cv.bitwise_or(image_lines_h, image_lines_v)
        if algorithm is DetectionAlgorithm.MSER:
            return self._get_MSER_mask()

    def get_result_by_algorithm(self, algorithm: DetectionAlgorithm) -> 'DetectionResult':
        return self.results[algorithm.vs()]

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

    def _detect_words(self, regions: List['Region']) -> np.ndarray:
        image_word_linearly_morphed = np.zeros(self.image_not_scaled.shape[:2], dtype=np.uint8)

        for region in regions:
            edges = region.crop_image(morph.mscale(self.image_edges, down=False))
            edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((3, 3)))
            filled = utils.fill_holes(edges)

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

    def create_visualization(self) -> NoReturn:

        def add_text(image: np.ndarray, text: str) -> np.ndarray:
            return cv.putText(np.copy(image), text, (50, 50), cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 1)

        images = [
            add_text(self.image_rgb, "orig"),
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
        r, g, b = cv.split(self.image_rgb)

        mr = utils.MSER(r)
        mb = utils.MSER(b)
        mg = utils.MSER(g)

        image_mser = cv.bitwise_and(cv.bitwise_and(mr, mb), mg)
        image_mser = utils.fill_holes(image_mser)
        image_mser = morph.apply_line_morphology(image_mser, morph.mscale(40))[1]

        return image_mser

    def _perform_major_voting(self) -> Dict[str, np.ndarray]:
        resulting_masks = dict()

        combinations = list(itertools.combinations(list(self.results.keys()), 3))
        for combination in combinations:
            alg1, alg2, alg3 = combination

            mask1 = self.results[alg1].get_default_mask()
            mask2 = self.results[alg2].get_default_mask()
            mask3 = self.results[alg3].get_default_mask()

            alg12 = cv.bitwise_and(mask1, mask2)
            alg13 = cv.bitwise_and(mask1, mask3)
            alg23 = cv.bitwise_and(mask2, mask3)

            resulting_masks["+".join([alg1, alg2, alg3])] = cv.bitwise_or(cv.bitwise_or(alg12, alg13), alg23)

        return resulting_masks


class DetectionResult:

    class Region:

        def __init__(self, contour: np.ndarray, method: ResultMethod) -> NoReturn:
            self.polygon = self.contour_to_polygon(contour, method)
            self.polygon_area = cv.contourArea(self.polygon)

            self.polygon_ravel: np.ndarray = np.transpose(self.polygon).ravel()
            self.brect = cv.boundingRect(self.polygon)

        @classmethod
        def contour_to_polygon(cls, contour: np.ndarray, method: ResultMethod) -> np.ndarray:
            if method is ResultMethod.Brect:
                return utils.get_brect_contour(contour)
            elif method is ResultMethod.Contour:
                return contour
            elif method is ResultMethod.Rrect:
                s = cv.boxPoints(cv.minAreaRect(contour)).astype(np.int0)
                return s
            elif method is ResultMethod.Hull:
                return cv.convexHull(contour).astype(np.int0)
            elif method is ResultMethod.Approximation:
                return utils.approximate_contour(cls.contour_to_polygon(contour, ResultMethod.Hull), epsilon=0.02)

        def crop_image(self, image: np.ndarray) -> np.ndarray:
            return utils.crop_image_by_contour(image, self.polygon, False)

        def to_dict(self) -> Dict[str, int]:
            return dict(zip(['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'], self.polygon_ravel))

    def __init__(self, image_input: np.ndarray) -> NoReturn:
        self.image_input = morph.mscale(image_input, down=False)

        self.masks: Dict[ResultMethod, np.ndarray] = dict()
        self.regions: Dict[ResultMethod, List['Region']] = dict()

        for method in ResultMethod:
            self.regions[method] = self.find_regions(method)
            self.masks[method] = self.draw_regions(method)

    def find_regions(self, method: ResultMethod) -> List['Region']:
        contours, _ = cv.findContours(self.image_input, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return sorted(
            [self.Region(contour, method) for contour in contours],
            key=lambda x: x.polygon_area,
            reverse=True
        )

    def draw_regions(self, method: ResultMethod) -> np.ndarray:
        image_result = np.zeros_like(self.image_input)
        for polygon in [region.polygon for region in self.regions[method]]:
            cv.drawContours(image_result, [polygon], -1, 255, -1)
        return image_result

    def get_coordinates_from_regions(self, method: ResultMethod) -> List[np.ndarray]:
        return [region.polygon_ravel for region in self.regions[method]]

    def get_default_mask(self) -> np.ndarray:
        return self.masks[config.approx_method]

    def get_default_regions(self) -> List['Region']:
        return self.regions[config.approx_method]
