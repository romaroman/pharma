import os
import logging
import itertools

from pathlib import Path
from re import Pattern
from typing import List, NoReturn, Dict, Union, Tuple

import cv2 as cv
import numpy as np

import morph
import config
from enums import DetectionAlgorithm, ApproximationMethod

import utils


logger = logging.getLogger('detector')


class Detector:

    def __init__(self, image_input: np.ndarray) -> NoReturn:
        self.image_not_scaled = image_input

        self.image_rgb = morph.mscale(self.image_not_scaled)
        self.image_gray: np.ndarray = cv.cvtColor(self.image_rgb, cv.COLOR_BGR2GRAY)
        self.image_bw = utils.thresh(self.image_gray)

        self.image_std_filtered: np.ndarray = utils.std_filter(self.image_gray, 6)

        self.image_edges: np.ndarray = np.zeros_like(self.image_gray)
        self.image_filled: np.ndarray = np.zeros_like(self.image_gray)
        self.image_cleared_borders: np.ndarray = np.zeros_like(self.image_gray)
        self.image_filtered: np.ndarray = np.zeros_like(self.image_gray)

        self.visualization: Union[np.ndarray, None] = None
        self.results: Dict[str, DetectionResult] = dict()

    def detect(self, algorithms: List[DetectionAlgorithm]) -> NoReturn:
        self.image_edges = morph.extract_edges(self.image_std_filtered, self.image_bw, post_morph=True)
        self.image_cleared_borders = morph.clear_borders(self.image_edges)
        self.image_filled = utils.fill_holes(self.image_cleared_borders)
        self.image_filtered = morph.filter_non_text_blobs(self.image_filled)

        for algorithm in algorithms:
            if algorithm is not DetectionAlgorithm.MajorVoting:
                self.results[algorithm.vs()] = DetectionResult(self._run_algorithm(algorithm))
            else:
                for algorithm, mask in self._perform_major_voting().items():
                    self.results[algorithm] = DetectionResult(mask)

        pass

    def _run_algorithm(self, algorithm: DetectionAlgorithm) -> np.ndarray:
        if algorithm is DetectionAlgorithm.MorphologyIteration1:
            return self._run_morphology_iteration_1()
        if algorithm is DetectionAlgorithm.MorphologyIteration2:
            return self._run_morphology_iteration_2()
        if algorithm is DetectionAlgorithm.LineSegmentation:
            return self._run_line_segmentation()
        if algorithm is DetectionAlgorithm.MSER:
            return self._run_MSER()

    def get_result_by_algorithm(self, algorithm: DetectionAlgorithm) -> 'DetectionResult':
        return self.results[algorithm.vs()]

    def _run_morphology_iteration_1(self) -> np.ndarray:
        image_morphed = morph.apply_line_morphology(self.image_filtered, morph.mscale(30))

        def does_contour_need_additional_morphology(contour: np.ndarray) -> bool:
            _, shape, _ = cv.minAreaRect(contour)
            w, h = shape
            aspect_ratio = float(max(w, h)) / min(w, h)

            area = cv.contourArea(contour)
            stddev = np.mean(cv.meanStdDev(
                self.image_not_scaled,
                mask=cv.drawContours(np.zeros_like(image_morphed), [contour], -1, 255, 1)
            )[1])

            return aspect_ratio < 1.5 and area > morph.mscale(25) ** 2 and stddev < 25

        contours, _ = cv.findContours(image_morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        image_to_morph = morph.filter_contours(image_morphed, key=does_contour_need_additional_morphology)
        image_morphed_again = morph.apply_line_morphology(image_to_morph, morph.mscale(15))

        return cv.bitwise_or(image_morphed, image_morphed_again)

    def _process_lines(self, lines: np.ndarray) -> np.ndarray:
        image_bw: np.ndarray = np.zeros_like(self.image_gray)

        for line_region in lines:
            x_min, x_max, y_min, y_max = line_region[0][1], line_region[1][1], line_region[0][0], line_region[1][0]

            if x_min > x_max:
                x_max, x_min = x_min, x_max

            if x_max - x_min < morph.mscale(15):
                continue

            image_filtered = self.image_filtered[x_min:x_max, y_min:y_max]

            if image_filtered.shape[0] != 0 and image_filtered.shape[0] != 0:
                contours, _ = cv.findContours(image_filtered, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                min_sides = list()
                for contour in contours:
                    x, y, w, h = cv.boundingRect(contour)
                    min_sides.append(min(w, h))

                try:
                    mean_min_side = np.mean(min_sides)
                    if mean_min_side < 10:
                        raise ValueError
                    morph_line_length = int(mean_min_side / 1.5)
                except ValueError:
                    morph_line_length = 10

                image_morphed = morph.apply_line_morphology(image_filtered, morph.mscale(morph_line_length), key='min')
                image_bw[x_min:x_max, y_min:y_max] = cv.bitwise_xor(image_morphed, image_bw[x_min:x_max, y_min:y_max])

        return image_bw

    def _run_morphology_iteration_2(self) -> np.ndarray:
        image_word_linearly_morphed = np.zeros(self.image_not_scaled.shape[:2], dtype=np.uint8)

        for region in self.results['MI1'].get_default_regions():
            edges = region.crop_image(morph.mscale(self.image_edges, down=False))
            edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((3, 3)))
            filled = utils.fill_holes(edges)

            contours, _ = cv.findContours(filled, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours = list(filter(lambda x: cv.contourArea(x) > morph.mscale(12) ** 2, contours))

            points = [cv.boxPoints(cv.minAreaRect(contour)) for contour in contours if len(contour) > 4]

            if len(points) > 2:
                distances = dict()

                for idx1, points1 in enumerate(points, start=0):
                    for idx2, points2 in enumerate(points[idx1 + 1:], start=idx1 + 1):
                        min_distance = 1000000
                        for point1 in points1:
                            for point2 in points2:
                                distance = utils.calc_distance(point1, point2)
                                if distance < min_distance:
                                    min_distance = distance

                        if idx1 in distances.keys():
                            if distances[idx1] > min_distance:
                                distances[idx1] = min_distance
                        else:
                            distances[idx1] = min_distance

                mean_distance = max(
                    int(1.5 * np.mean(np.asarray([v for _, v in distances.items()]))),
                    morph.mscale(20)
                )
            else:
                mean_distance = morph.mscale(25)

            morphed = morph.apply_line_morphology(filled, mean_distance, key='min')
            x, y, w, h = region.brect

            image_word_linearly_morphed[y:y + h, x:x + w] = cv.bitwise_xor(
                morphed, image_word_linearly_morphed[y:y + h, x:x + w]
            )

        return image_word_linearly_morphed

    def get_visualization(self) -> np.ndarray:
        self.visualization = self._create_visualization()
        return self.visualization

    def _create_visualization(self) -> NoReturn:
        images = [self.image_not_scaled]

        if self.results:
            for algorithm, result in self.results.items():
                images.append(utils.add_text(result.get_default_visualization(self.image_not_scaled), algorithm))

        for i, image in enumerate(images):
            if len(image.shape) != 3:
                images[i] = utils.to_rgb(image)

        return utils.combine_images(images)

    def _run_MSER(self) -> np.ndarray:
        r, g, b = cv.split(self.image_rgb)

        mr = utils.MSER(r)
        mb = utils.MSER(b)
        mg = utils.MSER(g)

        image_mser = cv.bitwise_and(cv.bitwise_and(mr, mb), mg)
        image_mser = utils.fill_holes(image_mser)
        image_mser = morph.apply_line_morphology(image_mser, morph.mscale(30))

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

    def _run_line_segmentation(self) -> np.ndarray:
        image_lines_v = self._process_lines(morph.apply_rectangular_segmentation(self.image_filtered, axis=0))
        image_lines_h = self._process_lines(morph.apply_rectangular_segmentation(self.image_filtered, axis=1))
        return cv.bitwise_or(image_lines_h, image_lines_v)

    def save_results(self, path_parent_folder: Path) -> NoReturn:
        os.makedirs(str(path_parent_folder.resolve()), exist_ok=True)

        cv.imwrite(str(path_parent_folder / 'image_ref.png'), self.image_not_scaled)

        for algorithm, result in self.results.items():
            os.makedirs(str((path_parent_folder / algorithm).resolve()), exist_ok=True)
            cv.imwrite(str(path_parent_folder / algorithm / 'image_mask.png'), result.get_default_mask())

    @classmethod
    def load_results(cls, path_parent_folder: Path) -> Tuple[np.ndarray, Dict[str, 'DetectionResult']]:
        dict_result = dict()
        image_ref = cv.imread(str(path_parent_folder / 'image_ref.png'))

        algorithm_folders = [d for d in path_parent_folder.iterdir() if d.is_dir()]
        for algorithm_folder in algorithm_folders:
            dict_result[algorithm_folder.stem] = DetectionResult(cv.imread(str(algorithm_folder / 'image_mask.png'), 0))

        return image_ref, dict_result

    @classmethod
    def load_results_by_pattern(
            cls,
            root_folder: Path,
            pattern: Pattern
    ) -> Tuple[np.ndarray, Dict[str, 'DetectionResult']]:
        for file in root_folder.glob("*"):
            if pattern.search(str(file)):
                return cls.load_results(file)
        else:
            raise FileNotFoundError


class DetectionResult:

    class Region:

        def __init__(self, contour: np.ndarray, method: ApproximationMethod) -> NoReturn:
            self.contour = np.copy(contour)
            self.polygon = self.contour_to_polygon(contour, method).clip(min=0)
            self.brect = cv.boundingRect(self.polygon)

        @classmethod
        def contour_to_polygon(cls, contour: np.ndarray, method: ApproximationMethod) -> np.ndarray:
            if method is ApproximationMethod.Contour:
                return contour
            if method is ApproximationMethod.Brect:
                return utils.get_brect_contour(contour)
            elif method is ApproximationMethod.Rrect:
                return cv.boxPoints(cv.minAreaRect(contour)).astype(np.int32).reshape(-1, 1, 2)
            elif method is ApproximationMethod.Hull:
                return cv.convexHull(contour).astype(np.int32)
            elif method is ApproximationMethod.Approximation:
                return utils.approximate_contour(cls.contour_to_polygon(contour, ApproximationMethod.Hull), 0.02)

        def crop_image(self, image: np.ndarray) -> np.ndarray:
            return utils.crop_image_by_contour(image, self.polygon, False)

        def draw(
                self,
                image: np.ndarray,
                color: Tuple[int, int, int] = (255, 255, 255),
                filled: bool = True
        ) -> np.ndarray:
            return cv.drawContours(image, [self.polygon], -1, color, -1 if filled else 2)

        def as_nn_input(self, image_to_crop: np.ndarray) -> np.ndarray:
            image_rgb = self.crop_image(image_to_crop)
            dst_size = 256
            biggest_side = max(image_rgb.shape)

            is_h_bigger = biggest_side == image_rgb.shape[0]
            image_rgb = utils.scale_image(image_rgb, 1 / (biggest_side / dst_size))

            image_rgb = image_rgb[:dst_size, :dst_size, :]
            h, w, c = image_rgb.shape

            image_rgb_centered = np.zeros((dst_size, dst_size, c), dtype=np.uint8)
            if is_h_bigger:
                diff = (dst_size - w) // 2
                image_rgb_centered[:h, diff:w + diff, :] = image_rgb
            else:
                diff = (dst_size - h) // 2
                image_rgb_centered[diff:h + diff, :w, :] = image_rgb

            image_gray = utils.to_gray(image_rgb_centered)

            image_mask_to_fill = np.all(image_rgb_centered == [0, 0, 0], axis=-1)
            image_contour_mask_ones = (~image_mask_to_fill).astype(np.uint8)
            image_contour_mask = cv.bitwise_not(image_mask_to_fill.astype(np.uint8) * 255)
            contour = cv.findContours(image_contour_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0][0]
            image_edge_mask = cv.drawContours(np.zeros_like(image_gray), [contour], -1, 255, 2)

            def get_row_or_col_index_with_most_of_nonblack_pixels(img, axis):
                reduced_pixels_amount = cv.reduce(img.astype(np.float32), axis, cv.REDUCE_SUM).reshape(-1).astype(np.int32)
                most_frequent_length = np.argmax(np.bincount(reduced_pixels_amount))
                rows_indexes = np.where(reduced_pixels_amount == most_frequent_length)
                mean_index = np.mean(rows_indexes)
                actual_index = (np.abs(reduced_pixels_amount - mean_index)).argmin()

                return reduced_pixels_amount[actual_index]

            row_idx = get_row_or_col_index_with_most_of_nonblack_pixels(image_contour_mask_ones, 0)
            col_idx = get_row_or_col_index_with_most_of_nonblack_pixels(image_contour_mask_ones, 1)

            h, w = image_gray.shape

            row_mean = cv.mean(image_gray[h // 2 - 4:h // 2 + 4, row_idx])[0]
            col_mean = cv.mean(image_gray[col_idx, w // 2 - 4:w // 2 + 4])[0]

            mean_edge = int(cv.mean(image_gray, image_edge_mask)[0])
            mean_center = int(cv.mean(image_gray[120:130, 120:130])[0])
            mean_all = int(cv.mean(image_gray, image_contour_mask_ones)[0])

            image_filled_colrow = np.copy(image_rgb_centered)
            image_filled_colrow[image_mask_to_fill] = int(row_mean + col_mean) / 2
            image_filled_colrow = utils.add_text(image_filled_colrow, "COL&ROW", scale=1)

            image_filled_edge = np.copy(image_rgb_centered)
            image_filled_edge[image_mask_to_fill] = mean_edge
            image_filled_edge = utils.add_text(image_filled_edge, "EDGE", scale=1)

            image_filled_center = np.copy(image_rgb_centered)
            image_filled_center[image_mask_to_fill] = mean_center
            image_filled_center = utils.add_text(image_filled_center, "CENTER", scale=1)

            image_filled_all = np.copy(image_rgb_centered)
            image_filled_all[image_mask_to_fill] = mean_all
            image_filled_all = utils.add_text(image_filled_all, "ALL", scale=1)

            return utils.combine_images([image_filled_colrow, image_filled_edge, image_filled_center, image_filled_all])

    def __init__(self, image_input: np.ndarray) -> NoReturn:
        self.image_input = morph.mscale(image_input, down=False)

        self.masks: Dict[ApproximationMethod, np.ndarray] = dict()
        self.regions: Dict[ApproximationMethod, List['Region']] = dict()

        for method in config.det_approximation_methods_used:
            self.regions[method] = self.find_regions(method)
            self.masks[method] = self.create_mask(method)

    def find_regions(self, method: ApproximationMethod) -> List['Region']:
        contours, _ = cv.findContours(self.image_input, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = list(filter(lambda c: cv.contourArea(c) > morph.mscale(5) ** 2, contours))
        return sorted(
            [self.Region(contour, method) for contour in contours],
            key=lambda r: cv.contourArea(r.polygon),
            reverse=True
        )

    def create_mask(self, method: ApproximationMethod) -> np.ndarray:
        image_result = np.zeros_like(self.image_input)
        for polygon in [region.polygon for region in self.regions[method]]:
            cv.drawContours(image_result, [polygon], -1, 255, -1)
        return image_result

    def create_visualization(self, image: np.ndarray, method: ApproximationMethod) -> np.ndarray:
        return cv.drawContours(np.copy(image), [region.polygon for region in self.regions[method]], -1, (0, 255, 0), 5)

    def get_default_visualization(self, image: np.ndarray) -> np.ndarray:
        return self.create_visualization(image, config.det_approximation_method_default)

    def get_default_mask(self) -> np.ndarray:
        return self.masks[config.det_approximation_method_default]

    def get_default_regions(self) -> List['Region']:
        return self.regions[config.det_approximation_method_default]

    def get_mask(self, method: ApproximationMethod) -> np.ndarray:
        return self.masks[method]

    def get_regions(self, method: ApproximationMethod) -> List['Region']:
        return self.regions[method]
