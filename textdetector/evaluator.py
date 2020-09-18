import logging
from typing import NoReturn, Dict, Union, List, Tuple

import cv2 as cv
import numpy as np

import config

from detector import Detector, DetectionResult
from annotation import Annotation
from enums import EvalMetric, AnnotationLabel

import utils


logger = logging.getLogger('evaluator')


class Evaluator:

    def __init__(self, annotation: Annotation) -> NoReturn:
        self.annotation: Annotation = annotation

        self.image_mask_ref_text: np.ndarray = utils.to_gray(self.annotation.create_mask_by_labels(
            labels=AnnotationLabel.get_list_of_text_labels(), color=(255, 255, 255)
        ))

        self.homo_mat: Union[None, np.ndarray] = None

        self.results_mask: Dict[str, Dict[str, Dict[str, float]]] = dict()
        self.results_regions: Dict[str, List[List[Dict[str, float]]]] = dict()

    def evaluate(self, detection: Detector) -> NoReturn:
        if config.need_warp():
            self.homo_mat = utils.find_homography_matrix(
                utils.to_gray(detection.image_not_scaled), utils.to_gray(self.annotation.image_ref)
            )

        for algorithm, result in detection.results.items():
            if config.ev_mask_used:
                image_ver_mask = result.get_default_mask()
                self.results_mask[algorithm] = self._evaluate_by_mask(image_ver_mask)

            if config.ev_regions_used:
                regions_ver = result.get_default_regions()
                self.results_regions[algorithm] = self._evaluate_by_regions(regions_ver)

    def _evaluate_by_mask(self, image_ver: np.ndarray) -> List[float]:
        if config.need_warp():
            image_ver = cv.warpPerspective(
                image_ver, self.homo_mat, utils.swap_dimensions(self.annotation.image_ref.shape)
            )

        return self.calc_all_metrics(image_ver, self.image_mask_ref_text, to_dict=False)

    def _evaluate_by_regions(self, regions: List[DetectionResult.Region]) -> np.ndarray:
        img_empty = np.zeros(self.annotation.image_ref.shape[:2], dtype=np.uint8)

        if config.need_warp():
            regions_polygons = [
                cv.perspectiveTransform(
                    region.polygon.astype(np.float32),
                    self.homo_mat
                ).astype(np.int32) for region in regions
            ]
        else:
            regions_polygons = [region.polygon.astype(np.int32) for region in regions]

        rows = []
        brect_masks = [brect.draw(np.copy(img_empty), 255, filled=True) for brect in
                      self.annotation.get_brects_by_labels(AnnotationLabel.get_list_of_text_labels())]
        region_masks = [cv.drawContours(np.copy(img_empty), [region], -1, 255, -1) for region in regions_polygons]


        for bi, brect_z in enumerate(zip(brect_masks, self.annotation.get_brects_by_labels(AnnotationLabel.get_list_of_text_labels())), start=1):
            bmask, brect = brect_z
            matched_region_number = 0

            for pmask, polygon in zip(region_masks, regions_polygons):
                if cv.countNonZero(cv.bitwise_and(bmask, pmask)) > 0:
                    matched_region_number += 1
                    scores = self.calc_all_metrics(pmask, bmask, to_dict=False)
                    rows.append([bi, matched_region_number] + scores)

            if matched_region_number == 0:
                rows.append([bi, matched_region_number] + self.generate_negative_result())

        return np.array(rows)

    def get_mask_results(self) -> Dict[str, float]:
        dict_result = dict()

        for alg, dict_alg in self.results_mask.items():
            for mode, dict_mode in dict_alg.items():
                for metric, score in dict_mode.items():
                    dict_result['_'.join([alg.upper(), mode.upper(), metric.upper()])] = score

        return dict_result

    def get_regions_results(self) -> Dict[str, List[List[Dict[str, float]]]]:
        return self.results_regions

    @classmethod
    def calc_true_positive(cls, image_ver: np.ndarray, image_ref: np.ndarray) -> int:
        return cv.countNonZero(cv.bitwise_and(image_ver, image_ref))

    @classmethod
    def calc_true_negative(cls, image_ver: np.ndarray, image_ref: np.ndarray) -> int:
        return cv.countNonZero(cv.bitwise_and(~image_ver, ~image_ref))

    @classmethod
    def calc_false_positive(cls, image_ver: np.ndarray, image_ref: np.ndarray) -> int:
        return cv.countNonZero(cv.bitwise_and(image_ver, ~image_ref))

    @classmethod
    def calc_false_negative(cls, image_ver: np.ndarray, image_ref: np.ndarray) -> int:
        return cv.countNonZero(cv.bitwise_and(~image_ver, image_ref))

    @classmethod
    def calc_intersection_over_union(cls, image_ver: np.ndarray, image_ref: np.ndarray) -> float:
        image_overlap = cv.bitwise_and(image_ver, image_ref)
        image_union = cv.bitwise_or(image_ver, image_ref)

        area_overlap = cv.countNonZero(image_overlap)
        area_union = cv.countNonZero(image_union)

        return abs(1 - area_overlap / area_union)

    @classmethod
    def calc_confusion(cls, image_ver: np.ndarray, image_ref: np.ndarray) -> Tuple[float, float, float, float]:
        return cls.calc_true_positive(image_ver, image_ref), cls.calc_true_negative(image_ver, image_ref), \
               cls.calc_false_positive(image_ver, image_ref), cls.calc_false_negative(image_ver, image_ref)

    @classmethod
    def generate_negative_result(cls) -> List[float]:
        results = list()
        for _ in EvalMetric:
            results.append(np.round(np.random.uniform(-1.25, -0.75), 3))
        return results

    @classmethod
    def calc_all_metrics(
            cls,
            image_ver: np.ndarray,
            image_ref: np.ndarray,
            to_dict: bool = True
    ) -> Union[Dict[str, float], List[float]]:
        tp, tn, fp, fn = cls.calc_confusion(image_ver, image_ref)

        if tp == 0:
            results = cls.generate_negative_result()
        else:
            iou = cls.calc_intersection_over_union(image_ver, image_ref)
            acc = (tp + tn) / (tp + fp + tn + fn)
            sen = tp / (tp + fn)
            pr = tp / (tp + fp)
            sp = tn / (tn + fp)

            results = [iou, acc, sen, pr, sp]

        for i, result in enumerate(results):
            results[i] = np.round(result, 3)

        if to_dict:
            calc_dict = dict()
            for metric in EvalMetric:
                calc_dict[metric.vs()] = results[::-1].pop()
            return calc_dict
        else:
            return results
