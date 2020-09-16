import logging
import random
from typing import NoReturn, Dict, Union, List, Tuple

import cv2 as cv
import numpy as np

from textdetector import config
from textdetector.detector import Detector, DetectionResult
from textdetector.annotation import Annotation
from textdetector.enums import EvalMetric, AnnotationLabel, AlignmentMethod
import utils


logger = logging.getLogger('evaluator')


class Evaluator:

    def __init__(self, annotation: Annotation) -> NoReturn:
        self.annotation: Annotation = annotation

        self.image_mask_ref_text: np.ndarray = utils.to_gray(self.annotation.create_mask_by_labels(
            labels=AnnotationLabel.get_list_of_text_labels(), color=(255, 255, 255)
        ))
        self.image_mask_ref_graphic: np.ndarray = utils.to_gray(self.annotation.create_mask_by_labels(
            labels=AnnotationLabel.get_list_of_graphic_labels(), color=(255, 255, 255)
        ))

        self.homo_mat: Union[None, np.ndarray] = None

        self.results_mask: Dict[str, Dict[str, Dict[str, float]]] = dict()
        self.results_mask_aggr: Dict[str, float] = dict()

        self.results_regions: Dict[str, List[List[Dict[str, float]]]] = dict()

    def evaluate(self, detection: Detector) -> NoReturn:
        if config.need_warp():
            self.homo_mat = utils.find_homography_matrix(
                utils.to_gray(detection.image_not_scaled), utils.to_gray(self.annotation.image_ref)
            )

        for algorithm, result in detection.results.items():
            image_ver_mask = result.get_default_mask()
            regions_ver = result.get_default_regions()

            if config.need_warp():
                regions_polygons = [
                    cv.perspectiveTransform(
                        region.polygon.astype(np.float32),
                        self.homo_mat
                    ).astype(np.int32) for region in regions_ver
                ]
            else:
                regions_polygons = [region.polygon.astype(np.int32) for region in regions_ver]

            self.results_mask[algorithm] = self._evaluate_by_mask(image_ver_mask)
            self.results_regions[algorithm] = self._evaluate_by_regions(regions_polygons)

        self._aggregate_results()


    def _evaluate_by_mask(self, image_ver: np.ndarray) -> Dict[str, Dict[str, float]]:
        if config.need_warp():
            image_ver = cv.warpPerspective(
                image_ver, self.homo_mat, utils.swap_dimensions(self.annotation.image_ref.shape)
            )

        result = dict()
        result['ALL'] = self.calc_all_metrics(image_ver, self.image_mask_ref_text)
        result['TXT'] = self.calc_all_metrics(cv.copyTo(image_ver, ~self.image_mask_ref_graphic), self.image_mask_ref_text)

        return result

    def _evaluate_by_regions(self, region_polygons: List[np.ndarray]) -> List[List[Dict[str, float]]]:
        results = list()

        for brect in self.annotation.get_brects_by_labels(AnnotationLabel.get_list_of_text_labels()):
            brect_results = list()

            for polygon in region_polygons:
                if utils.contour_intersect(brect.to_polygon(), polygon, edges_only=False):
                    img_empty = np.zeros(self.annotation.image_ref.shape[:2], dtype=np.uint8)

                    image_mask_ver = cv.drawContours(np.copy(img_empty), [polygon], -1, 255, -1)
                    image_mask_ref = cv.drawContours(np.copy(img_empty), [brect.to_polygon()], -1, 255, -1)

                    brect_results.append(self.calc_all_metrics(image_mask_ver, image_mask_ref))

            if not brect_results:
                brect_results.append(self.generate_negative_result())

            results.append(brect_results)

        return results


    def get_mask_results_aggregated(self) -> Dict[str, float]:
        dict_result = dict()

        for metric, alg_score_t in self.results_mask_aggr.items():
            _, score = alg_score_t
            dict_result[f'MIN_{metric}'] = score

        return dict_result

    def get_mask_results(self) -> Dict[str, float]:
        dict_result = self.get_mask_results_aggregated()

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
    def generate_negative_result(cls) -> Dict[str, int]:
        calc_dict = dict()
        for metric in EvalMetric.to_list():
            calc_dict[metric.vs()] = np.round(np.random.uniform(-1.25, -0.75), 3)
        return calc_dict

    @classmethod
    def calc_all_metrics(
            cls,
            image_ver: np.ndarray,
            image_ref: np.ndarray,
    ) -> Dict[str, float]:
        calc_dict = dict()

        tp, tn, fp, fn = cls.calc_confusion(image_ver, image_ref)

        if tp == 0:
            return cls.generate_negative_result()

        # cdict[EvalMetric.TruePositive.vs()] = tp / cv.countNonZero(image_ver)
        # cdict[EvalMetric.TrueNegative.vs()] = tn / cv.countNonZero(~image_ver)
        # cdict[EvalMetric.FalsePositive.vs()] = fp / cv.countNonZero(image_ver)
        # cdict[EvalMetric.FalseNegative.vs()] = fn / cv.countNonZero(~image_ver)

        calc_dict[EvalMetric.IntersectionOverUnion.vs()] = cls.calc_intersection_over_union(image_ver, image_ref)

        calc_dict[EvalMetric.Accuracy.vs()] = (tp + tn) / (tp + fp + tn + fn)
        calc_dict[EvalMetric.Sensitivity.vs()] = tp / (tp + fn)
        calc_dict[EvalMetric.Precision.vs()] = tp / (tp + fp)
        calc_dict[EvalMetric.Specificity.vs()] = tn / (tn + fp)

        for key, value in calc_dict.items():
            calc_dict[key] = np.round(value, 3)

        return calc_dict

    def _aggregate_results(self) -> NoReturn:
        self.results_mask_aggr = dict(zip([
            EvalMetric.IntersectionOverUnion.vs(),
            EvalMetric.Accuracy.vs(),
            EvalMetric.Sensitivity.vs(),
            EvalMetric.Precision.vs(),
            EvalMetric.Specificity.vs()
        ], [('any', 10000000)] * 5))

        for metric in self.results_mask_aggr.keys():
            for algorithm in self.results_mask.keys():
                score = self.results_mask[algorithm]['ALL'][metric]
                if score < self.results_mask_aggr[metric][1]:
                    self.results_mask_aggr[metric] = (algorithm, score)

