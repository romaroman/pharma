import logging
from typing import NoReturn, Dict, Union

import cv2 as cv
import numpy as np

from textdetector.detector import Detector
from textdetector.annotation import Annotation
from textdetector.enums import EvalMetric, AnnotationLabel
import utils


logger = logging.getLogger('evaluator')


class Evaluator:

    def __init__(self) -> NoReturn:
        self.results_complete: Dict[str, Dict[str, Dict[str, float]]] = dict()
        self.results_aggregated: Dict[str, float] = dict()

    def evaluate(self, detection: Detector, annotation: Annotation) -> NoReturn:
        image_ref = annotation.image_ref
        image_ver = detection.image_not_scaled

        homo_mat = utils.find_homography_matrix(utils.to_gray(image_ver), utils.to_gray(image_ref))

        image_ref_mask_text = utils.to_gray(annotation.create_mask_by_labels(
            labels=AnnotationLabel.get_list_of_text_labels(), color=(255, 255, 255)
        ))
        image_ref_mask_graphic = utils.to_gray(annotation.create_mask_by_labels(
            labels=AnnotationLabel.get_list_of_graphic_labels(), color=(255, 255, 255)
        ))

        for algorithm, result in detection.results.items():
            image_mask_warped = cv.warpPerspective(
                result.get_default_mask(), homo_mat, utils.swap_dimensions(image_ref.shape)
            )

            self.results_complete[algorithm] = dict()
            self._calc_metrics(image_mask_warped, image_ref_mask_text, algorithm)
            self._calc_metrics(image_mask_warped, image_ref_mask_text, algorithm, ~image_ref_mask_graphic)

            self.results_complete[algorithm]['ALL'][EvalMetric.RegionsAmount.vs()] = len(result.get_default_regions())

        self._aggregate_results()

    def to_dict_aggregated(self) -> Dict[str, float]:
        dict_result = dict()

        for metric, alg_score_t in self.results_aggregated.items():
            _, score = alg_score_t
            dict_result[f'MIN_{metric}'] = score

        return dict_result

    def to_dict_complete(self) -> Dict[str, float]:
        dict_result = self.to_dict_aggregated()

        for alg, dict_alg in self.results_complete.items():
            for mode, dict_mode in dict_alg.items():
                for metric, score in dict_mode.items():
                    dict_result['_'.join([alg.upper(), mode.upper(), metric.upper()])] = score

        return dict_result

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
        image_union = cv.bitwise_xor(image_ver, image_ref)

        area_overlap = cv.countNonZero(image_overlap)
        area_union = cv.countNonZero(image_union)

        return abs(1 - area_overlap / area_union)

    def _calc_metrics(
            self, image_ver: np.ndarray, image_ref: np.ndarray,
            algorithm: str, image_mask: Union[None, np.ndarray] = None
    ) -> NoReturn:
        postfix = 'ALL'

        if image_mask is not None:
            image_ver = cv.copyTo(image_ver, image_mask)
            postfix = 'TXT'

        self.results_complete[algorithm][postfix] = dict()
        cdict = self.results_complete[algorithm][postfix]

        tp = self.calc_true_positive(image_ver, image_ref)
        tn = self.calc_true_negative(image_ver, image_ref)
        fp = self.calc_false_positive(image_ver, image_ref)
        fn = self.calc_false_negative(image_ver, image_ref)

        if tp == 0 or fp == 0:
            for metric in EvalMetric.to_list():
                cdict[metric.vs()] = np.nan
            return

        cdict[EvalMetric.TruePositive.vs()] = tp / cv.countNonZero(image_ver)
        cdict[EvalMetric.TrueNegative.vs()] = tn / cv.countNonZero(~image_ver)
        cdict[EvalMetric.FalsePositive.vs()] = fp / cv.countNonZero(image_ver)
        cdict[EvalMetric.FalseNegative.vs()] = fn / cv.countNonZero(~image_ver)

        cdict[EvalMetric.IntersectionOverUnion.vs()] = self.calc_intersection_over_union(image_ver, image_ref)

        cdict[EvalMetric.Accuracy.vs()] = (tp + tn) / (tp + fp + tn + fn)
        cdict[EvalMetric.Sensitivity.vs()] = tp / (tp + fn)
        cdict[EvalMetric.Precision.vs()] = tp / (tp + fp)
        cdict[EvalMetric.Specificity.vs()] = tn / (tn + fp)

    def _aggregate_results(self) -> NoReturn:
        self.results_aggregated = dict(zip([
            EvalMetric.IntersectionOverUnion.vs(),
            EvalMetric.Accuracy.vs(),
            EvalMetric.Sensitivity.vs(),
            EvalMetric.Precision.vs(),
            EvalMetric.Specificity.vs()
        ], [('any', 10000000)] * 5))

        for metric in self.results_aggregated.keys():
            for algorithm in self.results_complete.keys():
                score = self.results_complete[algorithm]['ALL'][metric]
                if score < self.results_aggregated[metric][1]:
                    self.results_aggregated[metric] = (algorithm, score)

