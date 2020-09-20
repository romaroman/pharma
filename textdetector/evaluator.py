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

        self.results_mask: Dict[str, np.ndarray] = dict()
        self.results_regions: Dict[str, np.ndarray] = dict()

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

    def _evaluate_by_mask(self, image_ver: np.ndarray) -> np.ndarray:
        if config.need_warp():
            image_ver = cv.warpPerspective(
                image_ver, self.homo_mat, utils.swap_dimensions(self.annotation.image_ref.shape)
            )

        return np.array(self.calc_all_metrics(image_ver, self.image_mask_ref_text, to_dict=False))

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
                    mask_combined = cv.bitwise_or(bmask, pmask)
                    contour = cv.findContours(mask_combined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0][0]
                    pmask_cropped = utils.crop_image_by_contour(pmask, contour, True)
                    bmask_cropped = utils.crop_image_by_contour(bmask, contour, True)

                    matched_region_number += 1
                    scores = self.calc_all_metrics(pmask_cropped, bmask_cropped, to_dict=False)
                    rows.append([bi, matched_region_number] + scores)

            if matched_region_number == 0:
                rows.append([bi, matched_region_number] + self.generate_negative_result())

        return np.array(rows)

    def get_mask_results(self) -> Dict[str, np.ndarray]:
        return self.results_mask

    def get_regions_results(self) -> Dict[str, np.ndarray]:
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

        return area_overlap / area_union

    @classmethod
    def calc_confusion(cls, image_ver: np.ndarray, image_ref: np.ndarray) -> Tuple[int, int, int, int]:
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

        def clip(arg1: Union[int, float], arg2: Union[int, float]) -> float:
            return arg1 / arg2 if arg2 !=0 else 1.0

        tp, tn, fp, fn = cls.calc_confusion(image_ver, image_ref)

        if tp == 0:
            results = cls.generate_negative_result()
        else:
            iou = cls.calc_intersection_over_union(image_ver, image_ref)

            tpr = clip(tp, tp + fn)
            fpr = clip(fp, fp + tn)
            tnr = clip(tn, tn + fp)
            fnr = clip(fn, tp + fn)

            prv = clip(tp + fn, tp + fp + tn + fn)
            acc = clip(tp + tn, tp + fp + tn + fn)
            fdr = clip(fp, tp + fp)
            prc = clip(tp, tp + fp)
            fro = clip(fn, fn + tn)
            npv = clip(tn, tn + fn)

            plr = clip(tpr, fpr)
            nlr = clip(fnr, tnr)
            dor = clip(plr, nlr)
            f1s = 2 * clip(prc * tpr, prc + tpr)

            results = [iou, tpr, fpr, tnr, fnr, prv, acc, fdr, prc, fro, npv, plr, nlr, dor, f1s]

        for i, result in enumerate(results):
            results[i] = np.round(result, 3)

        if to_dict:
            calc_dict = dict()
            for metric in EvalMetric:
                calc_dict[metric.vs()] = results[::-1].pop()
            return calc_dict
        else:
            return results
