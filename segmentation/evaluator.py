import logging
from abc import ABC, abstractmethod
from typing import NoReturn, Dict, Union, List, Tuple

import cv2 as cv
import numpy as np

from common import config
from common.enums import EvalMetric, AnnotationLabel, ApproximationMethod

from segmentation.segmenter import Segmenter, SegmentationResult, SegmentationAlgorithm
from segmentation.annotation import Annotation

import utils


logger = logging.getLogger('segmentation | evaluator')


class Evaluator(ABC):

    def __init__(self, homo_mat: Union[np.ndarray, None] = None) -> NoReturn:
        self.homo_mat: Union[None, np.ndarray] = homo_mat

        self.results_mask: Dict[SegmentationAlgorithm, np.ndarray] = dict()
        self.results_regions: Dict[SegmentationAlgorithm, np.ndarray] = dict()

    @abstractmethod
    def evaluate(self, segmenter: Segmenter) -> NoReturn:
        raise NotImplemented

    def get_mask_results(self) -> Dict[SegmentationAlgorithm, np.ndarray]:
        return self.results_mask

    def get_regions_results(self) -> Dict[SegmentationAlgorithm, np.ndarray]:
        return self.results_regions


class EvaluatorByAnnotation(Evaluator):

    def __init__(self, annotation: Annotation, homo_mat: Union[np.ndarray, None] = None) -> NoReturn:
        super().__init__(homo_mat)

        self.annotation: Annotation = annotation

        self.image_mask_ref_text: np.ndarray = utils.to_gray(self.annotation.create_mask_by_labels(
            labels=AnnotationLabel.get_list_of_text_labels(), color=(255, 255, 255)
        ))

    def evaluate(self, segmenter: Segmenter) -> NoReturn:
        if config.segmentation.is_alignment_needed() and self.homo_mat is None:
            self.homo_mat = utils.find_homography_matrix(
                utils.to_gray(segmenter.image_not_scaled), utils.to_gray(self.annotation.image_ref)
            )

        for algorithm, result in segmenter.results.items():
            if config.segmentation.eval_annotation_mask:
                image_ver_mask = result.get_default_mask()
                self.results_mask[algorithm] = self._evaluate_by_mask(image_ver_mask)

            if config.segmentation.eval_annotation_regions:
                regions_ver = result.get_default_regions()
                self.results_regions[algorithm] = self._evaluate_by_regions(regions_ver)

    def _evaluate_by_mask(self, image_ver: np.ndarray) -> np.ndarray:
        if config.segmentation.is_alignment_needed():
            image_ver = cv.warpPerspective(
                image_ver, self.homo_mat, utils.swap_dimensions(self.annotation.image_ref.shape)
            )

        sc = ScoreCalculator()
        sc.calc_scores(image_ver, self.image_mask_ref_text)
        return np.array(sc.scores_list)

    def _evaluate_by_regions(self, regions: List[SegmentationResult.Region]) -> np.ndarray:
        img_empty = np.zeros(self.annotation.image_ref.shape[:2], dtype=np.uint8)

        if config.segmentation.is_alignment_needed():
            regions_ver = [
                cv.perspectiveTransform(
                    region.polygon.astype(np.float32),
                    self.homo_mat
                ).astype(np.int32) for region in regions
            ]
        else:
            regions_ver = [region.polygon.astype(np.int32) for region in regions]

        regions_ref = self.annotation.get_bounding_rectangles_by_labels(AnnotationLabel.get_list_of_text_labels())

        masks_ref = [region.draw(np.copy(img_empty), (255, 255, 255), filled=True) for region in regions_ref]
        masks_ver = [cv.drawContours(np.copy(img_empty), [region], -1, 255, -1) for region in regions_ver]

        rows = []
        index_ref = 1

        for mask_ref, region_ref in zip(masks_ref, regions_ref):
            matched_polygons_amount = 0
            index_ver = 0

            for mask_ver, region_ver in zip(masks_ver, regions_ver):
                if cv.countNonZero(cv.bitwise_and(mask_ref, mask_ver)) > 0:
                    mask_combined = cv.bitwise_or(mask_ref, mask_ver)
                    contour = cv.findContours(mask_combined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0][0]

                    pmask_cropped = utils.crop_image_by_contour(mask_ver, contour, True)
                    bmask_cropped = utils.crop_image_by_contour(mask_ref, contour, True)

                    sc = ScoreCalculator()
                    sc.calc_scores(pmask_cropped, bmask_cropped)
                    scores = sc.scores_list

                    matched_polygons_amount += 1
                    rows.append([index_ref, matched_polygons_amount] + scores)

                index_ver += 1
            if matched_polygons_amount == 0:
                rows.append([index_ref, matched_polygons_amount] + ScoreCalculator.generate_negative_results())
            index_ref += 1

        return np.array(rows)


class EvaluatorByVerification(Evaluator):

    def __init__(
            self,
            image_reference: np.ndarray,
            results: Dict[str, SegmentationResult],
            homo_mat: Union[np.ndarray, None] = None
    ) -> NoReturn:
        super().__init__(homo_mat)

        self.image_reference: np.ndarray = image_reference
        self.results: Dict[str, SegmentationResult] = results

    def evaluate(self, segmenter: Segmenter) -> NoReturn:
        if self.homo_mat is None:
            self.homo_mat = utils.find_homography_matrix(
                utils.to_gray(segmenter.image_not_scaled), utils.to_gray(self.image_reference)
            )

        for alg_ver, result_ver in segmenter.results.items():
            result_ref = self.results[alg_ver]

            if config.segmentation.eval_verification_mask:
                image_ver = result_ver.get_default_mask()
                image_ref = result_ref.get_mask(ApproximationMethod.Contour)
                self.results_mask[alg_ver] = self._evaluate_by_mask(image_ver, image_ref)

            if config.segmentation.eval_verification_regions:
                regions_ver = result_ver.get_default_regions()
                regions_ref = result_ref.get_regions(ApproximationMethod.Contour)
                self.results_regions[alg_ver] = self._evaluate_by_regions(regions_ver, regions_ref)

    def _evaluate_by_mask(self, image_ver: np.ndarray, image_ref: np.ndarray):
        image_ver = cv.warpPerspective(
            image_ver, self.homo_mat, utils.swap_dimensions(self.image_reference.shape)
        )

        sc = ScoreCalculator()
        sc.calc_scores(image_ver, image_ref)
        return np.array(sc.scores_list)

    def _evaluate_by_regions(
            self,
            regions_ver: List[SegmentationResult.Region],
            regions_ref: List[SegmentationResult.Region]
    ) -> NoReturn:

        img_empty = np.zeros(self.image_reference.shape[:2], dtype=np.uint8)

        if config.segmentation.is_alignment_needed():
            polygons_ver = [
                cv.perspectiveTransform(
                    region.polygon.astype(np.float32),
                    self.homo_mat
                ).astype(np.int32) for region in regions_ver
            ]
        else:
            polygons_ver = [region.polygon.astype(np.int32) for region in regions_ver]

        masks_ref = [region.draw(np.copy(img_empty), (255, 255, 255), filled=True) for region in regions_ref]
        masks_ver = [cv.drawContours(np.copy(img_empty), [polygon], -1, 255, -1) for polygon in polygons_ver]

        rows = []
        index_ref = 1
        for mask_ref, region_ref in zip(masks_ref, regions_ref):
            matched_regions_amount = 0

            for mask_ver, region_ver in zip(masks_ver, polygons_ver):
                if cv.countNonZero(cv.bitwise_and(mask_ref, mask_ver)) > 0:
                    mask_combined = cv.bitwise_or(mask_ref, mask_ver)
                    contour = cv.findContours(mask_combined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0][0]

                    mask_ver_cropped = utils.crop_image_by_contour(mask_ver, contour, True)
                    mask_ref_cropped = utils.crop_image_by_contour(mask_ref, contour, True)

                    sc = ScoreCalculator()
                    sc.calc_scores(mask_ver_cropped, mask_ref_cropped)

                    matched_regions_amount += 1
                    rows.append([index_ref, matched_regions_amount] + sc.scores_list)

            if matched_regions_amount == 0:
                rows.append([index_ref, matched_regions_amount] + ScoreCalculator.generate_negative_results())
            index_ref += 1

        return np.array(rows)


class ScoreCalculator:

    def __init__(self):
        self.scores_list: List[float] = list()
        self.scores_dict: Dict[str, float] = dict()

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
    def generate_negative_results(cls) -> List[float]:
        results = list()
        for _ in config.segmentation.eval_metrics:
            results.append(np.round(np.random.uniform(-1.25, -0.75), 3))
        return results

    def calc_scores(self, image_ver: np.ndarray, image_ref: np.ndarray) -> NoReturn:

        def clip(arg1: Union[int, float], arg2: Union[int, float]) -> float:
            return arg1 / arg2 if arg2 != 0 else 1.0

        tp, tn, fp, fn = self.calc_confusion(image_ver, image_ref)

        if tp == 0:
            self.scores_list = self.generate_negative_results()
        else:
            iou = self.calc_intersection_over_union(image_ver, image_ref)

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
            f1s = 2 * clip(prc * tpr, prc + tpr)

            self.scores_list = [iou, tpr, fpr, tnr, fnr, prv, acc, fdr, prc, fro, npv, plr, nlr, f1s]

        # round scores
        for i, result in enumerate(self.scores_list):
            self.scores_list[i] = np.round(result, 3)
        
        # create result dict
        for i, metric in enumerate(config.segmentation.eval_metrics):
            self.scores_dict[metric.blob()] = self.scores_list[i]

    def get_essential_scores(self) -> Dict[str, float]:
        names = [m.blob() for m in EvalMetric.get_essential()]
        return dict(filter(lambda i: i[0] in names, self.scores_dict.items()))
