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
                self.results_regions[algorithm] = self._evaluate_by_regions(regions_ver, detection.image_not_scaled)

    def _evaluate_by_mask(self, image_ver: np.ndarray) -> np.ndarray:
        if config.need_warp():
            image_ver = cv.warpPerspective(
                image_ver, self.homo_mat, utils.swap_dimensions(self.annotation.image_ref.shape)
            )

        sc = ScoreCalculator()
        sc.calc_scores(image_ver, self.image_mask_ref_text)
        return np.array(sc.scores_list)

    def _evaluate_by_regions(self, regions: List[DetectionResult.Region], image_draw: np.ndarray) -> np.ndarray:
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

        image_vis = np.copy(image_draw)
        bi = 1

        for bmask, brect in zip(brect_masks, self.annotation.get_brects_by_labels(AnnotationLabel.get_list_of_text_labels())):
            matched_polygons_amount = 0
            pi = 0

            for pmask, polygon in zip(region_masks, regions_polygons):
                if cv.countNonZero(cv.bitwise_and(bmask, pmask)) > 0:
                    mask_combined = cv.bitwise_or(bmask, pmask)
                    contour = cv.findContours(mask_combined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0][0]

                    pmask_cropped = utils.crop_image_by_contour(pmask, contour, True)
                    bmask_cropped = utils.crop_image_by_contour(bmask, contour, True)

                    sc = ScoreCalculator()
                    sc.calc_scores(pmask_cropped, bmask_cropped)
                    scores = sc.scores_list

                    matched_polygons_amount += 1
                    rows.append([bi, matched_polygons_amount] + scores)
                    scores_es = sc.get_essential_scores()

                    image_vis = regions[pi].draw(image_vis, (0, 255, 0), filled=False)
                    image_vis = cv.putText(
                        img=image_vis, text=str(scores_es.items()[len(scores_es.items())/2:]),
                        org=utils.get_contour_center(regions[pi].contour), fontFace=cv.FONT_HERSHEY_PLAIN,
                        fontScale=1, color=(255, 0, 0), thickness=1
                    )
                    image_vis = cv.putText(
                        img=image_vis, text=str(scores_es.items()[:len(scores_es.items())/2]),
                        org=utils.get_contour_center(regions[pi].contour), fontFace=cv.FONT_HERSHEY_PLAIN,
                        fontScale=1, color=(255, 0, 0), thickness=1
                    )

                pi += 1
            if matched_polygons_amount == 0:
                rows.append([bi, matched_polygons_amount] + ScoreCalculator.generate_negative_results())
            bi += 1
        utils.display(image_vis)
        return np.array(rows)

    def get_mask_results(self) -> Dict[str, np.ndarray]:
        return self.results_mask

    def get_regions_results(self) -> Dict[str, np.ndarray]:
        return self.results_regions


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
        for _ in config.ev_metrics:
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

        self.round_scores()
        self.create_dict()

    def round_scores(self) -> NoReturn:
        for i, result in enumerate(self.scores_list):
            self.scores_list[i] = np.round(result, 3)

    def create_dict(self) -> NoReturn:
        for i, metric in enumerate(config.ev_metrics):
            self.scores_dict[metric.vs()] = self.scores_list[i]

    def get_essential_scores(self) -> Dict[str, float]:
        names = [m.vs() for m in EvalMetric.get_most_valuable()]
        return dict(filter(lambda k: k[0] in names, self.scores_dict.items()))