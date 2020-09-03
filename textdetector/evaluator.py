import logging
from typing import NoReturn, Dict, List, Union

import cv2 as cv
import numpy as np
import pandas as pd

import textdetector.config as config
from textdetector.annotation import Annotation, AnnotationLabel
from textdetector.detector import Detector
from textdetector.file_info import FileInfo

import utils


logger = logging.getLogger('evaluator')


class Evaluator:

    def __init__(self) -> NoReturn:
        self.df_result: pd.Pandas = pd.DataFrame()
        self.dict_result: Dict[str, float] = dict()

    def evaluate(self, detection: Detector, annotation: Annotation) -> NoReturn:
        image_reference = annotation.load_reference_image(config.src_folder / "references")

        image_ref_mask_text = utils.to_gray(annotation.create_mask_by_labels(
            labels=[AnnotationLabel.Text, AnnotationLabel.Number]
        ))

        image_verification = detection.image_orig

        homo_mat = utils.find_homography_matrix(
            utils.to_gray(image_verification),
            utils.to_gray(image_reference)
        )

        for algorithm, result in detection.results.items():
            image_mask, regions = result

            image_mask_warped = cv.warpPerspective(
                image_mask, homo_mat, image_reference.shape[:2][::-1]
            )

            self.dict_result[f'{algorithm}_text_ratio'] = self._calc_iou_ratio(
                image_mask_warped, image_ref_mask_text
            )

            self.dict_result[f'{algorithm}_regions_amount'] = len(regions)

    def to_dict(self) -> Dict[str, float]:
        return self.dict_result

    @classmethod
    def _calc_iou_ratio(cls, image_verification_mask: np.ndarray, image_reference_mask: np.ndarray) -> float:
        image_overlap = cv.bitwise_and(image_verification_mask, image_reference_mask)
        image_union = cv.bitwise_xor(image_verification_mask, image_reference_mask)

        area_overlap = cv.countNonZero(image_overlap)
        area_union = cv.countNonZero(image_union)

        tp_ratio = area_overlap / area_union
        return abs(1 - tp_ratio)
