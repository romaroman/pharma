import re
import glob
from pathlib import Path

from typing import NoReturn, Dict, List, Union

import cv2 as cv
import numpy as np
import pandas as pd

import textdetector.config as config
from textdetector.annotation import Annotation, AnnotationLabel
from textdetector.detector import Detector
from textdetector.file_info import FileInfo

import utils


logger = utils.get_logger(__name__, config.logging_level)


class Evaluator:

    def __init__(self) -> NoReturn:
        self.df_result: pd.Pandas = pd.DataFrame()

    def evaluate(self, detection: Detector, file_info: FileInfo) -> Dict[str, float]:
        try:
            annotation = Annotation.load_annotation_by_pattern(config.root_folder, file_info.get_annotation_pattern())
        except FileNotFoundError:
            logger.warning(f"Not found an annotation for {file_info.filename}")
            return {}

        image_reference = annotation.load_reference_image(config.root_folder / "references")

        image_ref_mask_text = annotation.get_mask_by_labels([
            AnnotationLabel.Text, AnnotationLabel.Number
        ])
        image_ref_mask_graph = annotation.get_mask_by_labels([
            AnnotationLabel.Watermark, AnnotationLabel.Image, AnnotationLabel.Barcode
        ])

        image_verification = detection.image_orig

        homo_mat = utils.find_homography(
            utils.to_gray(image_verification),
            utils.to_gray(image_reference)
        )

        h, w = image_reference.shape[0], image_reference.shape[1]

        image_ver_mask_warped_text = cv.warpPerspective(detection.image_text_masks, homo_mat, (w, h))
        iou_text_text_ratio = self._calc_iou_ratio(image_ver_mask_warped_text, image_ref_mask_text)
        iou_text_graph_ration = self._calc_iou_ratio(image_ver_mask_warped_text, image_ref_mask_graph)

        image_ver_mask_warped_word = cv.warpPerspective(detection.image_word_masks, homo_mat, (w, h))
        iou_word_text_ratio = self._calc_iou_ratio(image_ver_mask_warped_word, image_ref_mask_text)
        iou_word_graph_ratio = self._calc_iou_ratio(image_ver_mask_warped_word, image_ref_mask_graph)

        evaluation_result = {
            'amount_ver_regions_text': len(detection.text_regions),
            'iou_text_text_ratio': iou_text_text_ratio,
            'iou_text_graph_ration': iou_text_graph_ration,

            'amount_ver_regions_word': len(detection.word_regions),
            'iou_word_text_ratio': iou_word_text_ratio,
            'iou_word_graph_ratio': iou_word_graph_ratio,
        }

        return evaluation_result

    @classmethod
    def _calc_iou_ratio(cls, image_verification_mask: np.ndarray, image_reference_mask: np.ndarray) -> float:
        image_overlap = cv.bitwise_and(image_verification_mask, image_reference_mask)
        image_union = cv.bitwise_xor(image_verification_mask, image_reference_mask)

        area_overlap = cv.countNonZero(image_overlap)
        area_union = cv.countNonZero(image_union)

        tp_ratio = area_overlap / area_union
        return tp_ratio