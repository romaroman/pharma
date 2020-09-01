import logging
from typing import NoReturn, Dict, List, Union, Tuple

import cv2 as cv
import numpy as np
import pandas as pd

import textdetector.config as config
from textdetector.annotation import Annotation, AnnotationLabel, BoundingRectangleABC, BoundingRectangle, \
    BoundingRectangleRotated
from textdetector.detector import Detector
from textdetector.file_info import FileInfo

import utils


class Referencer:

    def __init__(self, image_orig: np.ndarray, file_info: FileInfo):
        self.image_ver: np.ndarray = image_orig

        self.file_info: FileInfo = file_info
        self.annotation: Annotation = \
            Annotation.load_annotation_by_pattern(config.root_folder, self.file_info.get_annotation_pattern())

        self.image_ref: np.ndarray = self.annotation.load_reference_image(config.root_folder / "references")

    def extract_reference_regions(self):

        if len(self.annotation.bounding_rectangles_rotated) == 0:
            print(f'ommited {self.file_info.filename}')
            return

        def extract_reference_region(
                brect: Union[BoundingRectangle, BoundingRectangleRotated]
        ) -> Tuple[np.ndarray, np.ndarray]:

            image_mask = np.zeros(self.image_ref.shape[:2], dtype=np.uint8)
            brect.draw(image_mask, color=(255, 255, 255), filled=True)

            image_mask_warped = cv.warpPerspective(
                image_mask, homo_mat,
                (self.image_ver.shape[1], self.image_ver.shape[0])
            )
            image_part_masked = cv.bitwise_and(self.image_ver, utils.to_rgb(image_mask_warped))

            contour = cv.findContours(image_mask_warped, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0][0]
            bx, by, bw, bh = cv.boundingRect(contour)

            image_part = image_part_masked[by:by + bh, bx:bx + bw]
            brect_dst = cv.boxPoints(cv.minAreaRect(contour))

            return image_part, brect_dst

        homo_mat = utils.find_homography(
            utils.to_gray(self.image_ref),
            utils.to_gray(self.image_ver),
        )

        image_parts = list()
        contour_parts = list()
        for brect in self.annotation.bounding_rectangles:
            image_part, brect_part = extract_reference_region(brect)
            image_parts.append(image_part)
            contour_parts.append(brect_part)

        for rbrect in self.annotation.bounding_rectangles_rotated:
            image_part, rbrect_part = extract_reference_region(rbrect)
            image_parts.append(image_part)
            contour_parts.append(rbrect_part)

        pass
