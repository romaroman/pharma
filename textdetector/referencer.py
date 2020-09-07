import logging
from typing import NoReturn, Dict, List, Union, Tuple

import cv2 as cv
import numpy as np

import textdetector.config as config
from textdetector.annotation import Annotation, BoundingRectangle, BoundingRectangleRotated
from textdetector.detector import Detector
from textdetector.file_info import FileInfo

import utils


logger = logging.getLogger('referencer')

class Referencer:

    def __init__(self, image_orig: np.ndarray, file_info: FileInfo):
        self.image_ver: np.ndarray = image_orig

        self.file_info: FileInfo = file_info
        self.annotation: Annotation = \
            Annotation.load_annotation_by_pattern(config.src_folder, self.file_info.get_annotation_pattern())

        self.image_ref: np.ndarray = self.annotation.load_reference_image(config.src_folder / "references")

        self.dict_results: Dict[str, Tuple[np.ndarray, np.nda]] = dict()

    def extract_reference_regions(self) -> NoReturn:

        if self.annotation.is_empty():
            logger.warning(f'Omitting {self.file_info.filename} because annotation contains 0 regions')
            return

        def extract_reference_region(
                brect: Union[BoundingRectangle, BoundingRectangleRotated]
        ) -> Tuple[np.ndarray, np.ndarray]:

            image_mask = np.zeros(self.image_ref.shape[:2], dtype=np.uint8)
            brect.draw(image_mask, color=(255, 255, 255), filled=True)

            image_mask_warped = cv.warpPerspective(
                image_mask, homo_mat,
                utils.swap_dimensions(self.image_ver.shape)
            )
            image_part_masked = cv.bitwise_and(self.image_ver, utils.to_rgb(image_mask_warped))

            contour = cv.findContours(image_mask_warped, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0][0]
            bx, by, bw, bh = cv.boundingRect(contour)

            image_part = image_part_masked[by:by + bh, bx:bx + bw]
            coords = np.transpose(np.asarray(cv.boxPoints(cv.minAreaRect(contour)), dtype=np.int0)).ravel()

            return image_part, coords

        homo_mat = utils.find_homography_matrix(
            utils.to_gray(self.image_ref),
            utils.to_gray(self.image_ver),
        )

        for index, brect in enumerate(self.annotation.bounding_rectangles, start=1):
            dict_key = f"{str(index).zfill(4)}_{brect.label}"
            try:
                self.dict_results[dict_key] = extract_reference_region(brect)
            except:
                logger.warning(f"SKIPPED {dict_key} from {self.file_info.filename}")

    def get_coordinates(self) -> List[np.ndarray]:
        return [v[1] for _, v in self.dict_results.items()]