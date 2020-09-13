import logging
from typing import NoReturn, Dict, List, Union, Tuple

import cv2 as cv
import numpy as np

from . import FileInfo, Annotation, BoundingRectangle, BoundingRectangleRotated
import utils


logger = logging.getLogger('referencer')


class Referencer:

    def __init__(self, image_ver: np.ndarray, file_info: FileInfo, annotation: Annotation):
        self.image_ver: np.ndarray = image_ver
        self.file_info: FileInfo = file_info
        self.annotation: Annotation = annotation

        self.results: Dict[str, Tuple[np.ndarray, np.ndarray]] = dict()

    def extract_reference_regions(self) -> NoReturn:
        if self.annotation.is_empty_text():
            logger.warning(f'Omitting {self.file_info.filename} because annotation contains 0 regions')
            return

        def extract_reference_region(
                brect: Union[BoundingRectangle, BoundingRectangleRotated]
        ) -> Tuple[np.ndarray, np.ndarray]:

            image_mask = np.zeros(image_ref.shape[:2], dtype=np.uint8)
            brect.draw(image_mask, color=(255, 255, 255), filled=True)

            image_mask_warped = cv.warpPerspective(
                image_mask, homo_mat,
                utils.swap_dimensions(self.image_ver.shape)
            )
            image_part_masked = cv.bitwise_and(self.image_ver, utils.to_rgb(image_mask_warped))

            contours = cv.findContours(image_mask_warped, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
            if not contours:
                raise IndexError
            contour = contours[0]

            bx, by, bw, bh = cv.boundingRect(contour)

            image_part = image_part_masked[by:by + bh, bx:bx + bw]
            coords = np.transpose(np.asarray(cv.boxPoints(cv.minAreaRect(contour)), dtype=np.int0)).ravel()
            return image_part, coords

        image_ref = np.copy(self.annotation.image_ref)
        homo_mat = utils.find_homography_matrix(
            utils.to_gray(image_ref),
            utils.to_gray(self.image_ver),
        )

        for index, brect in enumerate(self.annotation.bounding_rectangles, start=1):
            dict_key = f"{str(index).zfill(4)}_{brect.label}"
            try:
                self.results[dict_key] = extract_reference_region(brect)
            except IndexError:
                pass

    def get_coordinates(self) -> List[np.ndarray]:
        return [v[1] for _, v in self.results.items()]

    def to_dict(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return self.results
