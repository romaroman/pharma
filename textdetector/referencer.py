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

        self.results: Dict[str, np.ndarray] = dict()

    def extract_reference_regions(self) -> NoReturn:
        if self.annotation.is_empty_text():
            logger.warning(f'Omitting {self.file_info.filename} because annotation contains 0 regions')
            return

        def extract_reference_region(brect: Union[BoundingRectangle, BoundingRectangleRotated]) -> np.ndarray:
            polygon_warped = utils.perspective_transform_contour(brect.to_polygon(), homo_mat)
            return utils.crop_image_by_contour(self.image_ver, polygon_warped, roughly_by_brect=False)

        image_ref = np.copy(self.annotation.image_ref)
        homo_mat = utils.find_homography_matrix(
            utils.to_gray(image_ref),
            utils.to_gray(self.image_ver),
        )

        for index, brect in enumerate(self.annotation.bounding_rectangles, start=1):
            try:
                self.results[f"{str(index).zfill(4)}_{brect.label}"] = extract_reference_region(brect)
            except IndexError:
                pass

    def get_coordinates(self) -> List[np.ndarray]:
        return [v[1] for _, v in self.results.items()]

    def to_dict(self) -> Dict[str, np.ndarray]:
        return self.results
