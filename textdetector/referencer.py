import logging
from typing import NoReturn, Dict, Union

import numpy as np

from annotation import Annotation, BoundingRectangle, BoundingRectangleRotated

import utils


logger = logging.getLogger('referencer')


class Referencer:

    def __init__(self, image_ver: np.ndarray, annotation: Annotation):
        self.image_ver: np.ndarray = image_ver
        self.annotation: Annotation = annotation

        self.results: Dict[str, np.ndarray] = dict()

    def extract_reference_regions(self) -> NoReturn:
        if self.annotation.is_empty_text():
            return None

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

    def to_dict(self) -> Dict[str, np.ndarray]:
        return self.results
