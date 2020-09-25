import logging
from typing import Tuple, Union

import cv2 as cv
import numpy as np

import config
import morph
from enums import AlignmentMethod

import utils


logger = logging.getLogger('aligner')


class Aligner:

    @classmethod
    def align_with_reference(
            cls,
            image_to_warp: np.ndarray,
            image_ref: np.ndarray,
            homo_mat: Union[None, np.ndarray] = None
    ) -> np.ndarray:
        if homo_mat is None:
            homo_mat = utils.find_homography_matrix(utils.to_gray(image_to_warp), utils.to_gray(image_ref))
        return cv.warpPerspective(image_to_warp, homo_mat, utils.swap_dimensions(image_ref.shape))

    @classmethod
    def align_to_corners(cls, image_ver: np.ndarray) -> Tuple[bool, Union[np.ndarray, None]]:
        image_gray = cv.cvtColor(image_ver, cv.COLOR_BGR2GRAY)

        thresh_adjusts = list(range(-40, -100, -10))
        found_package_border = False

        while not found_package_border:

            image_bw = utils.thresh(image_gray, thresh_adjust=thresh_adjusts.pop())

            image_closed = cv.morphologyEx(image_bw, cv.MORPH_CLOSE, morph.mscale((15, 15)))
            image_dilated = cv.dilate(image_closed, morph.mscale((10, 10)))

            image_filled = utils.fill_holes(image_dilated)
            image_filled = cv.erode(image_filled, np.ones(morph.mscale((10, 10))))

            contours, _ = cv.findContours(image_filled, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            contour = max(contours, key=lambda x: cv.contourArea(x))

            contour_area = cv.contourArea(contour)
            image_contour = utils.crop_image_by_contour(image_filled, contour, False)

            if contour_area / image_filled.size > 0.2 and cv.countNonZero(image_contour) > contour_area / 2:
                found_package_border = True
            elif len(thresh_adjusts) == 0:
                return found_package_border, image_ver

        image_aligned = cls.crop_and_align_contour(image_ver, contour)
        return found_package_border, image_aligned

    @classmethod
    def crop_and_align_contour(cls, image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        rrect = cv.minAreaRect(contour)
        box = cv.boxPoints(rrect)
        box = np.asarray(box, dtype=np.int0)
    
        xs = [i[0] for i in box]
        ys = [i[1] for i in box]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    
        angle = rrect[2]
        if angle < -45:
            angle += 90
    
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
    
        size = (x2 - x1, y2 - y1)
        m = cv.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)
    
        image_cropped = cv.getRectSubPix(image, size, center)
        image_cropped = cv.warpAffine(image_cropped, m, size)
    
        image_bw = utils.thresh(utils.to_gray(image_cropped), thresh_adjust=-100)
        contours, _ = cv.findContours(image_bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour_max = max(contours, key=lambda x: cv.contourArea(x))
        x, y, w, h = cv.boundingRect(contour_max)
        image_result = image_cropped[y:y+h, x:x+w]
    
        return image_result

    @classmethod
    def align(
            cls,
            image_input: np.ndarray,
            image_ref: Union[None, np.ndarray] = None,
            homo_mat: Union[None, np.ndarray] = None
    ) -> np.ndarray:
        if config.alignment_method is AlignmentMethod.Reference:
            if image_ref is not None:
                image_aligned = cls.align_with_reference(image_input, image_ref, homo_mat)
            else:
                logger.warning('Cannot align image without reference')
                raise ValueError
        elif config.alignment_method is AlignmentMethod.ToCorners:
            image_aligned = cls.align_to_corners(image_input)
        else:
            image_aligned = np.copy(image_input)

        morph.prepare_image(image_aligned, config.det_scale_factor)
        return image_aligned
