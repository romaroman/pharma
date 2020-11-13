from typing import NoReturn, Union, Dict, Any

import cv2 as cv
import numpy as np

from common import config

import utils


class ImageIsBlurred(BaseException):
    pass


class ImageContainsGlares(BaseException):
    pass


class QualityEstimator:

    def __init__(self) -> NoReturn:
        self.blur_score_ver: float = 0
        self.blur_score_ref: float = 100
        self.glares_score: float = 1

        self.is_blurred: bool = False
        self.is_glared: bool = False

        self.image_glares_visualization: Union[np.ndarray, None] = None
        self.image_blur_visualization: Union[np.ndarray, None] = None

    @classmethod
    def calculate_blur_score(cls, image: np.ndarray) -> float:
        image_blurred = cv.GaussianBlur(image, (3, 3), 0.0)

        image_laplacian = np.abs(cv.Laplacian(image_blurred, cv.CV_32F, 1, 1, 1, cv.BORDER_REPLICATE))
        hist = cv.calcHist([image_laplacian], [0], None, [256], [0, 256])

        count_threshold = 10
        it = 0
        for it, value in enumerate(hist):
            if value < count_threshold:
                break

        if it <= 1:
            return 0

        diff_high = hist[it - 1] - count_threshold
        diff_low = count_threshold - hist[it]

        score = ((it - 1) * diff_low + it * diff_high) / (diff_low + diff_high)

        return score[0]

    def is_image_blurred(self, image_ver: np.ndarray, image_ref: np.ndarray) -> bool:
        self.blur_score_ver = self.calculate_blur_score(image_ver)
        self.blur_score_ref = self.calculate_blur_score(image_ref)

        if config.general.is_debug():
            self.image_blur_visualization = np.hstack([
                cv.putText(
                    img=utils.to_rgb(np.copy(image_ver)), text=str(self.blur_score_ver), org=(100, 100),
                    fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=3, color=(0, 255, 0), thickness=4
                ),
                cv.putText(
                    img=utils.to_rgb(np.copy(image_ref)), text=str(self.blur_score_ref), org=(100, 100),
                    fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=3, color=(0, 255, 0), thickness=4
                )
            ])

        return self.blur_score_ref - self.blur_score_ver > 2

    def calculate_glares_score(self, image: np.ndarray) -> float:
        glares_value_threshold = 220

        erd_kernel_s = min(2, round(image.size * 0.0075))
        erd_kernel = cv.getStructuringElement(cv.MORPH_RECT, (erd_kernel_s, erd_kernel_s))

        dil_kernel_s = min(2, round(image.size * 0.02))
        dil_kernel = cv.getStructuringElement(cv.MORPH_RECT, (dil_kernel_s, dil_kernel_s))

        image_mask_not_glared = (image >= glares_value_threshold).astype(np.uint8) * 255

        image_morphed = cv.erode(image_mask_not_glared, erd_kernel)
        image_morphed = cv.dilate(image_morphed, dil_kernel)

        contours, _ = cv.findContours(image_morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        glare_size_threshold = round((0.020 ** 2) * image.size)

        ellipses = list()
        for contour in contours:
            el = cv.minAreaRect(contour)
            if (np.math.pi * el[1][0] * el[1][1]) < glare_size_threshold:
                continue
            ellipses.append((el[0], (el[1][0] + glare_size_threshold, el[1][1] + glare_size_threshold), el[2]))

        image_ellipses_enlarged = np.full_like(image, fill_value=255)
        for el in ellipses:
            cv.ellipse(image_ellipses_enlarged, el, 0, cv.FILLED, cv.LINE_8)

        if config.general.is_debug():
            contours, _ = cv.findContours(image_ellipses_enlarged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            self.image_glares_visualization = cv.drawContours(utils.to_rgb(np.copy(image)), contours, -1, (0, 0, 255), 4)

        glares_area = image.size - cv.countNonZero(image_ellipses_enlarged)
        glares_ratio = max(0, min(1, glares_area / image.size))

        return glares_ratio

    def does_image_contain_glares(self, image_ver: np.ndarray) -> bool:
        ratio_glares_amount_threshold = 0.4
        self.glares_score = self.calculate_glares_score(image_ver)
        return self.glares_score >= ratio_glares_amount_threshold

    def estimate_not_aligned(self, image_ver: np.ndarray, image_ref: np.ndarray, homo_mat: np.ndarray) -> bool:
        image_ver = cv.warpPerspective(image_ver, homo_mat, utils.swap_dimensions(image_ref.shape))
        return self.estimate_aligned(image_ver, image_ref)

    def estimate_aligned(self, image_ver: np.ndarray, image_ref: np.ndarray) -> bool:
        image_ver = cv.cvtColor(image_ver, cv.COLOR_BGR2GRAY)
        image_ref = cv.cvtColor(image_ref, cv.COLOR_BGR2GRAY)

        if config.segmentation.quality_glares:
            self.is_glared = self.does_image_contain_glares(image_ver)
            if config.segmentation.quality_raise and self.is_glared:
                raise ImageContainsGlares

        if config.segmentation.quality_blur:
            self.is_blurred = self.is_image_blurred(image_ver, image_ref)
            if config.segmentation.quality_raise and self.is_blurred:
                raise ImageIsBlurred

        return not self.is_glared and not self.is_blurred

    def perform_estimation(self, image_ver: np.ndarray, image_ref: np.ndarray, homo_mat: np.ndarray) -> NoReturn:
        if config.segmentation.is_alignment_needed():
            self.estimate_not_aligned(image_ver, image_ref, homo_mat)
        else:
            self.estimate_aligned(image_ver, image_ref)

    def to_dict(self) -> Dict[str, Any]:
        dict_blur = {
            'blur_score_ver': self.blur_score_ver,
            'blur_score_ref': self.blur_score_ref,
            'is_blurred': self.is_blurred,
        }
        dict_glares = {
            'glares_score': self.glares_score,
            'is_glared': self.is_glared,
        }

        dict_result = dict()
        if config.segmentation.quality_blur:
            dict_result.update(dict_blur)
        if config.segmentation.quality_glares:
            dict_result.update(dict_glares)

        return dict_result
