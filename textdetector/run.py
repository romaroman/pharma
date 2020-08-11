import os
import glob
import random
import sys
import re

from typing import NoReturn, List, Union

import numpy as np
import pandas as pd
import cv2 as cv

from textdetector.detector import Detector
from textdetector.args import parser, Options
from textdetector.file_info import get_file_info, FileInfo
from textdetector.annotation import Annotation

import utils


logger = utils.get_logger(__name__)


class Evaluator:

    def __init__(self, options: Options) -> NoReturn:
        self.df_result: pd.Pandas = pd.DataFrame({
            'filename': pd.Series([], dtype='str'),
            'phone': pd.Series([], dtype='int'),
            'package_class': pd.Series([], dtype='int'),
            'distinct': pd.Series([], dtype='int'),
            'sample': pd.Series([], dtype='int'),
            'size': pd.Series([], dtype='int'),

            'amount_ref': pd.Series([], dtype='int'),
            'amount_ver': pd.Series([], dtype='int'),

            'tp_text': pd.Series([], dtype='float'),
            'fp_text': pd.Series([], dtype='float'),
            'tp_number': pd.Series([], dtype='float'),
            'fp_number': pd.Series([], dtype='float'),
            'tp_watermark': pd.Series([], dtype='float'),
            'fp_watermark': pd.Series([], dtype='float'),
            'tp_image': pd.Series([], dtype='float'),
            'fp_image': pd.Series([], dtype='float'),
            'tp_barcode': pd.Series([], dtype='float'),
            'fp_barcode': pd.Series([], dtype='float'),
        })

        self.root_folder: str = options.base_folder + options.database

        self.annotations: List[Annotation] = []
        self._load_annotations()

    def evaluate(self, detection: Detector, file_info: FileInfo) -> NoReturn:
        annotation = self._find_annotation_by_pattern(re.compile(file_info.get_annotation_pattern()))

        if not annotation:
            logger.warning(f"Not found an annotation for {file_info.filename}")

        image_reference = annotation.load_reference_image(self.root_folder + '/references')
        image_verification = detection.image_orig

        homo_mat = utils.find_homography(
            utils.to_gray(image_verification),
            utils.to_gray(image_reference)
        )

        h, w = image_reference.shape[0], image_reference.shape[1]
        image_ver_mask_warped = cv.warpPerspective(detection.image_text_masks, homo_mat, (w, h))
        image_ver_warped = cv.warpPerspective(detection.image_orig, homo_mat, (w, h))

        image_ref_mask = annotation.create_mask()

        image_ref_mask_text_num = annotation.get_mask_by_labels(['text', 'number'])
        image_ref_mask = np.asarray(image_ref_mask != 0, dtype=np.uint8) * 255

        image_tp = cv.bitwise_and(image_ver_mask_warped, image_ref_mask)
        image_fp = cv.bitwise_xor(image_ver_mask_warped, image_tp)

        tp_area = cv.countNonZero(image_ver_mask_warped)

        tp_ratio = cv.countNonZero(image_tp) / tp_area
        fp_ratio = cv.countNonZero(image_fp) / tp_area


    def _find_annotation_by_pattern(self,pattern: re.Pattern) -> Union[Annotation, None]:
        for annotation in self.annotations:
            if pattern.match(annotation.filename):
                return annotation
        else:
            return None

    def _load_annotations(self) -> NoReturn:

        src_folder = self.root_folder + "/annotations"

        for annotation_file in glob.glob(src_folder + "/*.xml"):
            self.annotations.append(Annotation(annotation_file))

        logger.info(f"Loaded {str(len(self.annotations)).zfill(4)} annotations")


class Run:

    def __init__(self, options: Options) -> NoReturn:
        self.options = options
        self.logger = utils.get_logger(__name__, options.log_level)
        self.image_paths = self._load_images()

        if self.options.evaluate:
            self.evaluator: Evaluator = Evaluator(options)

    def process(self) -> NoReturn:
        random.shuffle(self.image_paths)

        for image_path in self.image_paths:
            file_info = get_file_info(image_path, self.options.database)

            image_orig = cv.imread(image_path)
            flags = Detector.Flags(self.options.visualize, self.options.time_profiling)

            detection = Detector(image_orig, flags)
            detection.detect_text_regions()

            if self.options.write:
                self.write_result(detection, file_info.filename)

            if self.options.evaluate:
                self.evaluator.evaluate(detection, file_info)

    def _load_images(self) -> List[str]:
        return glob.glob(self.options.base_folder + self.options.database + "/cropped/*.png")

    def write_result(self, detect: Detector, filename: str) -> NoReturn:
        common_folder = self.options.base_folder + self.options.database

        def write_entity(entity: Union[List[np.ndarray], np.ndarray],folder_suffix: str,extension: str) -> NoReturn:
            dst_folder = f"{common_folder}/{folder_suffix}"
            os.makedirs(dst_folder, exist_ok=True)
            dst_path = f"{dst_folder}/{filename}.{extension}"

            if extension == 'png':
                cv.imwrite(dst_path, entity)
            elif extension == 'csv':
                np.savetxt(dst_path, entity, delimiter=",", fmt='%i')

        write_entity(detect.get_text_coordinates(), 'python/text_coords', 'csv')
        write_entity(detect.get_words_coordinates(), 'python/word_coords', 'csv')
        write_entity(detect.image_text_regions, 'python/word_regions', 'png')
        write_entity(detect.image_word_regions, 'python/text_regions', 'png')
        write_entity(detect.image_text_masks, 'python/text_masks', 'png')
        write_entity(detect.image_visualization, 'python/visualizations', 'png')


def main() -> int:
    options = Options(parser)
    runner = Run(options)
    runner.process()
    return sys.exit(0)


if __name__ == '__main__':
    main()
