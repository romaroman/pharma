import sys
import glob
import random
import logging

from typing import NoReturn, List, Union

import cv2 as cv

import textdetector.config as config
from textdetector.annotation import Annotation
from textdetector.detector import Detector, Algorithm
from textdetector.referencer import Referencer
from textdetector.file_info import FileInfo
from textdetector.evaluator import Evaluator
from textdetector.writer import Writer

import utils


logger = logging.getLogger('runner')


class Run:

    def __init__(self) -> NoReturn:
        self.image_paths: List[str] = list()

        self._load_images()

        if config.evaluate:
            self.evaluator: Evaluator = Evaluator()

    def process(self) -> NoReturn:
        logger.warning(f'Preparing to process {len(self.image_paths)} images...\n\n')

        writer = Writer()

        for index, image_path in enumerate(self.image_paths, start=1):
            index_b = str(index).zfill(6)

            file_info = FileInfo.get_file_info(image_path, config.database)
            writer.add_dict_result('file_info', file_info.to_dict())

            image_orig = cv.imread(image_path)

            detection = Detector(image_orig)
            detection.detect([alg for alg in Algorithm])
            writer.add_dict_result('detection', detection.to_dict())

            if config.extract_reference:
                referencer = Referencer(image_orig, file_info)
                referencer.extract_reference_regions()

            if config.evaluate:
                annotation = Annotation.load_annotation_by_pattern(config.root_folder, file_info.get_annotation_pattern())
                writer.add_dict_result('annotation_info', annotation.to_dict())

                evaluation_result = self.evaluator.evaluate(detection, annotation)
                writer.add_dict_result('evaluation_result', evaluation_result)

            if config.profile:
                writer.add_dict_result('profiling', utils.profiler.get_results())

            if config.write:
                writer.save_results(detection, file_info.filename)

            writer.update_dataframe()

            logger.info(f'#{index_b} PROCESSED {image_path}')

        writer.save_dataframe()

    def _load_images(self):
        self.image_paths = glob.glob(str(config.root_folder / "cropped/*.png"))

        if config.shuffle:
            random.shuffle(self.image_paths)

        if config.percentage is not None:
            if config.percentage < 100:
                self.image_paths = self.image_paths[:int(len(self.image_paths) * abs(config.percentage)/100)]


def main() -> int:
    utils.setup_logger('textdetector', config.logging_level)

    runner = Run()
    runner.process()

    return sys.exit(0)


if __name__ == '__main__':
    main()
