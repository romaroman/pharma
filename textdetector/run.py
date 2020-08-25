import sys
import glob
import random
import logging

from typing import NoReturn, List, Union

import cv2 as cv

import textdetector.config as config
from textdetector.detector import Detector
from textdetector.referencer import Referencer
from textdetector.file_info import get_file_info
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

            file_info = get_file_info(image_path, config.database)
            image_orig = cv.imread(image_path)
            # detection = Detector(image_orig)
            #
            # writer.add_dict_result({'filename': file_info.filename})
            #
            # detection.detect_text_regions()

            if config.extract_reference:
                referencer = Referencer(image_orig, file_info)
                referencer.extract_reference_regions()

    #         if config.evaluate:
    #             evaluation_result = self.evaluator.evaluate(detection, file_info)
    #             writer.add_dict_result(evaluation_result)
    #
    #         if config.profile:
    #             writer.add_dict_result(utils.profiler.get_results())
    #
    #         if config.write:
    #             writer.save_single_detection(detection, file_info.filename)
    #
    #         writer.update_dataframe()
    #         if index % 10 == 0:
    #             writer.save_dataframe()
    #
    #         logger.info(f'#{index_b} PROCESSED {image_path}')
    #
        #     except FileNotFoundError as e:
        #         logger.warning(f"#{index_b} FAILED Not found an annotation for {file_info.filename}")
        #     except Exception as e:
        #         logger.error(f'#{index_b} FAILED {image_path}, error: {e}')
        #     finally:
        #         if e:
        #             writer.clear_current_results()
        #             writer.add_failed_file(image_path, e.strerror)
        #
        # writer.save_dataframe()

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
