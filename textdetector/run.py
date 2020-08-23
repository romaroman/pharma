import sys
import glob

from typing import NoReturn, List, Union

import cv2 as cv

import textdetector.config as config
from textdetector.detector import Detector
from textdetector.file_info import get_file_info
from textdetector.evaluator import Evaluator
from textdetector.writer import Writer

import utils


class Run:

    def __init__(self) -> NoReturn:
        self.image_paths: List[str] = glob.glob(str(config.root_folder / "cropped/*.png"))

        if config.evaluate:
            self.evaluator: Evaluator = Evaluator()

    def process(self) -> NoReturn:
        writer = Writer()

        for image_path in self.image_paths[:10]:
            file_info = get_file_info(image_path, config.database)

            image_orig = cv.imread(image_path)

            detection = Detector(image_orig)
            detection.detect_text_regions()

            writer.add_dict_result(file_info.to_dict())

            if config.evaluate:
                evaluation_result = self.evaluator.evaluate(detection, file_info)
                writer.add_dict_result(evaluation_result)

            if config.profile:
                writer.add_dict_result(utils.profiler.get_results())

            if config.write:
                writer.save_single_detection(detection, file_info.filename)

            writer.update_dataframe()
            utils.logger.debug(f"Successfully processed {file_info.filename}")

        writer.save_dataframe()


def main() -> int:
    runner = Run()
    runner.process()
    return sys.exit(0)


if __name__ == '__main__':
    main()
