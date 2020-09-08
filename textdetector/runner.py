import sys
import glob
import random
import logging

from multiprocessing import Pool, cpu_count, Value
from typing import NoReturn, List

import cv2 as cv

import textdetector.config as config
from textdetector.annotation import Annotation
from textdetector.detector import Detector
from textdetector.referencer import Referencer
from textdetector.file_info import FileInfo
from textdetector.evaluator import Evaluator
from textdetector.writer import Writer

import utils


logger = logging.getLogger('runner')
counter: Value = Value('i', 0)


class Runner:

    def __init__(self) -> NoReturn:
        self.image_paths: List[str] = list()
        self._load_images()

    def process(self) -> NoReturn:
        cpus = cpu_count()
        logger.info(f'Preparing to process {len(self.image_paths)} images via {cpus} threads...\n\n\n')

        for images_chunk in utils.chunks(self.image_paths, cpus):
            with Pool(processes=cpus if config.multiprocessing else 1) as pool:
                data = pool.map(self._process_thread, images_chunk)
                pool.close()
                Writer.update_dataframe(data)

    @staticmethod
    def _process_thread(image_path: str) -> NoReturn:
        writer = Writer()

        file_info = FileInfo.get_file_info(image_path, config.database)
        writer.add_dict_result('file_info', file_info.to_dict())

        image_input = cv.imread(image_path)

        detection = Detector(image_input)
        detection.detect(config.algorithms)

        writer.add_dict_result('detection_result', detection.to_dict())

        if config.extract_reference:
            referencer = Referencer(image_input, file_info)
            referencer.extract_reference_regions()
            writer.save_reference_results(referencer, file_info.filename)

        if config.evaluate:
            annotation = Annotation.load_annotation_by_pattern(config.src_folder, file_info.get_annotation_pattern())
            writer.add_dict_result('annotation_info', annotation.to_dict())

            evaluator = Evaluator()
            evaluator.evaluate(detection, annotation)
            writer.add_dict_result('evaluation_result', evaluator.to_dict())

        if config.profile:
            writer.add_dict_result('profiling_result', utils.profiler.to_dict())

        if config.write:
            writer.save_all_results(detection, file_info.filename)

        with counter.get_lock():
            counter.value += 1

        writer.add_dict_result('index', {'#': counter.value})
        logger.info(f'#{str(counter.value).zfill(6)} PROCESSED {image_path}')

        return writer.get_current_results()

    def _load_images(self):
        self.image_paths = glob.glob(str(config.src_folder / "cropped/*.png"))

        if config.shuffle:
            if config.random_seed:
                random.seed(123)
            random.shuffle(self.image_paths)

        if config.percentage < 100:
            self.image_paths = self.image_paths[:int(len(self.image_paths) * abs(config.percentage)/100)]


def setup() -> NoReturn:
    utils.setup_logger('text_detector', config.logging_level)
    utils.suppress_warnings()

    config.validate()
    Writer.prepare_output_folder()
    logger.info(f"Currently used configuration:\n{utils.pretty(config.to_dict())}")


def main() -> int:
    setup()
    runner = Runner()
    runner.process()
    return sys.exit(0)


if __name__ == '__main__':
    main()
