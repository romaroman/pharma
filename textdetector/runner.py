import glob
import random
import logging

from multiprocessing import Pool, cpu_count, Value
from typing import NoReturn, List, Dict, Union

import cv2 as cv

from textdetector import config
from textdetector.annotation import Annotation
from textdetector.detector import Detector
from textdetector.referencer import Referencer
from textdetector.fileinfo import FileInfo
from textdetector.evaluator import Evaluator
from textdetector.writer import Writer

import utils

logger = logging.getLogger('runner')
counter: Value = Value('i', 0)

DEBUG_FILES = {
    # 'PFP_Ph1_P0510_D01_S001_C2_az040_side1',
    # 'PFP_Ph2_P0506_D01_S001_C2_az180_side1',
    # 'PFP_Ph2_P0517_D01_S001_C1_az340_side1',
    # 'PFP_Ph2_P0510_D01_S001_C2_az020_side1',
    # 'PFP_Ph3_P0507_D01_S001_C2_az080_side1',
    # 'PFP_Ph2_P0733_D01_S001_C2_az200_side1',
    # 'PFP_Ph1_P0547_D01_S001_C3_az320_side1',
    'PFP_Ph3_P0385_D01_S001_C3_az160_side1',
    # 'PFP_Ph2_P0110_D01_S001_C2_az060_side1',
    # 'PFP_Ph3_P0516_D01_S001_C1_az300_side1',
    # 'PFP_Ph1_P0523_D01_S001_C4_az340_side1',
    # 'PFP_Ph2_P0514_D01_S001_C2_az160_side1',
    # 'PFP_Ph3_P0237_D01_S001_C3_az180_side1',
    # 'PFP_Ph3_P0512_D01_S001_C2_az040_side1',
    # 'PFP_Ph1_P0511_D01_S001_C2_az320_side1',
    # 'PFP_Ph3_P0509_D01_S001_C2_az100_side1'
}


class Runner:

    def __init__(self) -> NoReturn:
        self.image_paths: List[str] = list()
        self._load_images()

    def process(self) -> NoReturn:
        cpus = cpu_count() - 2
        logger.info(f"Preparing to process {len(self.image_paths)} images"
                    f"{f' via {cpus} threads' if not config.debug else ''}...\n\n\n")

        if config.debug:
            for image_path in self.image_paths:
                result = self._process_single_image(image_path)
                Writer.update_session([result])
        else:
            for images_chunk in utils.chunks(self.image_paths, cpus * 10):
                with Pool(processes=cpus if config.multithreading else 1) as pool:
                    results = pool.map(self._process_single_image, images_chunk)
                    pool.close()
                    Writer.update_session(results)

    @staticmethod
    def _process_single_image(image_path: str) -> Dict[str, Dict[str, Union[int, float]]]:
        writer = Writer()

        status = 'success'
        exception = None

        try:
            file_info = FileInfo.get_file_info(image_path, config.database)
            writer.add_dict_result('file_info', file_info.to_dict())

            image_input = utils.prepare_image(cv.imread(image_path), config.scale_factor)

            detection = Detector(image_input)
            detection.detect(config.algorithms)

            writer.add_dict_result('detection_result', detection.to_dict())

            if config.extract_reference:
                referencer = Referencer(image_input, file_info)
                referencer.extract_reference_regions()
                writer.save_reference_results(referencer, file_info.filename)

            if config.evaluate:
                annotation = Annotation.load_annotation_by_pattern(config.src_folder,
                                                                   file_info.get_annotation_pattern())
                writer.add_dict_result('annotation_info', annotation.to_dict())

                evaluator = Evaluator()
                evaluator.evaluate(detection, annotation)
                writer.add_dict_result('evaluation_result', evaluator.to_dict())

            if config.profile:
                writer.add_dict_result('profiling_result', utils.profiler.to_dict())

            if config.write:
                writer.save_all_results(detection, file_info.filename)

        except Exception as e:
            status = 'fail'
            exception = e
            logger.warning(e)
        finally:
            with counter.get_lock():
                counter.value += 1

            writer.add_dict_result('session', {
                'index': counter.value,
                'status': status,
                'exception': exception
            })

            logger.info(f"#{str(counter.value).zfill(6)} {status.upper()} {image_path}")
            return writer.get_current_results()

    def _load_images(self):
        if DEBUG_FILES:
            for file in DEBUG_FILES:
                self.image_paths.append(str(config.src_folder / f"cropped/{file}.png"))
            return

        self.image_paths = glob.glob(str(config.src_folder / "cropped/*.png"))

        if config.shuffle:
            if config.random_seed:
                random.seed(123)
            random.shuffle(self.image_paths)

        if config.percentage < 100:
            self.image_paths = self.image_paths[:int(len(self.image_paths) * abs(config.percentage) / 100)]
