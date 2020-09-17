import glob
import random
import logging
import traceback

from typing import NoReturn, List, Dict, Union, Any
from multiprocessing import Pool, Value, Manager

import cv2 as cv
import numpy as np

from textdetector import config
from textdetector.writer import Writer
from textdetector.aligner import Aligner
from textdetector.detector import Detector
from textdetector.fileinfo import FileInfo
from textdetector.loader import Loader
from textdetector.evaluator import Evaluator
from textdetector.referencer import Referencer
from textdetector.annotation import Annotation
from textdetector.enums import AlignmentMethod, Mode

import utils

logger = logging.getLogger('runner')
counter: Value = Value('i', 0)
DEBUG_FILES = [
    # 'PFP_Ph1_P0001_D01_S001_C4_az220_side1'
]


class Runner:

    def __init__(self) -> NoReturn:
        self.results: Dict[str, Any] = Manager().dict()

    def process(self) -> NoReturn:

        loader = Loader()

        logger.info(f"\nPreparing to process {len(self.image_paths)} images"
                    f"{f' via {config.mlt_cpus} threads' if not config.mode is Mode.Debug else ''}...\n")

        if not config.is_multithreading_used():
            for image_path in self.image_paths:
                result = self._process_single_image(image_path)
                Writer.update_session_with_pd([result])
        else:
            for images_chunk in utils.chunks(self.image_paths, config.mlt_cpus * 10):
                with Pool(processes=config.mlt_cpus) as pool:
                    results = pool.map(self._process_single_image, images_chunk)
                    pool.close()
                    Writer.update_session_with_pd(results)

    def _process_single_image(self, image_path: str) -> Dict[str, Dict[str, Union[int, float]]]:
        writer = Writer()

        session_dict = {'status': 'success'}

        file_info = FileInfo.get_file_info(image_path, config.database)
        filename = file_info.filename

        annotation = Annotation.load_annotation_by_pattern(config.dir_source, file_info.get_annotation_pattern())

        try:
            image_input = cv.imread(image_path)

            if config.alg_alignment_method is AlignmentMethod.Reference:
                image_aligned = Aligner.align_with_reference(image_input, annotation.image_ref)
            elif config.alg_alignment_method is AlignmentMethod.ToCorners:
                image_aligned = Aligner.align_to_corners(image_input)
            else:
                image_aligned = np.copy(image_input)

            detection = Detector(image_aligned)
            detection.detect(config.alg_algorithms)

            if config.op_extract_references:
                referencer = Referencer(image_aligned, file_info, annotation)
                referencer.extract_reference_regions()

                if config.wr_images:
                    writer.save_reference_results(referencer, file_info.filename)

            if config.op_evaluate:
                evaluator = Evaluator(annotation)
                evaluator.evaluate(detection)

                self.results[filename]['evaluation_mask'] = evaluator.get_mask_results()
                self.results[filename]['evaluation_regions'] = evaluator.get_regions_results()

            if config.wr_images:
                writer.save_all_results(detection, file_info.filename)

            print(self.results)

        except Exception as exception:
            session_dict.update({
                'status': 'fail',
                'exc': str(exception),
                'trcbk': traceback.format_exc()
            })

            if config.is_debug():
                traceback.print_exc()
        finally:
            with counter.get_lock():
                counter.value += 1

            session_dict.update({
                'idx': counter.value,
            })
            writer.add_dict_result('ses', session_dict)

            logger.info(f"#{str(counter.value).zfill(6)} | {session_dict['status'].upper()} | {file_info.filename}")

            self.results[file_info.filename]['session'] = session_dict

            return writer.get_current_results()


