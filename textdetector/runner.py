import logging
import traceback

from typing import NoReturn, Dict, Union
from multiprocessing import Pool, Value

import cv2 as cv

import writer
import config

from aligner import Aligner
from detector import Detector
from fileinfo import FileInfo
from loader import Loader
from evaluator import Evaluator
from referencer import Referencer
from annotation import Annotation

import utils


logger = logging.getLogger('runner')


class Runner:

    def __init__(self) -> NoReturn:
        self.counter: Value = Value('i', 0)

    def process(self) -> NoReturn:
        files = Loader().get_files()

        logger.info(f"\nPreparing to process {len(files)} images"
                    f"{f' via {config.mlt_cpus} threads' if not config.is_debug() else ''}...\n")

        if not config.is_multithreading_used():
            for file in files:
                result = self._process_single_file(file)
                writer.update_session_with_pd([result])
        else:
            for files_chunk in utils.chunks(files, config.mlt_cpus * 10):
                with Pool(processes=config.mlt_cpus) as pool:
                    results = pool.map(self._process_single_file, files_chunk)
                    pool.close()
                    writer.update_session_with_pd(results)

    def _process_single_file(self, file: FileInfo) -> Dict[str, Dict[str, Union[int, float]]]:
        result = {'session': {'status': 'success'}}
        try:
            annotation = Annotation.load_annotation_by_pattern(config.dir_source, file.get_annotation_pattern())

            image_input = cv.imread(str(file.path.resolve()))
            image_aligned = Aligner.align(image_input)

            detection = Detector(image_aligned)
            detection.detect(config.det_algorithms)

            if config.det_write:
                writer.save_detection_results(detection, file.filename)

            if config.exr_used:
                referencer = Referencer(image_aligned, annotation)
                referencer.extract_reference_regions()

                if config.exr_write:
                    writer.save_reference_results(referencer, file.filename)

            if config.ev_mask_used or config.ev_regions_used:
                evaluator = Evaluator(annotation)
                evaluator.evaluate(detection)

                result['evaluation_mask'] = evaluator.get_mask_results()
                result['evaluation_regions'] = evaluator.get_regions_results()

                # writer.save_evaluation_results()

        except Exception as exception:
            result['session'].update({
                'status': 'fail',
                'exc': str(exception),
                'trcbk': traceback.format_exc()
            })

            if config.is_debug():
                traceback.print_exc()
        finally:
            with self.counter.get_lock():
                self.counter.value += 1

            result['session'].update({
                'idx': self.counter.value,
            })

            logger.info(f"#{str(self.counter.value).zfill(6)} | "
                        f"{result['session']['status'].upper()} | {file.filename}")
            return result
