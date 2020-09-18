import logging
import traceback

from typing import NoReturn, Dict, Any
from multiprocessing import Pool, Value

import cv2 as cv

import config
import writer

from loader import Loader
from aligner import Aligner
from detector import Detector
from fileinfo import FileInfo
from evaluator import Evaluator
from referencer import Referencer
from annotation import Annotation

import utils


logger = logging.getLogger('runner')
counter: Value = Value('i', 0)


class Runner:

    @classmethod
    def process(cls) -> NoReturn:
        files = Loader().get_files()

        logger.info(f"\nPreparing to process {len(files)} images"
                    f"{f' via {config.mlt_cpus} threads' if not config.is_debug() else ''}...\n")

        if not config.is_multithreading_used():
            for file in files:
                writer.update_session_with_pd([cls._process_single_file(file)])
        else:
            for files_chunk in utils.chunks(files, config.mlt_cpus * 10):
                with Pool(processes=config.mlt_cpus) as pool:
                    results = pool.map(cls._process_single_file, files_chunk)
                    pool.close()
                    writer.update_session_with_pd(results)

    @classmethod
    def _process_single_file(cls, file: FileInfo) -> Dict[str, Any]:
        result = {'ses': {'status': 'success'}, 'fi': file.to_dict()}

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

                result['evm'] = evaluator.get_mask_results()
                result['evr'] = evaluator.get_regions_results()

        except Exception as exception:
            result['ses'].update({
                'status': 'fail',
                'exc': str(exception),
                'trcbk': traceback.format_exc()
            })

            if config.is_debug():
                traceback.print_exc()
        finally:
            with counter.get_lock():
                counter.value += 1

            result['ses'].update({
                'idx': counter.value,
            })

            logger.info(f"#{str(counter.value).zfill(6)} | "
                        f"{result['ses']['status'].upper()} | {file.filename}")
            return result
