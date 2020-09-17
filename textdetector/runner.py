import logging
import traceback

from typing import NoReturn, List, Dict, Union, Any
from multiprocessing import Pool, Value

import cv2 as cv
import numpy as np

from textdetector import config
from textdetector.writer import Writer
from textdetector.aligner import Aligner
from textdetector.collector import Collector
from textdetector.detector import Detector
from textdetector.fileinfo import FileInfo
from textdetector.loader import Loader
from textdetector.evaluator import Evaluator
from textdetector.referencer import Referencer
from textdetector.annotation import Annotation
from textdetector.enums import AlignmentMethod, Mode

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
                Writer.update_session_with_pd([result])
        else:
            for files_chunk in utils.chunks(files, config.mlt_cpus * 10):
                with Pool(processes=config.mlt_cpus) as pool:
                    results = pool.map(self._process_single_file, files_chunk)
                    pool.close()
                    Writer.update_session_with_pd(results)

    def _process_single_file(self, file: FileInfo) -> Dict[str, Dict[str, Union[int, float]]]:
        writer = Writer()

        session_dict = {'status': 'success'}

        result = {}

        annotation = Annotation.load_annotation_by_pattern(config.dir_source, file.get_annotation_pattern())

        try:
            image_input = cv.imread(file.path)

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
                    writer.save_reference_results(referencer, file.filename)

            if config.op_evaluate:
                evaluator = Evaluator(annotation)
                evaluator.evaluate(detection)

                result['evaluation_mask'] = evaluator.get_mask_results()
                result['evaluation_regions'] = evaluator.get_regions_results()

            if config.wr_images:
                writer.save_all_results(detection, file.filename)


        except Exception as exception:
            session_dict.update({
                'status': 'fail',
                'exc': str(exception),
                'trcbk': traceback.format_exc()
            })

            if config.is_debug():
                traceback.print_exc()
        finally:
            with self.counter.get_lock():
                self.counter.value += 1

            session_dict.update({
                'idx': self.counter.value,
            })
            writer.add_dict_result('ses', session_dict)

            logger.info(f"#{str(self.counter.value).zfill(6)} | {session_dict['status'].upper()} | {file.filename}")

            result['session'] = session_dict

            return writer.get_current_results()
