import glob
import random
import logging
import traceback

from typing import NoReturn, List, Dict, Union
from multiprocessing import Pool, Value, cpu_count

import cv2 as cv
import numpy as np

from textdetector import config
from textdetector.writer import Writer
from textdetector.aligner import Aligner
from textdetector.detector import Detector
from textdetector.fileinfo import FileInfo
from textdetector.evaluator import Evaluator
from textdetector.referencer import Referencer
from textdetector.annotation import Annotation
from textdetector.enums import AlignmentMethod, Mode

import utils

logger = logging.getLogger('runner')
counter: Value = Value('i', 0)
DEBUG_FILES = []


class Runner:

    def __init__(self) -> NoReturn:
        self.image_paths: List[str] = list()
        self._load_images()

    def process(self) -> NoReturn:
        proc_amount = cpu_count() - 2
        logger.info(f"\nPreparing to process {len(self.image_paths)} images"
                    f"{f' via {proc_amount} threads' if not config.mode is Mode.Debug else ''}...\n")

        if config.mode is Mode.Debug or (config.mode is Mode.Release and not config.run_multithreading):
            for image_path in self.image_paths:
                result = self._process_single_image(image_path)
                Writer.update_session_with_pd([result])
        else:
            for images_chunk in utils.chunks(self.image_paths, proc_amount * 10):
                with Pool(processes=proc_amount) as pool:
                    results = pool.map(self._process_single_image, images_chunk)
                    pool.close()
                    Writer.update_session_with_pd(results)

    @staticmethod
    def _process_single_image(image_path: str) -> Dict[str, Dict[str, Union[int, float]]]:
        writer = Writer()

        session_dict = dict()
        status = 'success'

        file_info = FileInfo.get_file_info(image_path, config.database)
        annotation = Annotation.load_annotation_by_pattern(config.dir_source, file_info.get_annotation_pattern())
        writer.add_dict_result('fi', file_info.to_dict())

        try:
            image_input = cv.imread(image_path)

            if config.alg_alignment_method is AlignmentMethod.Reference:
                image_aligned = Aligner.align_with_reference(image_input, annotation.image_ref)
            elif config.alg_alignment_method is AlignmentMethod.ToCorners:
                image_aligned = Aligner.align_to_corners(image_input)
            elif config.alg_alignment_method is AlignmentMethod.NoAlignment:
                image_aligned = np.copy(image_input)

            detection = Detector(image_aligned)
            detection.detect(config.alg_algorithms)

            if config.op_extract_references:
                referencer = Referencer(image_aligned, file_info, annotation)
                referencer.extract_reference_regions()
                if config.wr_images:
                    writer.save_reference_results(referencer, file_info.filename)

            if config.op_evaluate:
                evaluator = Evaluator()
                evaluator.evaluate(detection, annotation)
                writer.add_dict_result('er', evaluator.to_dict_complete())

            if config.out_profile:
                writer.add_dict_result('prf', utils.profiler.to_dict())

            if config.wr_images:
                writer.save_all_results(detection, file_info.filename)

        except Exception as exception:
            status = 'fail'

            session_dict.update({
                'exc': str(exception),
                'trcbk': traceback.format_exc()
            })

            if config.Mode is Mode.Debug:
                traceback.print_exc()
        finally:
            with counter.get_lock():
                counter.value += 1

            session_dict.update({
                'idx': counter.value,
                'status': status
            })
            writer.add_dict_result('ses', session_dict)

            logger.info(f"#{str(counter.value).zfill(6)} | {status.upper()} | {file_info.filename}")
            return writer.get_current_results()

    def _load_images(self) -> NoReturn:
        if DEBUG_FILES:
            for file in DEBUG_FILES:
                self.image_paths.append(str(config.dir_source / f"cropped/{file}.png"))
            return

        self.image_paths = glob.glob(str(config.dir_source / "cropped/*.png"))
        # self.image_paths = [path for path in self.image_paths if path.find('P0034') != -1]

        if config.imgl_shuffle:
            if config.imgl_seed:
                random.seed(666)

            random.shuffle(self.image_paths)

        if config.imgl_percentage < 100:
            self.image_paths = self.image_paths[:int(len(self.image_paths) * abs(config.imgl_percentage) / 100)]
