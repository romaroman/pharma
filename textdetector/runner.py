import logging
import traceback

from typing import NoReturn, Dict, Any, Union
from multiprocessing import Pool, Value

import cv2 as cv

import config
import writer

from loader import Loader
from aligner import Aligner
from detector import Detector
from fileinfo import FileInfoEnrollment, FileInfoRecognition
from evaluator import EvaluatorByAnnotation, EvaluatorByVerification
from referencer import Referencer
from annotation import Annotation
from collector import Collector


logger = logging.getLogger('runner')
counter: Value = Value('i', 0)


class Runner:

    @classmethod
    def process(cls) -> NoReturn:
        loader = Loader()
        chunks, chunks_amount = loader.get_chunks()

        collector = Collector()

        logger.info(f"Preparing to process {len(loader.image_files)} images "
                    f"split on {chunks_amount} chunks "
                    f"{f'via {config.mlt_cpus} threads' if not config.is_debug() else ''}...\n")

        if not config.is_multithreading_used():
            for chunk in chunks:
                results = [cls._process_single_file(file) for file in chunk]
                collector.add_results(results)
        else:
            for chunk in chunks:
                with Pool(processes=config.mlt_cpus) as pool:
                    results = pool.map(cls._process_single_file, chunk)
                    pool.close()
                    collector.add_results(results)
        collector.dump()

    @classmethod
    def _process_single_file(cls, file: Union[FileInfoEnrollment, FileInfoRecognition]) -> Dict[str, Any]:
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

            if config.ev_ano_mask or config.ev_ano_regions:
                evaluator = EvaluatorByAnnotation(annotation)
                evaluator.evaluate(detection)

                result['evanomsk'] = evaluator.get_mask_results()
                result['evanoreg'] = evaluator.get_regions_results()

            if config.ev_ver_mask or config.ev_ver_regions:
                image_ref, results = Detector.load_results_by_pattern(
                    config.dir_source / 'verasref', file.get_verification_pattern()
                )
                evaluator = EvaluatorByVerification(image_ref, results)
                evaluator.evaluate(detection)

                result['evvermsk'] = evaluator.get_mask_results()
                result['evverreg'] = evaluator.get_regions_results()

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
