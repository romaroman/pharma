import logging
import traceback

from typing import NoReturn, Dict, Any, Union
from multiprocessing import Pool, Value

import cv2 as cv

import config
import utils
import writer

from loader import Loader
from aligner import Aligner
from detector import Detector
from fileinfo import FileInfoEnrollment, FileInfoRecognition
from evaluator import EvaluatorByAnnotation, EvaluatorByVerification
from referencer import Referencer
from annotation import Annotation
from collector import Collector
from qualityestimator import QualityEstimator


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
                    f"{f'via {config.mlt_threads} threads' if not config.is_debug() else ''}...\n")

        if not config.mlt_used:
            for chunk in chunks:
                results = [cls._process_single_file(file) for file in chunk]
                collector.add_results(results)
                collector.dump()

        else:
            for chunk in chunks:
                with Pool(processes=config.mlt_threads) as pool:
                    results = pool.map(cls._process_single_file, chunk)
                    pool.close()
                    collector.add_results(results)
                    collector.dump()

        collector.dump()

    @classmethod
    def _process_single_file(cls, file: Union[FileInfoEnrollment, FileInfoRecognition]) -> Dict[str, Any]:
        result = {'session': {'status': 'success'}, 'file_info': file.to_dict()}

        try:
            annotation = Annotation.load_annotation_by_pattern(file.get_annotation_pattern())
            image_input = cv.imread(str(file.path.resolve()))

            homo_mat = utils.find_homography_matrix(utils.to_gray(image_input), utils.to_gray(annotation.image_ref))
            image_aligned = Aligner.align(image_input, annotation.image_ref, homo_mat)

            if config.qe_blur or config.qe_glares:
                qe = QualityEstimator()
                qe.perform_estimation(image_aligned, annotation.image_ref, homo_mat)
                result['quality_estimation'] = qe.to_dict()

            detection = Detector(image_aligned)
            detection.detect(config.det_algorithms)

            writer.save_detection_results(detection, file)

            if config.exr_used:
                referencer = Referencer(image_aligned, annotation)
                referencer.extract_reference_regions()
                writer.save_reference_results(referencer, file.filename)

            if config.ev_ano_mask or config.ev_ano_regions:
                evaluator = EvaluatorByAnnotation(annotation, homo_mat)
                evaluator.evaluate(detection)

                result['eval_annotation_mask'] = evaluator.get_mask_results()
                result['eval_annotation_regions'] = evaluator.get_regions_results()

            if config.ev_ver_mask or config.ev_ver_regions:
                image_ref, results = Detector.load_results_by_pattern(file.get_verification_pattern())
                evaluator = EvaluatorByVerification(image_ref, results)
                evaluator.evaluate(detection)

                result['eval_verification_mask'] = evaluator.get_mask_results()
                result['eval_verification_regions'] = evaluator.get_regions_results()

        except Exception as exception:
            result['session'].update({
                'status': 'fail',
                'exception': str(exception),
                'traceback': traceback.format_exc()
            })

            if config.is_debug():
                traceback.print_exc()
        finally:
            with counter.get_lock():
                counter.value += 1

            result['session'].update({
                'index': counter.value,
            })

            logger.info(f"#{utils.zfill_n(counter.value, 6)} | {result['session']['status'].upper()} | {file.filename}")
            return result
