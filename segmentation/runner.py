import logging
import traceback

from typing import NoReturn, Dict, Any, Union
from multiprocessing import Pool, Value

import cv2 as cv

from common import config

from segmentation import writer
from segmentation.loader import Loader
from segmentation.aligner import Aligner
from segmentation.segmenter import Segmenter
from segmentation.fileinfo import FileInfoEnrollment, FileInfoRecognition
from segmentation.evaluator import EvaluatorByAnnotation, EvaluatorByVerification
from segmentation.referencer import Referencer
from segmentation.annotation import Annotation
from segmentation.collector import Collector
from segmentation.qualityestimator import QualityEstimator

import utils


logger = logging.getLogger('runner')
counter: Value = Value('i', 0)


class Runner:

    @classmethod
    def process(cls) -> NoReturn:
        loader = Loader((config.general.dir_source / str(config.general.database) / "cropped"))
        chunks, chunks_amount = loader.get_chunks()

        collector = Collector()

        logger.info(f"Preparing to segment {len(loader.image_files)} images "
                    f"split on chunks with {chunks_amount} length "
                    f"{f'via {config.general.threads} threads' if not config.general.is_debug() else ''}...\n")

        for chunk in chunks:
            if config.general.multithreading:
                pool = Pool(processes=config.general.threads)
                results = pool.map(cls._process_single_file, chunk)
                pool.close()
            else:
                results = [cls._process_single_file(file) for file in chunk]

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

            if config.segmentation.quality_blur or config.segmentation.quality_glares:
                qe = QualityEstimator()
                qe.perform_estimation(image_aligned, annotation.image_ref, homo_mat)
                result['quality_estimation'] = qe.to_dict()

            segmenter = Segmenter(image_aligned)
            segmenter.segment(config.segmentation.algorithms)
            writer.save_segmentation_results(segmenter, file)
            return

            if config.segmentation.extract_reference:
                referencer = Referencer(image_aligned, annotation)
                referencer.extract_reference_regions()
                writer.save_reference_results(referencer, file.filename)

            if config.segmentation.eval_annotation_mask or config.segmentation.eval_annotation_regions:
                evaluator = EvaluatorByAnnotation(annotation, homo_mat)
                evaluator.evaluate(segmenter)

                result['eval_annotation_mask'] = evaluator.get_mask_results()
                result['eval_annotation_regions'] = evaluator.get_regions_results()

            if config.segmentation.eval_verification_mask or config.segmentation.eval_verification_regions:
                image_ref, results = Segmenter.load_results_by_pattern(file.get_verification_pattern())
                evaluator = EvaluatorByVerification(image_ref, results)
                evaluator.evaluate(segmenter)

                result['eval_verification_mask'] = evaluator.get_mask_results()
                result['eval_verification_regions'] = evaluator.get_regions_results()

        except Exception as exception:
            result['session'].update({
                'status': 'fail',
                'exception': str(exception),
                'traceback': traceback.format_exc()
            })

            if config.general.is_debug():
                traceback.print_exc()
        finally:
            with counter.get_lock():
                counter.value += 1

            result['session'].update({
                'index': counter.value,
            })

            logger.info(f"#{utils.zfill_n(counter.value, 6)} | {result['session']['status'].upper()} | {file.filename}")
            return result
