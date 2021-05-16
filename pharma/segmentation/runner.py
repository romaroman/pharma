import logging
import traceback
from multiprocessing import Pool, Value
from typing import NoReturn, Dict, Any, Union

import cv2 as cv

import pyutils as pu
from pharma.common import config
from pharma.common.enums import ApproximationMethod, SegmentationAlgorithm, AlignmentMethod

from pharma.finegrained import Detector, Serializer
from pharma.segmentation import writer
from pharma.segmentation.aligner import Aligner
from pharma.segmentation.annotation import Annotation
from pharma.segmentation.collector import Collector
from pharma.segmentation.evaluator import EvaluatorByAnnotation, EvaluatorByVerification
from pharma.segmentation.fileinfo import FileInfoEnrollment, FileInfoRecognition
from pharma.segmentation.loader import Loader
from pharma.segmentation.qualityestimator import QualityEstimator
from pharma.segmentation.referencer import Referencer
from pharma.segmentation.segmenter import Segmenter

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
            annotation = Annotation.load_by_pattern(file.get_annotation_pattern())
            image_input = cv.imread(str(file.path.resolve()))

            homo_mat = pu.find_homography_matrix(pu.to_gray(image_input), pu.to_gray(annotation.image_ref))
            image_aligned = Aligner.align(image_input, annotation.image_ref, homo_mat)

            if config.segmentation.quality_blur or config.segmentation.quality_glares:
                qe = QualityEstimator()
                qe.perform_estimation(image_aligned, annotation.image_ref, homo_mat)
                result['quality_estimation'] = qe.to_dict()

            segmenter = Segmenter(image_aligned, image_input, homo_mat)
            segmenter.segment(config.segmentation.algorithms)

            writer.save_segmentation_results(segmenter, file)
            # return
            for algorithm, res in segmenter.results_unaligned.items():
                detections = Detector.detect_descriptors(
                    image_input, config.finegrained.descriptors_used, res.get_default_mask()
                )
                Serializer.save_detections_to_file(detections, f"{algorithm.blob()}:{file.filename}", False)

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
                image_verref, results_verref = Segmenter.load_results(
                    config.general.dir_source / "VerificationReferences", file.get_verification_pattern()
                )
                evaluator = EvaluatorByVerification(image_verref, results_verref)
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

            logger.info(
                f" | {pu.zfill_n(counter.value, 6)} | {result['session']['status'].upper()} | {file.filename}"
            )
            return result
