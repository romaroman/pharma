import os
import logging

from abc import ABC, abstractmethod
from confuse import Configuration
from typing import List, Dict
from pathlib import Path
from multiprocessing import cpu_count

from common.enums import ApproximationMethod, EvalMetric, SegmentationAlgorithm, AlignmentMethod, Mode, FileDatabase,\
    Descriptor

import utils


class __Config(ABC):

    def __init__(self, confuse: Configuration):
        self.confuse: Configuration = confuse

    # @abstractmethod
    # def validate(self) -> bool:
    #     raise NotImplemented


class GeneralConfig(__Config):

    def __init__(self, confuse: Configuration):
        super().__init__(confuse)

        self.timestamp = utils.get_str_timestamp()

        self.seed: int = confuse['Seed'].as_number()

        self.mode: Mode = Mode[confuse['Mode'].as_str()]
        self.database: FileDatabase = FileDatabase[confuse['Database'].as_str()]

        self.dir_source: Path = confuse['DirSource'].as_path()
        self.dir_output: Path = confuse['DirOutput'].as_path()

        self.clear_dir_output: bool = confuse['ClearDirOutput'].get()

        self.log_level: int = getattr(logging, confuse['LogLevel'].as_str())
        self.profile: bool = confuse['Profile'].get()

        self.multithreading: bool = confuse['Multithreading'].get()
        self.threads: int = confuse['Threads'].get()

        if str(self.threads).lower().startswith('a'):
            self.threads = cpu_count() - 2

    def is_debug(self) -> bool:
        return self.mode is Mode.Debug


class FineGrainedConfig(__Config):

    def __init__(self, confuse: Configuration):
        super().__init__(confuse)

        self.descriptor_default: Descriptor = Descriptor[confuse['DescriptorDefault'].as_str()]
        self.descriptors_used: List[Descriptor] = [Descriptor[dt] for dt in confuse['DescriptorsUsed'].get()]


class LoadingConfig(__Config):

    def __init__(self, confuse: Configuration):
        super().__init__(confuse)

        self.use_debug_files: bool = confuse['UseDebugFiles'].get()

        self.shuffle: bool = confuse['Shuffle'].get()
        self.use_seed: bool = confuse['UseSeed'].get()
        self.fraction: int = confuse['Fraction'].as_number()

        self.filter_by: Dict[str, List[int]] = dict()
        self.filter_mode: str = confuse['Filter']['Mode'].as_str().lower()

        for key in confuse['Filter']:
            if key.lower() in ['packageclass', 'phone', 'distinct', 'sample', 'size', 'angle', 'side']:
                values = confuse['Filter'][key].get()
                if values:
                    self.filter_by.update({key.lower(): values})

        self.sort_by: List[str] = [i.lower() for i in confuse['SortBy'].get()]
        self.group_by: List[str] = [i.lower() for i in confuse['GroupBy'].get()]


class SegmentationConfig(__Config):

    def __init__(self, confuse: Configuration):
        super().__init__(confuse)

        self.scale_factor: float = confuse['ScaleFactor'].as_number()

        self.alignment_method: AlignmentMethod = \
            AlignmentMethod[confuse['AlignmentMethod'].as_str()]

        self.algorithms: List[SegmentationAlgorithm] = \
            [SegmentationAlgorithm[alg] for alg in confuse['Algorithms'].get()]

        self.approximation_method_default: ApproximationMethod = \
            ApproximationMethod[confuse['ApproximationMethodDefault'].as_str()]

        self.approximation_methods_used: List[ApproximationMethod] = \
            [ApproximationMethod[ap] for ap in confuse['ApproximationMethodsUsed'].get()]

        self.write: List[str] = [str(ap).lower() for ap in confuse['Write'].get()]

        self.quality_raise: bool = confuse['QualityEstimation']['Raise'].get()
        self.quality_glares: bool = confuse['QualityEstimation']['UseGlares'].get()
        self.quality_blur: bool = confuse['QualityEstimation']['UseBlur'].get()

        self.eval_annotation_mask: bool = confuse['Evaluation']['Annotation']['Mask'].get()
        self.eval_annotation_regions: bool = confuse['Evaluation']['Annotation']['Regions'].get()

        self.eval_verification_mask: bool = confuse['Evaluation']['Verification']['Mask'].get()
        self.eval_verification_regions: bool = confuse['Evaluation']['Verification']['Regions'].get()

        self.eval_metrics: List[EvalMetric] = list()
        eval_metrics = confuse['Evaluation']['Metrics'].get()
        if not eval_metrics or str(eval_metrics[0]).lower().startswith('a'):
            self.eval_metrics = EvalMetric.to_list()
        else:
            self.eval_metrics = [EvalMetric[metric] for metric in eval_metrics]

        self.extract_reference: bool = confuse['ExtractReference']['Used'].get()
        self.extract_reference_write: bool = confuse['ExtractReference']['Write'].get()

    def is_alignment_needed(self) -> bool:
        return self.alignment_method is not AlignmentMethod.Reference


class HashConfig(__Config):

    def __init__(self, confuse: Configuration):
        super().__init__(confuse)

        self.algorithms: List[SegmentationAlgorithm] = \
            [SegmentationAlgorithm[alg] for alg in confuse['Algorithms'].get()]

        self.base_model: str = confuse['BaseModel'].as_str()
        self.descriptor_length: int = confuse['DescriptorLength'].as_number()

        self.neighbours_amount: int = confuse['NeighboursAmount'].as_number()
        self.top_neighbours_amount: int = confuse['TopNeighboursAmount'].as_number()
        self.top_ranked_size: int = confuse['TopRankedSize'].as_number()

        self.redis_complete: int = confuse['Storage']['Redis']['Complete'].as_number()
        self.redis_insert: int = confuse['Storage']['Redis']['Insert'].as_number()
        self.redis_port: int = confuse['Storage']['Redis']['Port'].as_number()
        self.redis_host: str = confuse['Storage']['Redis']['Host'].as_str()

        self.lmdb_path: Path = confuse['Storage']['LMDB']['Path'].as_path()

        self.nearpy_length: int = confuse['NearPy']['Length'].as_number()

        self.lopq_coarse_clusters: int = confuse['LOPQ']['CoarseClusters'].as_number()
        self.lopq_subvectors: int = confuse['LOPQ']['Subvectors'].as_number()
        self.lopq_subquantizer_clusters: int = confuse['LOPQ']['SubquantizerClusters'].as_number()


confuse_main = Configuration('pharmapack-recognition', 'config')
confuse_main.set_file(f"{os.getenv('PHARMAPACK_PROJECT_DIR')}/config.yaml")

general = GeneralConfig(confuse_main['General'])
finegrained = FineGrainedConfig(confuse_main['FineGrained'])
loading = LoadingConfig(confuse_main['Loading'])
segmentation = SegmentationConfig(confuse_main['Segmentation'])
