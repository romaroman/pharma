import logging
import confuse

from pathlib import Path
from multiprocessing import cpu_count

from enums import *

import utils


logger = logging.getLogger('config')


timestamp = utils.get_str_timestamp()
confuse = confuse.Configuration('_', __name__)
confuse.set_file('config.yaml')

mode: Mode = Mode[confuse['Mode'].as_str()]
database: FileDatabase = FileDatabase[confuse['Database'].as_str()]

dir_source: Path = confuse['Dirs']['SourceDir'].as_path() / str(database)
dir_output: Path = confuse['Dirs']['OutputDir'].as_path() / timestamp

out_clear_output_dir: bool = confuse['Output']['ClearOutputDir'].get()
out_log_level: int = getattr(logging, confuse['Output']['LogLevel'].as_str())
out_profile: bool = confuse['Output']['Profile'].get()

mlt_used: bool = confuse['Multithreading']['Used'].get()
mlt_cpus: bool = cpu_count() - 2 if confuse['Multithreading']['CPUs'].as_str().lower() == 'auto'\
    else confuse['Multithreading']['CPUs'].as_number()

det_scale_factor: float = confuse['Detection']['ScaleFactor'].as_number()
det_algorithms: List[DetectionAlgorithm] = [DetectionAlgorithm[alg] for alg in confuse['Detection']['Algorithms'].get()]

det_approximation_method_default: ApproximationMethod = \
    ApproximationMethod[confuse['Detection']['ApproximationMethodDefault'].as_str()]
det_approximation_methods_used: List[ApproximationMethod] = \
    [ApproximationMethod[ap] for ap in confuse['Detection']['ApproximationMethodsUsed'].get()]

det_alignment_method: AlignmentMethod = AlignmentMethod[confuse['Detection']['AlignmentMethod'].as_str()]
det_write: List['str'] = confuse['Detection']['Write'].get()

ev_mask_used: bool = confuse['Evaluation']['Mask']['Used'].get()
ev_mask_aggregate: bool = confuse['Evaluation']['Mask']['Aggregate'].get()
ev_mask_write: bool = confuse['Evaluation']['Mask']['Write'].get()

ev_regions_used: bool = confuse['Evaluation']['Regions']['Used'].get()
ev_regions_aggregate: bool = confuse['Evaluation']['Regions']['Aggregate'].get()
ev_regions_write: bool = confuse['Evaluation']['Regions']['Write'].get()

ev_metrics: List[EvalMetric] = EvalMetric.to_list() if confuse['Evaluation']['Metrics'].get()[0] == 'a' \
    else [EvalMetric[metric] for metric in confuse['Evaluation']['Metrics'].get()]

exr_used: bool = confuse['ExtractReference']['Used'].get()
exr_write: bool = confuse['ExtractReference']['Write'].get()


def need_warp() -> bool:
    return not det_alignment_method is AlignmentMethod.Reference


def is_debug() -> bool:
    return mode is Mode.Debug


def is_multithreading_used() -> bool:
    return mlt_used