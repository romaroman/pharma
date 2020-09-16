import logging
import confuse
from pathlib import Path

from textdetector.enums import *
import utils


logger = logging.getLogger('config')


timestamp = utils.get_str_timestamp()
confuse = confuse.Configuration('_', __name__)
confuse.set_file('config.yaml')

mode: Mode = Mode[confuse['Mode'].as_str()]

database: FileDatabase = FileDatabase[confuse['Database'].as_str()]
dir_source: Path = confuse['Dirs']['SourceDir'].as_path() / str(database)
dir_output: Path = confuse['Dirs']['OutputDir'].as_path() / timestamp

imgl_shuffle: bool = confuse['ImageLoading']['Shuffle'].get()
imgl_seed: bool = confuse['ImageLoading']['Seed'].get()
imgl_split_on_chunks: bool = confuse['ImageLoading']['SplitOnChunks'].get()
imgl_percentage: int = confuse['ImageLoading']['Percentage'].as_number()

out_log_level: int = getattr(logging, confuse['Output']['LogLevel'].as_str())
out_profile: bool = confuse['Output']['Profile'].get()

wr_clean_before: bool = confuse['Write']['CleanBefore'].get()
wr_images: bool = confuse['Write']['Images'].get()
wr_visualization: bool = confuse['Write']['Visualization'].get()

run_multithreading: bool = confuse['Runtime']['Multithreading'].get()

alg_scale_factor: float = confuse['Algorithm']['ScaleFactor'].as_number()
alg_algorithms: List[DetectionAlgorithm] = [DetectionAlgorithm[alg] for alg in confuse['Algorithm']['Algorithms'].get()]
alg_approximation_method: ApproximationMethod = ApproximationMethod[confuse['Algorithm']['ApproximationMethod'].as_str()]
alg_alignment_method: AlignmentMethod = AlignmentMethod[confuse['Algorithm']['AlignmentMethod'].as_str()]

op_detect: bool = confuse['Operations']['Detect'].get()
op_evaluate: bool = confuse['Operations']['Evaluate'].get()
op_extract_references: bool = confuse['Operations']['ExtractReferences'].get()


def need_warp() -> bool:
    return not alg_alignment_method is AlignmentMethod.Reference


def is_debug() -> bool:
    return mode is Mode.Debug


def is_multithreading() -> bool:
    return run_multithreading