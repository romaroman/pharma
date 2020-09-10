import logging
from pathlib import Path
from typing import List, Any, Dict

from textdetector.args import parser
from textdetector.enums import DetectionAlgorithm, FileDatabase, ResultMethod

import utils


logger = logging.getLogger('config')
args = parser.parse_args()

timestamp: str = utils.get_str_timestamp()

database: FileDatabase = FileDatabase[args.database]
src_folder: Path = utils.init_path(args.src_folder) / str(database)
dst_folder: Path = utils.init_path(args.dst_folder) / timestamp

shuffle: bool = args.shuffle
seed: bool = args.seed
split_on_chunks: bool = args.split_on_chunks
percentage: int = args.percentage

logging_level: int = getattr(logging, args.logging_level)
debug: bool = args.debug
profile: bool = args.profile

write: bool = args.write
visualize: bool = args.visualize
clear_output: bool = args.clear_output

multithreading: bool = args.multithreading
scale_factor: float = args.scale_factor
algorithms: List[DetectionAlgorithm] = [DetectionAlgorithm[algorithm] for algorithm in args.algorithms.replace(' ', '').split(',')]
approx_method: ResultMethod = ResultMethod[args.approx_method]

evaluate: bool = args.evaluate
extract_reference: bool = args.extract_reference


def to_dict() -> Dict[str, Any]:
    dict_result = dict()

    dict_result['timestamp'] = timestamp

    dict_result['database'] = str(database)
    dict_result['src_folder'] = str(src_folder)
    dict_result['dst_folder'] = str(dst_folder)

    dict_result['shuffle'] = shuffle
    dict_result['seed'] = seed
    dict_result['split_on_chunks'] = split_on_chunks
    dict_result['percentage'] = percentage

    dict_result['logging_level'] = logging_level
    dict_result['debug'] = debug
    dict_result['profile'] = profile

    dict_result['write'] = write
    dict_result['visualize'] = visualize
    dict_result['clear_output'] = clear_output

    dict_result['multithreading'] = multithreading
    dict_result['scale_factor'] = scale_factor
    dict_result['algorithms'] = ", ".join([str(alg) for alg in algorithms])

    dict_result['evaluate'] = evaluate
    dict_result['extract_reference'] = extract_reference

    return dict_result


def validate():
    global multithreading, debug, algorithms

    if DetectionAlgorithm.MajorVoting in algorithms:
        if len(algorithms) - 1 < 3:
            algorithms.remove(DetectionAlgorithm.MajorVoting)
            logger.warning("Cannot perform Algorithm.MajorVoting algorithm if there's less than 3 algorithms")

    if DetectionAlgorithm.MorphologyIteration2 in algorithms and DetectionAlgorithm.MorphologyIteration1 not in algorithms:
        logger.warning("Algorithm.MorphologyIteration2 cannot be performed without Algorithm.MorphologyIteration1\n"
                       ", adding that to algorithms list")
        algorithms.insert(0, DetectionAlgorithm.MorphologyIteration1)

    if debug and multithreading:
        logger.warning("Multithreading isn't supported during debug")
        multithreading = False
