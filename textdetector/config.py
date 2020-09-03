import logging
from pathlib import Path
from typing import List, Any, Dict

from textdetector.file_info import Database
from textdetector.detector import Algorithm
from textdetector.args import parser


import utils


args = parser.parse_args()

database: Database = Database.Enrollment

src_folder: Path = utils.init_path(args.src_folder) / str(database)
dst_folder: Path = utils.init_path(args.dst_folder)

logging_level: int = logging.DEBUG

shuffle: bool = False
random_seed: bool = False
percentage: int = 100

debug: bool = True
profile: bool = True

write: bool = True
visualize: bool = True
clear_output: bool = True

align_package: bool = False
algorithms: List[Algorithm] = Algorithm.to_list()
evaluate: bool = True
extract_reference: bool = True


def to_dict() -> Dict[str, Any]:
    dict_result = dict()

    dict_result['src_folder'] = str(src_folder)
    dict_result['dst_folder'] = str(src_folder)
    dict_result['database'] = str(database)

    dict_result['logging_level'] = logging_level

    dict_result['shuffle'] = shuffle
    dict_result['random_seed'] = random_seed
    dict_result['percentage'] = percentage

    dict_result['debug'] = debug
    dict_result['profile'] = profile

    dict_result['write'] = write
    dict_result['visualize'] = visualize
    dict_result['clear_output'] = clear_output

    dict_result['algorithms'] = str(evaluate)
    dict_result['algorithms'] = Algorithm.to_string_list()
    dict_result['evaluate'] = evaluate
    dict_result['extract_reference'] = extract_reference

    return dict_result
