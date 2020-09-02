import logging
from pathlib import Path
from typing import List, Any, Dict

from textdetector.file_info import Database
from textdetector.detector import Algorithm


base_folder: Path = Path("D:/pharmapack")
database: Database = Database.Enrollment
root_folder: Path = base_folder / str(database)

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

    dict_result['base_folder'] = str(base_folder)
    dict_result['database'] = str(database)
    dict_result['root_folder'] = str(root_folder)

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
