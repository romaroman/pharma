import logging
from pathlib import Path

from textdetector.file_info import Database


base_folder: Path = Path("D:/pharmapack")
database: Database = Database.Enrollment
root_folder: Path = base_folder / str(database)

logging_level: int = logging.INFO

write: bool = True
shuffle: bool = True
percentage: int = 10
debug: bool = True
visualize: bool = True
profile: bool = True
evaluate: bool = True
clear_output: bool = True

extract_reference: bool = True
