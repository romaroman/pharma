import logging
from pathlib import Path


base_folder: Path = Path("D:/pharmapack")
database: str = "Enrollment"
root_folder: Path = base_folder / database

logging_level: int = logging.INFO

write: bool = True
debug: bool = True
visualize: bool = False
profile: bool = False
evaluate: bool = True
clear_output: bool = True
