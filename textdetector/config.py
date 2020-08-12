import logging
from pathlib import Path


base_folder: Path = Path("D:/pharmapack")
database: str = "Enrollment"
root_folder: Path = base_folder / database

logging_level: int = logging.WARNING

write: bool = True
debug: bool = True
visualize: bool = True
profile: bool = True
evaluate: bool = True
clear_output: bool = True
