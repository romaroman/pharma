from enum import Enum, auto
from typing import List, Tuple


_rrect = List[Tuple[Tuple[float, float], Tuple[float, float], float]]


class PreprocessMethod(Enum):

    def __str__(self) -> str:
        return str(self.value)

    BasicMorphology = auto(),
    EdgeExtraction = auto(),
    EdgeExtractionAndFiltration = auto()
