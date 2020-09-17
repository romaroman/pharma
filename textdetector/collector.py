from multiprocessing import Manager
from typing import NoReturn, Dict

from textdetector import FileInfo


class Collector:

    class ResultContainer:

        def __init__(self) -> NoReturn:
            self.eval: dict = dict()

    def __init__(self):
        pass
