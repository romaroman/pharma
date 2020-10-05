import time
import logging
from copy import deepcopy
from typing import Dict, NoReturn

from utils.helpers import Singleton


logger = logging.getLogger('profiler')


class _Profiler(metaclass=Singleton):

    def __init__(self):
        self._timestamp = time.time()
        self._results: Dict[str, float] = dict()

    def _update_timestamp(self) -> NoReturn:
        self._timestamp = time.time()

    def add_timestamp(self, message: str) -> NoReturn:
        difference = round(time.time() - self._timestamp, 4)
        logger.debug(f"{message} --- {difference} sec ---")

        entry_text = message.lower().replace(' ', '_')
        self._results[entry_text] = difference

        self._update_timestamp()

    def to_dict(self) -> Dict[str, float]:
        self._update_timestamp()

        dict_copy = deepcopy(self._results)
        self._results.clear()

        for key, value in dict_copy:
            dict_copy[key] = round(value, 4)

        return dict_copy


profiler = _Profiler()
