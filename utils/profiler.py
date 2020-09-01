import time
import logging
from copy import deepcopy
from typing import Dict, NoReturn

import utils
import textdetector.config as config


logger = logging.getLogger('profiler')


class _Profiler(metaclass=utils.Singleton):

    def __init__(self):
        self._timestamp = time.time()
        self._dict_results: Dict[str, float] = dict()

    def _update_timestamp(self) -> NoReturn:
        self._timestamp = time.time()

    def add_timestamp(self, message: str) -> NoReturn:
        difference = round(time.time() - self._timestamp, 4)
        if config.profile:
            logger.debug(f"{message} --- {difference} sec ---")

        entry_text = message.lower().replace(' ', '_')
        self._dict_results[entry_text] = difference

        self._update_timestamp()

    def to_dict(self) -> Dict[str, float]:
        self._update_timestamp()

        dict_copy = deepcopy(self._dict_results)
        self._dict_results.clear()

        for key, value in dict_copy:
            dict_copy[key] = round(value, 4)

        return dict_copy


profiler = _Profiler()
