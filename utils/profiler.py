import time
from typing import Dict, NoReturn

import utils
import textdetector.config as config


class _Profiler(metaclass=utils.Singleton):

    def __init__(self):
        self._timestamp = time.time()
        self._dict_results: Dict[str, float] = {}

    def _update_timestamp(self) -> NoReturn:
        self._timestamp = time.time()

    def add_timestamp(self, message: str) -> NoReturn:
        if config.profile:
            utils.logger.debug(f"{message} --- {(time.time() - self._timestamp)} sec ---")

        entry_text = message.lower().replace(' ', '_')
        self._dict_results[entry_text] = self._timestamp

        self._update_timestamp()

    def get_results(self) -> Dict[str, float]:
        self._update_timestamp()
        return self._dict_results


profiler = _Profiler()
