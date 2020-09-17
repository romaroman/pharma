import random

from typing import NoReturn

import config


class Loader:

    DEBUG_FILES = []

    def __init__(self) -> NoReturn:
        self._parse_config()
        if self.DEBUG_FILES:
            for file in self.DEBUG_FILES:
                self.image_paths.append(config.dir_source / f"cropped/{file}.png")
            return

        self.image_paths = (config.dir_source / "cropped").glob('/*.png')
        self.image_paths = [path for path in self.image_paths if path.find('P0003') != -1]

        if config.imgl_shuffle:
            if config.imgl_seed:
                random.seed(666)

            random.shuffle(self.image_paths)

        if config.imgl_percentage < 100:
            self.image_paths = self.image_paths[:int(len(self.image_paths) * abs(config.imgl_percentage) / 100)]

    def _parse_config(self) -> NoReturn:
        s = config.image_loading