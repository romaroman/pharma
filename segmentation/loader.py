import random
import logging

from pathlib import Path
from typing import NoReturn, List, Union, Dict, Tuple, Any

from common import config
from segmentation.fileinfo import FileInfoEnrollment, FileInfoRecognition, FileInfo

import utils


logger = logging.getLogger('segmentation | loader')


class Loader:
    __DEBUG_FILES = [
        # 'PharmaPack_R_I_S1_Ph1_P0056_D01_S001_C1_P1',
    ]

    def __init__(self, dir_source: Path) -> NoReturn:
        self.dir_source: Path = dir_source
        self.image_files: Union[List[FileInfoEnrollment], List[FileInfoRecognition]] = list()
        self.image_chunks: Dict[Tuple, Union[List[FileInfoEnrollment], List[FileInfoRecognition]]] = dict()

        if config.loading.use_debug_files:
            self._handle_debug_files()
        else:
            self._handle_regular_loading()


    def _load(self) -> NoReturn:
        self.image_files = [FileInfo.get_file_info_by_path(file) for file in self.dir_source.glob('*.png')]
        logger.info(f"Loaded {len(self.image_files)} files from {self.dir_source}")

    def _is_filtration_needed(self) -> bool:
        return len(config.loading.filter_by.items()) > 0

    def _filter(self) -> NoReturn:
        image_files_filter = self.image_files.copy()
        amount_before = len(image_files_filter)
        self.image_files.clear()

        while image_files_filter:
            file = image_files_filter.pop()

            valid = False
            for keyword, values in config.loading.filter_by.items():
                if file.get_attribute_by_key(keyword) in values:
                    valid = config.loading.filter_mode == 'inclusive'
                    if not valid:
                        break
                else:
                    valid = False
                    break

            if valid:
                self.image_files.append(file)

        logger.info(f"Filtered {amount_before - len(self.image_files)} files")

    def _sort(self) -> NoReturn:
        for key in config.loading.sort_by:
            self.image_files.sort(key=lambda f: f.get_attribute_by_key(key))

    def _group(self) -> NoReturn:
        for file in self.image_files:
            unique = tuple(file.get_attribute_by_keys(config.loading.group_by))

            if unique in self.image_chunks.keys():
                self.image_chunks[unique].append(file)
            else:
                self.image_chunks[unique] = [file]

    def get_files(self) -> Union[List, Dict]:
        if not config.loading.group_by:
            return self.image_files
        else:
            return self.image_chunks

    def get_chunks(self) -> Tuple[Any, int]:
        if config.loading.group_by:
            return self.image_chunks.values(), len(self.image_chunks.values())
        else:
            chunks_length = 100
            return utils.chunks(self.image_files, chunks_length), chunks_length

    def _handle_debug_files(self) -> NoReturn:
        if not self.__DEBUG_FILES:
            logger.warning("Debug files list is empty...")
        else:
            for file in self.__DEBUG_FILES:
                self.image_files.append(FileInfo.get_file_info_by_path(self.dir_source / f"{file}.png"))

    def _handle_regular_loading(self) -> NoReturn:
        self._load()

        if self._is_filtration_needed():
            self._filter()

        if config.loading.fraction < 1:
            self.image_files = self.image_files[:int(len(self.image_files) * abs(config.loading.fraction))]
            logger.info(f"Shrank files list by {config.loading.fraction * 100:.0f}%")

        if config.loading.sort_by and config.loading.shuffle:
            logger.info("Shuffle is dominant over SortBy flags, shuffling files list...")
            if config.loading.use_seed:
                random.seed(config.general.seed)
            random.shuffle(self.image_files)
        else:
            if config.loading.sort_by:
                self._sort()

        if config.loading.group_by:
            self._group()
