import random
import logging

from typing import NoReturn, List, Union, Dict, Tuple, Any

import config
import utils

from fileinfo import FileInfoEnrollment, FileInfoRecognition, FileInfo

logger = logging.getLogger('loader')


class Loader:
    DEBUG_FILES = [
        'PharmaPack_R_I_S1_Ph1_P0056_D01_S001_C1_P1',
    ]

    def __init__(self) -> NoReturn:
        self.image_files: Union[List[FileInfoEnrollment], List[FileInfoRecognition]] = list()
        self.image_chunks: Dict[Tuple, Union[List[FileInfoEnrollment], List[FileInfoRecognition]]] = dict()
        self._parse_config()

        if self.use_debug_files:
            self._handle_debug_files()
        else:
            self._handle_regular_loading()

    def _parse_config(self) -> NoReturn:
        self.use_debug_files: bool = config.confuse['ImageLoading']['UseDebugFiles'].get()
        self.shuffle: bool = config.confuse['ImageLoading']['Shuffle'].get()
        self.seed: bool = config.confuse['ImageLoading']['Seed'].get()
        self.fraction: int = config.confuse['ImageLoading']['Fraction'].as_number()

        self.filter_by: Dict[str, List[int]] = dict()
        self.filter_mode: str = config.confuse['ImageLoading']['Filter']['Mode'].as_str().lower()

        for key in config.confuse['ImageLoading']['Filter']:
            if key.lower() in FileInfoEnrollment.keywords:
                values = config.confuse['ImageLoading']['Filter'][key].get()
                if values:
                    self.filter_by.update({key.lower(): values})

        self.sort_by: List[str] = [i.lower() for i in config.confuse['ImageLoading']['SortBy'].get()]
        self.group_by: List[str] = [i.lower() for i in config.confuse['ImageLoading']['GroupBy'].get()]

    def _load(self) -> NoReturn:
        self.image_files = [
            FileInfo.get_file_info(file) for file in
            (config.dir_source / str(config.database) / "cropped").glob('*.png')
        ]

        logger.info(f'Loaded {len(self.image_files)} files from {config.dir_source / str(config.database) / "cropped"}')

    def _is_filtration_needed(self) -> bool:
        return len(self.filter_by.items()) > 0

    def _filter(self) -> NoReturn:
        image_files_filter = self.image_files.copy()
        amount_before = len(image_files_filter)
        self.image_files.clear()

        while image_files_filter:
            file = image_files_filter.pop()

            valid = False
            for keyword, values in self.filter_by.items():
                if file.get_attribute_by_key(keyword) in values:
                    valid = self.filter_mode == 'inclusive'
                    if not valid:
                        break
                else:
                    break

            if valid:
                self.image_files.append(file)

        logger.info(f"Filtered {amount_before - len(self.image_files)} files")

    def _sort(self) -> NoReturn:
        for key in self.sort_by:
            self.image_files.sort(key=lambda f: f.get_attribute_by_key(key))

    def _group(self) -> NoReturn:
        for file in self.image_files:
            unique = tuple(file.get_attribute_by_keys(self.group_by))

            if unique in self.image_chunks.keys():
                self.image_chunks[unique].append(file)
            else:
                self.image_chunks[unique] = [file]

    def get_files(self) -> Union[List, Dict]:
        if not self.group_by:
            return self.image_files
        else:
            return self.image_chunks

    def get_chunks(
            self
    ) -> Tuple[Any, int]:
        if self.group_by:
            return self.image_chunks.values(), len(self.image_chunks.values())
        else:
            chunks_amount = 100
            return utils.chunks(self.image_files, chunks_amount), chunks_amount

    def _handle_debug_files(self) -> NoReturn:
        if not self.DEBUG_FILES:
            logger.warning("Debug files list is empty...")
        else:
            for file in self.DEBUG_FILES:
                self.image_files.append(
                    FileInfo.get_file_info(config.dir_source / str(config.database) / f"cropped/{file}.png")
                )

    def _handle_regular_loading(self) -> NoReturn:
        self._load()

        if self._is_filtration_needed():
            self._filter()

        if self.fraction < 1:
            self.image_files = self.image_files[:int(len(self.image_files) * abs(self.fraction))]
            logger.info(f"Shrank files list by {self.fraction * 100:.0f}%")

        if self.sort_by and self.shuffle:
            logger.info("Shuffle is dominant over SortBy flags, shuffling files list...")
            if self.seed:
                random.seed(666)
            random.shuffle(self.image_files)
        else:
            if self.sort_by:
                self._sort()

        if self.group_by:
            self._group()
