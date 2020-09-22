import random
import logging

from typing import NoReturn, List, Union, Dict, Tuple

import config
import utils

from fileinfo import FileInfoEnrollment, FileInfoRecognition, FileInfo


logger = logging.getLogger('loader')


class Loader:

    DEBUG_FILES = [
        # '',
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

        self.filter_mode: str = config.confuse['ImageLoading']['Filter']['Mode'].as_str().lower()
        self.filter_package_class: List[int] = config.confuse['ImageLoading']['Filter']['PackageClass'].get()
        self.filter_phone: List[int] = config.confuse['ImageLoading']['Filter']['Phone'].get()
        self.filter_distinct: List[int] = config.confuse['ImageLoading']['Filter']['Distinct'].get()
        self.filter_sample: List[int] = config.confuse['ImageLoading']['Filter']['Sample'].get()
        self.filter_angle: List[int] = config.confuse['ImageLoading']['Filter']['Angle'].get()
        self.filter_size: List[int] = config.confuse['ImageLoading']['Filter']['Size'].get()

        tolower = lambda l: [str(e).lower() for e in l]
        self.sort_by: List[str] = tolower(config.confuse['ImageLoading']['SortBy'].get())
        self.group_by: List[str] = tolower(config.confuse['ImageLoading']['GroupBy'].get())

    def _load(self) -> NoReturn:
        self.image_files = [FileInfo.get_file_info(file, config.database)
                            for file in (config.dir_source / "cropped").glob('*.png')]

        logger.info(f'Loaded {len(self.image_files)} files from {config.dir_source / "cropped"}')

    def _is_filtration_needed(self) -> bool:
        return bool(self.filter_package_class) or bool(self.filter_phone) or bool(self.filter_size) \
               or bool(self.filter_distinct) or bool(self.filter_angle) or bool(self.filter_sample)

    def _filter(self) -> NoReturn:
        apply_mode = lambda predicate: predicate if self.filter_mode == 'inclusive' else not predicate

        image_files_filter = self.image_files.copy()
        amount_before = len(image_files_filter)
        self.image_files.clear()

        while image_files_filter:
            file = image_files_filter.pop()
            if self.filter_package_class and apply_mode(file.package_class in self.filter_package_class):
                self.image_files.append(file)
            elif self.filter_phone and apply_mode(file.phone in self.filter_phone):
                self.image_files.append(file)
            elif self.filter_distinct and apply_mode(file.distinct in self.filter_distinct):
                self.image_files.append(file)
            elif self.filter_sample and apply_mode(file.sample in self.filter_sample):
                self.image_files.append(file)
            elif self.filter_angle and apply_mode(file.angle in self.filter_angle):
                self.image_files.append(file)
            elif self.filter_size and apply_mode(file.size in self.filter_size):
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

    def get_chunks(self):
        return self.image_chunks.values() if self.group_by else utils.chunks(self.image_files, config.mlt_cpus * 10)

    def _handle_debug_files(self) -> NoReturn:
        if not self.DEBUG_FILES:
            logger.warning("Debug files list is empty...")
        else:
            for file in self.DEBUG_FILES:
                self.image_files.append(
                    FileInfo.get_file_info(config.dir_source / f"cropped/{file}.png", config.database)
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
