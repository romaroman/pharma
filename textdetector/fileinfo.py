import re

from pathlib import Path
from abc import ABC
from typing import NoReturn, Union, Dict, List, Pattern

import config
import utils

from enums import FileDatabase


class FileInfo(ABC):
    keywords = ['packageclass', 'phone', 'distinct', 'sample', 'size']

    regular_expressions = dict(zip(keywords, [
        r'P[0-9][0-9][0-9][0-9]',
        r'Ph[1-3]',
        r'D[0-9][0-9]',
        r'S[0-9][0-9][0-9]',
        r'C[0-9]'
    ]))

    @staticmethod
    def get_file_info(file_path: Path) -> Union["FileInfoEnrollment", "FileInfoRecognition"]:

        if config.database is FileDatabase.Enrollment:
            return FileInfoEnrollment(file_path)
        elif config.database in FileDatabase.get_list_of_recognition_databases():
            return FileInfoRecognition(file_path)

    def __init__(self, file_path: Path) -> NoReturn:
        self.path: Path = file_path
        self.filename: str = file_path.stem

        self.package_class: int = self._extract_numerical_info('packageclass')
        self.phone: int = self._extract_numerical_info('phone')
        self.distinct: int = self._extract_numerical_info('distinct')
        self.sample: int = self._extract_numerical_info('sample')
        self.size: int = self._extract_numerical_info('size')

    def get_annotation_pattern(self) -> Pattern:
        return re.compile(f"PFP_Ph._P{utils.zfill_n(self.package_class)}_D0{self.distinct}_S00._C._az..._side.")

    def get_verification_pattern(self) -> Pattern:
        return re.compile(f"PFP_Ph1_P{utils.zfill_n(self.package_class)}_D0{self.distinct}_S00{self.sample}_C._az360_side.")

    def get_unique_identifier(self) -> str:
        return f"{utils.zfill_n(self.package_class, 4)}_{utils.zfill_n(self.distinct, 2)}"

    def to_dict(self) -> Dict[str, Union[str, int]]:
        return dict(zip(self.keywords, self.to_list()))

    def to_list(self) -> List[Union[str, int]]:
        return [self.package_class, self.phone, self.distinct, self.sample, self.size]

    def get_attribute_by_key(self, key: str) -> Union[int, None]:
        return self.to_dict().get(key, None)

    def get_attribute_by_keys(self, keys: List[str]) -> List[Union[str, int]]:
        return [self.get_attribute_by_key(key) for key in keys]

    def _extract_numerical_info(self, keyword: str) -> int:
        substring = self._extract_str_info(keyword)
        return int(re.sub(r'[^0-9]', '', substring))

    def _extract_str_info(self, keyword: str) -> str:
        return re.search(self.regular_expressions[keyword], self.filename).group(0)


class FileInfoEnrollment(FileInfo):
    keywords_cls = ['angle', 'side']
    keywords = FileInfo.keywords + keywords_cls

    regular_expressions = dict(
        FileInfo.regular_expressions,
        **dict(zip(keywords_cls, [r'az[0-3][0-9]0', r'side[0-9]']))
    )

    def __init__(self, file_path: Path) -> NoReturn:
        """
        :param filename: without extension, example: "PFP_Ph1_P0003_D01_S001_C2_az360_side1"
        """
        super().__init__(file_path)

        self.angle: int = self._extract_numerical_info('angle')
        self.side: int = self._extract_numerical_info('side')

    def to_dict(self) -> Dict[str, int]:
        return dict(super(FileInfoEnrollment, self).to_dict(), **dict(zip(self.keywords_cls, [self.angle, self.side])))

    def to_list(self) -> List[Union[str, int]]:
        return super(FileInfoEnrollment, self).to_list() + [self.angle, self.side]


class FileInfoRecognition(FileInfo):
    keywords_cls = ['RS']
    keywords = FileInfo.keywords + keywords_cls

    regular_expressions = dict(
        FileInfo.regular_expressions,
        **dict(zip(keywords_cls, [r'S[0-9]']))
    )

    def __init__(self, file_path: Path) -> NoReturn:
        """
        :param filename: without extension, example: "PharmaPack_R_I_S1_Ph1_P0016_D01_S001_C2_S1"
        """
        super().__init__(file_path)
        self.RS: int = self._extract_numerical_info('RS')

    def to_dict(self) -> Dict[str, int]:
        return dict(super(FileInfoRecognition, self).to_dict(), **{'RS': self.RS})

    def to_list(self) -> List[Union[str, int]]:
        return super(FileInfoRecognition, self).to_list() + [self.RS]
