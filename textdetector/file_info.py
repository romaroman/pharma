import re
import pathlib

from abc import ABC
from enum import Enum, auto
from collections import OrderedDict
from typing import NoReturn, Union, Dict, List, Pattern

import utils


class Phone(utils.CustomEnum):
    Phone1 = auto(),
    Phone2 = auto(),
    Phone3 = auto()


class Database(utils.CustomEnum):
    Enrollment = auto(),
    PharmaPack_R_I_S1 = auto(),
    PharmaPack_R_I_S2 = auto(),
    PharmaPack_R_I_S3 = auto(),


class FileInfo(ABC):

    regular_expressions = {
        'phone': r'Ph[1-3]',
        'package_class': r'P[0-9][0-9][0-9][0-9]',
        'distinct': r'D[0-9][0-9]',
        'sample': r'S[0-9][0-9][0-9]',
        'size': r'C[0-9]'
    }

    @staticmethod
    def get_file_info(file_path: str, database: Database) -> Union["FileInfoEnrollment", "FileInfoRecognition"]:

        if database is Database.Enrollment:
            return FileInfoEnrollment(file_path)
        if database in [Database.PharmaPack_R_I_S1, Database.PharmaPack_R_I_S2, Database.PharmaPack_R_I_S3]:
            return FileInfoRecognition(file_path)
        else:
            raise ValueError

    def __init__(self, file_path: str) -> NoReturn:
        self.filename = pathlib.Path(file_path).stem

        self.phone = self._extract_numerical_info('phone')
        self.package_class = self._extract_numerical_info('package_class')
        self.distinct = self._extract_numerical_info('distinct')
        self.sample = self._extract_numerical_info('sample')
        self.size = self._extract_numerical_info('size')

    def get_annotation_pattern(self) -> Pattern:
        return re.compile(
            f"PFP_Ph._P{str(self.package_class).zfill(4)}_D{str(self.distinct).zfill(2)}_S00._C._az..._side."
        )

    def to_dict(self) -> Dict[str, int]:
        return {
           'filename': self.filename,
           'phone': self.phone,
           'package_class': self.package_class,
           'distinct': self.distinct,
           'sample': self.sample,
           'size': self.size
        }

    def to_list(self) -> List[Union[str, int]]:
        return [self.filename, self.phone, self.package_class, self.distinct, self.sample, self.size]

    def _extract_numerical_info(self, keyword: str) -> int:
        substring = self._extract_str_info(keyword)
        return int(re.sub(r'[^0-9]', '', substring))

    def _extract_str_info(self, keyword: str) -> str:
        return re.search(self.regular_expressions[keyword], self.filename).group(0)


class FileInfoEnrollment(FileInfo):

    regular_expressions = dict({
        'angle': r'az[0-3][0-9]0',
        'side': r'side[0-9]',
    }, **FileInfo.regular_expressions)

    def __init__(self, filename: str) -> NoReturn:
        """
        :param filename: without extension, example: "PFP_Ph1_P0003_D01_S001_C2_az360_side1"
        """
        super().__init__(filename)

        self.angle = self._extract_numerical_info('angle')
        self.side = self._extract_numerical_info('side')

    def to_dict(self) -> Dict[str, int]:
        return dict(super(FileInfoEnrollment, self).to_dict(), **{'angle': self.angle, 'side': self.side})

    def to_list(self) -> List[Union[str, int]]:
        return super(FileInfoEnrollment, self).to_list() + [self.angle, self.side]


class FileInfoRecognition(FileInfo):

    regular_expressions = dict({
        'RS': r'S[0-9]',
    }, **FileInfo.regular_expressions)

    def __init__(self, filename: str) -> NoReturn:
        """
        :param filename: without extension, example: "PharmaPack_R_I_S1_Ph1_P0016_D01_S001_C2_S1"
        """
        super().__init__(filename)
        self.RS = self._extract_numerical_info('RS')

    def to_dict(self) -> Dict[str, int]:
        return dict(super(FileInfoRecognition, self).to_dict(), **{'RS': self.RS})

    def to_list(self) -> List[Union[str, int]]:
        return super(FileInfoRecognition, self).to_list() + [self.RS]
