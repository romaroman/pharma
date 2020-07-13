import re
import pathlib

from typing import NoReturn, Union
from enum import Enum
from abc import ABC


class Phone(Enum):

    def __str__(self):
        return str(self.value)

    Phone1 = 1,
    Phone2 = 2,
    Phone3 = 3


class FileInfo(ABC):

    res = {
        'phone': r'Ph[1-3]',
        'package_class': r'P[0-9][0-9][0-9][0-9]',
        'distinct': r'D[0-9][0-9]',
        'sample': r'S[0-9][0-9][0-9]',
        'size': r'C[0-9]'
   }

    def __init__(self, file_path: str):
        self.filename = pathlib.Path(file_path).stem

        self.phone = self._extract_numerical_info('phone')
        self.package_class = self._extract_numerical_info('package_class')
        self.distinct = self._extract_numerical_info('distinct')
        self.sample = self._extract_numerical_info('sample')
        self.size = self._extract_numerical_info('size')

    def _extract_numerical_info(self, keyword: str) -> int:
        substring = self._extract_substring(keyword)
        return int(re.sub(r'[^0-9]', '', substring))

    def _extract_substring(self, keyword: str) -> str:
        return re.search(self.res[keyword], self.filename).group(0)


class FileInfoEnrollment(FileInfo):

    res = dict({
        'angle': r'az[0-3][0-9]0',
        'side': r'side[0-9]',
    }, **FileInfo.res)

    def __init__(self, filename: str):
        """
        :param filename: without extension, example: "PFP_Ph1_P0003_D01_S001_C2_az360_side1"
        """
        super().__init__(filename)

        self.angle = self._extract_numerical_info('angle')
        self.side = self._extract_numerical_info('side')


class FileInfoRecognition(FileInfo):

    res = dict({
        'RS': r'S[0-9]',
    }, **FileInfo.res)

    def __init__(self, filename: str) -> NoReturn:
        """
        :param filename: without extension, example: "PharmaPack_R_I_S1_Ph1_P0016_D01_S001_C2_S1"
        """
        super().__init__(filename)

        self.RS = self._extract_numerical_info('RS')


def get_file_info(file_path: str, database: str) -> Union[FileInfoEnrollment, FileInfoRecognition]:

    if database == 'Enrollment':
        return FileInfoEnrollment(file_path)
    if database.startswith('PharmaPack'):
        return FileInfoRecognition(file_path)
