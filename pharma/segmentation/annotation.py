import math

from pathlib import Path
from xml.etree import ElementTree
from abc import ABC, abstractmethod
from typing import List, Tuple, NoReturn, Union, Dict

import cv2 as cv
import numpy as np
import pandas as pd

from pharma.common import config
from pharma.common.enums import AnnotationLabel

import pyutils as pu


class BoundingRectangleABC(ABC):

    def __init__(self, rectangle: ElementTree.Element) -> NoReturn:
        self.label: AnnotationLabel = AnnotationLabel[rectangle.find('name').text.capitalize()]

        self.width: Union[int, float] = 0
        self.height: Union[int, float] = 0

        self._set_color()

    @abstractmethod
    def to_polygon(self) -> np.ndarray:
        raise NotImplementedError

    def get_ratio(self) -> float:
        try:
            return self.width / self.height
        except ZeroDivisionError:
            return 0

    def get_area(self) -> int:
        return self.width * self.height

    def get_info(self) -> List:
        return [self.label, self.height, self.width, self.get_area(), self.get_ratio()]

    def _set_color(self):
        self.draw_color = self.label.get_color()


class BoundingRectangle(BoundingRectangleABC):

    def to_polygon(self) -> np.ndarray:
        return self.points.reshape(-1, 1, 2)

    def __init__(self, rectangle: ElementTree.Element) -> NoReturn:
        super().__init__(rectangle)

        bndbox = rectangle.find('bndbox')

        self.x_min = int(bndbox.find('xmin').text)
        self.y_min = int(bndbox.find('ymin').text)
        self.x_max = int(bndbox.find('xmax').text)
        self.y_max = int(bndbox.find('ymax').text)

        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min

        self.points = np.asarray(
            [
                (self.x_min, self.y_min),
                (self.x_max, self.y_min),
                (self.x_max, self.y_max),
                (self.x_min, self.y_max)
            ],
            dtype=np.int32
        )

    def draw(
            self,
            image: np.ndarray,
            color: Union[Tuple[int, int, int], None] = None,
            filled: bool = False,
    ) -> np.ndarray:
        return cv.rectangle(
            image,
            (self.x_min, self.y_min), (self.x_max, self.y_max),
            color=color if color else self.draw_color,
            thickness=-1 if filled else 2
        )

    def get_info(self) -> List:
        info = ["regular"]
        info.extend(BoundingRectangleABC.get_info(self))
        return info


class BoundingRectangleRotated(BoundingRectangleABC):

    def to_polygon(self) -> np.ndarray:
        return self.points

    def __init__(self, rectangle: ElementTree.Element) -> NoReturn:
        super().__init__(rectangle)

        robndbox = rectangle.find('robndbox')

        self.cx: float = float(robndbox.find('cx').text)
        self.cy: float = float(robndbox.find('cy').text)
        self.width: float = float(robndbox.find('w').text)
        self.height: float = float(robndbox.find('h').text)

        self.angle: float = float(robndbox.find('angle').text)

        self.points: np.ndarray = self._get_points()

    def _get_points(self) -> np.ndarray:
        def rotate_point(xc: float, yc: float, xp: float, yp: float, theta: float) -> Tuple[float, float]:
            x_offset = xp - xc
            y_offset = yp - yc

            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)

            p_resx = cos_theta * x_offset + sin_theta * y_offset
            p_resy = - sin_theta * x_offset + cos_theta * y_offset

            return xc + p_resx, yc + p_resy

        p0x, p0y = rotate_point(
            self.cx, self.cy,
            self.cx - self.width / 2, self.cy - self.height / 2,
            -self.angle
        )
        p1x, p1y = rotate_point(
            self.cx, self.cy,
            self.cx + self.width / 2, self.cy - self.height / 2,
            -self.angle
        )
        p2x, p2y = rotate_point(
            self.cx, self.cy,
            self.cx + self.width / 2, self.cy + self.height / 2,
            -self.angle
        )
        p3x, p3y = rotate_point(
            self.cx, self.cy,
            self.cx - self.width / 2, self.cy + self.height / 2,
            -self.angle
        )

        return np.asarray([(p0x, p0y), (p1x, p1y), (p2x, p2y), (p3x, p3y)], dtype=np.int32)

    def draw(
            self,
            image: np.ndarray,
            color: Union[Tuple[int, int, int], None] = None,
            filled: bool = True
    ) -> np.ndarray:
        if filled:
            return cv.fillPoly(image, [self.points], color if color else self.draw_color)
        else:
            return self._draw_not_filled(image)

    def get_area(self) -> float:
        return cv.contourArea(self.points)

    def _draw_not_filled(self, image: np.ndarray, color: Union[Tuple[int, int, int], None] = None) -> np.ndarray:

        def draw_line(image: np.ndarray, point1: np.ndarray, point2: np.ndarray) -> NoReturn:
            point1_tuple = pu.to_tuple(point1)
            point2_tuple = pu.to_tuple(point2)

            cv.line(image, point1_tuple, point2_tuple, color if color else self.draw_color, thickness=2)

        draw_line(image, self.points[0], self.points[1])
        draw_line(image, self.points[1], self.points[2])
        draw_line(image, self.points[2], self.points[3])
        draw_line(image, self.points[3], self.points[0])

        return image

    def get_info(self) -> List[str]:
        info = ['rotated']
        info.extend(BoundingRectangleABC.get_info(self))
        return info


class Annotation:

    def __init__(self, path: Path) -> NoReturn:
        self.root: ElementTree.Element = ElementTree.parse(str(path)).getroot()

        self.filename: str = self.root.find('filename').text
        self.size: Tuple[int, int, int] = self._get_size()
        self.bounding_rectangles: \
            List[Union[BoundingRectangle, BoundingRectangleRotated]] = self._get_bounding_rectangles()

        self.image_ref: np.ndarray = cv.imread(str(path).replace('.xml', '.png'), cv.IMREAD_COLOR)
        self.image_mask: np.ndarray = self.create_mask()

    def _get_size(self) -> Tuple[int, int, int]:
        width = int(self.root.find('size').find('width').text)
        height = int(self.root.find('size').find('height').text)
        depth = int(self.root.find('size').find('depth').text)

        return height, width, depth

    def _get_bounding_rectangles(self) -> List[Union[BoundingRectangle, BoundingRectangleRotated]]:
        rectangles = self.root.findall('object')

        bounding_rectangles = list()

        for rectangle in rectangles:
            rectangle_type = rectangle.find('type').text

            if rectangle_type == 'bndbox':
                bounding_rectangles.append(BoundingRectangle(rectangle))
            elif rectangle_type == 'robndbox':
                bounding_rectangles.append(BoundingRectangleRotated(rectangle))

        return bounding_rectangles

    def is_empty(self) -> bool:
        return self.get_amount_of_regions_by_labels(AnnotationLabel.to_list()) == 0

    def is_empty_text(self) -> bool:
        return self.get_amount_of_regions_by_labels(AnnotationLabel.get_list_of_text_labels()) == 0

    def is_empty_graphic(self) -> bool:
        return self.get_amount_of_regions_by_labels(AnnotationLabel.get_list_of_graphic_labels()) == 0

    def create_mask(
            self,
            color: Union[None, Tuple[int, int, int]] = None,
            filled: bool = False
    ) -> np.ndarray:
        self.image_mask = np.zeros_like(self.image_ref)

        for bounding_rectangle in self.bounding_rectangles:
            bounding_rectangle.draw(self.image_mask, color=color, filled=filled)

        return self.image_mask

    def create_mask_by_labels(
            self,
            labels: List[AnnotationLabel],
            color: Union[None, Tuple[int, int, int]] = None,
            filled: bool = False
    ) -> np.ndarray:

        image_mask = np.zeros_like(self.image_mask)

        for bounding_rectangle in self.bounding_rectangles:
            if bounding_rectangle.label in labels:
                bounding_rectangle.draw(image_mask, color=color, filled=filled)

        return image_mask

    def add_statistics(self, df: pd.DataFrame) -> NoReturn:
        file_info = [self.filename]

        for bounding_rectangle in self.bounding_rectangles:
            df.loc[len(df.index)] = file_info + bounding_rectangle.get_info()

    def get_amount_of_regions_by_labels(self, labels: List[AnnotationLabel]) -> int:
        amount = 0

        for bounding_rectangle in self.bounding_rectangles:
            if bounding_rectangle.label in labels:
                amount += 1

        return amount

    def get_list_of_labels_amount(self) -> List[int]:
        return [self.get_amount_of_regions_by_labels([label]) for label in AnnotationLabel]

    def get_dict_of_labels_amount(self) -> Dict[str, int]:
        result = dict()

        result['Total'] = 0
        for label in AnnotationLabel:
            current_amount = self.get_amount_of_regions_by_labels([label])

            result[str(label)] = current_amount
            result['Total'] += current_amount

        return result

    def to_dict(self) -> Dict[str, Union[int, str]]:
        return {'filename': self.filename, **self.get_dict_of_labels_amount()}

    def get_bounding_rectangles_by_labels(
            self,
            labels: List[AnnotationLabel]
    ) -> List[Union[BoundingRectangle, BoundingRectangleRotated]]:
        return [
            bounding_rectangle for bounding_rectangle in self.bounding_rectangles if bounding_rectangle.label in labels
        ]

    @staticmethod
    def load_by_pattern(pattern: str) -> 'Annotation':
        files = list((config.general.dir_source / 'Annotations').glob(pattern + ".xml"))

        if files:
            return Annotation(files[0])
        else:
            raise FileNotFoundError
