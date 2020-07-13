import cv2 as cv
import numpy as np
import pandas as pd

import glob
import math
import pathlib

import xml.etree.ElementTree as ET
from typing import List, Tuple, NoReturn, Union
from abc import ABC, abstractmethod

import utils


class BoundingRectangleABC(ABC):

    def __init__(
            self,
            rectangle: ET.Element
    ) -> NoReturn:
        self.label: str = rectangle.find('name').text

        self.width: Union[int, float] = 0
        self.height: Union[int, float] = 0

        self.draw_color = {
            'text': (255, 0, 255),
            'number': (0, 255, 255),
            'watermark': (0, 0, 255),
            'image': (255, 0, 0),
            'barcode': (255, 255, 0),
        }.get(self.label, (128, 128, 128))

    def get_ratio(
            self
    ) -> float:
        try:
            return self.width / self.height
        except:
            return 0

    def get_area(
            self
    ) -> int:
        return self.width * self.height

    def get_info(self) -> List:
        return [self.label, self.height, self.width, self.get_area(), self.get_ratio()]


class Annotation:

    class BoundingRectangle(BoundingRectangleABC):

        def __init__(
                self,
                rectangle: ET.Element
                ) -> NoReturn:

            super().__init__(rectangle)

            bndbox = rectangle.find('bndbox')

            self.x_min = int(bndbox.find('xmin').text)
            self.y_min = int(bndbox.find('ymin').text)
            self.x_max = int(bndbox.find('xmax').text)
            self.y_max = int(bndbox.find('ymax').text)

            self.width = self.x_max - self.x_min
            self.height = self.y_max - self.y_min

            self.points = np.asarray([(self.x_min, self.y_min),
                                      (self.x_max, self.y_min),
                                      (self.x_max, self.y_max),
                                      (self.x_min, self.y_max)], dtype=np.int32)

        def draw(
                self,
                image: np.ndarray,
                filled: bool = False
                ) -> np.ndarray:

            return cv.rectangle(
                image,
                (self.x_min, self.y_min), (self.x_max, self.y_max), self.draw_color,
                thickness=-1 if filled else 2
            )

        def get_info(self) -> List:
            info = ['regular']
            info.extend(BoundingRectangleABC.get_info(self))
            return info

    class BoundingRectangleRotated(BoundingRectangleABC):

        def __init__(
                self,
                rectangle: ET.Element
                ) -> NoReturn:

            super().__init__(rectangle)

            robndbox = rectangle.find('robndbox')

            self.cx = float(robndbox.find('cx').text)
            self.cy = float(robndbox.find('cy').text)
            self.width = float(robndbox.find('w').text)
            self.height = float(robndbox.find('h').text)

            self.angle = float(robndbox.find('angle').text)

            self.points = self._get_points()

        def _get_points(
                self
                ) -> np.ndarray:

            def rotate_point(
                    xc: float,
                    yc: float,
                    xp: float,
                    yp: float,
                    theta
                    ) -> Tuple[float, float]:

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
                filled: bool = True
                ) -> np.ndarray:

            if filled:
                return cv.fillPoly(image, [self.points], self.draw_color)
            else:
                return self._draw_not_filled(image)

        def get_area(
                self
                ) -> float:
            return cv.contourArea(self.points)

        def _draw_not_filled(
                self,
                image: np.ndarray,
                ) -> np.ndarray:
            def draw_line(
                    image: np.ndarray,
                    point1: np.ndarray,
                    point2: np.ndarray
                    ) -> np.ndarray:

                point1_tuple = utils.to_tuple(point1)
                point2_tuple = utils.to_tuple(point2)

                return cv.line(image, point1_tuple, point2_tuple, self.draw_color, thickness=2)

            image = draw_line(image, self.points[0], self.points[1])
            image = draw_line(image, self.points[1], self.points[2])
            image = draw_line(image, self.points[2], self.points[3])
            image = draw_line(image, self.points[3], self.points[0])

            return image

        def get_info(self):
            info = ['rotated']
            info.extend(BoundingRectangleABC.get_info(self))
            return info

    def __init__(
            self,
            path_xml: str
            ) -> NoReturn:

        self.root = ET.parse(path_xml).getroot()
        self.filename = self.root.find('filename').text
        self.size = self._get_size()

        self.bounding_rectangles, self.bounding_rectangles_rotated = self._get_bounding_rectangles()

    def _get_size(
            self
            ) -> Tuple[int, int, int]:

        width = int(self.root.find('size').find('width').text)
        height = int(self.root.find('size').find('height').text)
        depth = int(self.root.find('size').find('depth').text)

        return height, width, depth

    def _get_bounding_rectangles(
            self
            ) -> Tuple[List[BoundingRectangle], List[BoundingRectangleRotated]]:

        rectangles = self.root.findall('object')

        bounding_rectangles = []
        bounding_rectangles_rotated = []

        for rectangle in rectangles:
            rectangle_type = rectangle.find('type').text

            if rectangle_type == "bndbox":
                bounding_rectangles.append(self.BoundingRectangle(rectangle))
            elif rectangle_type == "robndbox":
                bounding_rectangles_rotated.append(self.BoundingRectangleRotated(rectangle))

        return bounding_rectangles, bounding_rectangles_rotated

    def create_mask(
            self
            ) -> np.ndarray:

        image_mask = np.zeros(self.size, dtype=np.uint8)

        for bounding_rectangle in self.bounding_rectangles:
            bounding_rectangle.draw(image_mask)

        for bounding_rectangle_rotated in self.bounding_rectangles_rotated:
            bounding_rectangle_rotated.draw(image_mask)

        return image_mask

    def add_statistics(
            self,
            df: pd.DataFrame
            ) -> NoReturn:
        file_info = [self.filename]

        for bounding_rectangle in self.bounding_rectangles:
            df.loc[len(df.index)] = file_info + bounding_rectangle.get_info()

        for bounding_rectangle_rotated in self.bounding_rectangles_rotated:
            df.loc[len(df.index)] = file_info + bounding_rectangle_rotated.get_info()


if __name__ == '__main__':
    SRC_FOLDER = "D:/pharmapack/Enrollment/annotations/"

    files = ["PFP_Ph1_P0198_D01_S001_C4_az360_side1", "PFP_Ph1_P0090_D01_S001_C3_az360_side1", "PFP_Ph1_P0835_D01_S001_C2_az360_side1", "PFP_Ph1_P0837_D01_S001_C1_az360_side1"]

    df_statistics = pd.DataFrame({
        'filename': pd.Series([], dtype='float'),
        'type': pd.Series([], dtype='int'),
        'label': pd.Series([], dtype='str'),
        'height': pd.Series([], dtype='str'),
        'width': pd.Series([], dtype='float'),
        'area': pd.Series([], dtype='float'),
        'ratio': pd.Series([], dtype='object'),
        # TODO: add what needed
    })
    annotations = []

    # for file in [SRC_FOLDER + file + ".xml" for file in files]:
    for file in glob.glob(SRC_FOLDER + "/*.xml"):
        # print(file)
        annotation = Annotation(file)
        annotations.append(annotation)

        # image_mask = annotation.create_mask()
        # image_annotated = cv.imread(f"D:/pharmapack/Enrollment/reference_dataset/{annotation.filename}.png")

        # utils.display(cv.bitwise_xor(image_mask, image_annotated))
        annotation.add_statistics(df_statistics)

    df_statistics.to_csv('stats.csv')
