import cv2 as cv
import numpy as np
import glob
import random

import utils
from text_detection.morph import Morph


class Edges:

    def __init__(self, image_bgr: np.ndarray):
        self.image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
        self.image_yuv = cv.cvtColor(image_bgr, cv.COLOR_RGB2YUV)
        self.image_lab = cv.cvtColor(image_bgr, cv.COLOR_RGB2LAB)
        self.image_hsv = cv.cvtColor(image_bgr, cv.COLOR_RGB2HSV)

        # self._process_rgb()
        # self._process_yuv()
        # self._process_lab()
        self._process_hsv()

        # self._visualize()

    def _process_rgb(self):
        r, g, b = cv.split(self.image_yuv)
        self.image_rgb_edges_r = Morph.extract_edges(r)
        self.image_rgb_edges_g = Morph.extract_edges(g)
        self.image_rgb_edges_b = Morph.extract_edges(b)

    def _process_yuv(self):
        y, u, v = cv.split(self.image_yuv)
        self.image_yuv_edges_y = Morph.extract_edges(y)
        self.image_yuv_edges_u = Morph.extract_edges(u)
        self.image_yuv_edges_v = Morph.extract_edges(v)

    def _process_lab(self):
        l, a, b = cv.split(self.image_lab)
        self.image_lab_edges_l = Morph.extract_edges(l)
        self.image_lab_edges_a = Morph.extract_edges(a)
        self.image_lab_edges_b = Morph.extract_edges(b)

    def _process_hsv(self):
        h, s, v = cv.split(self.image_hsv)

        self.image_hsv_edges_h = Morph.extract_edges(h)
        self.image_hsv_edges_s = Morph.extract_edges(s)
        self.image_hsv_edges_v = Morph.extract_edges(v)

    def _visualize(self):
        self.image_visualization: np.ndarray = np.zeros((self.image_rgb.shape[0] * 4, self.image_rgb.shape[1] * 4, 3))

        def add_text(image: np.ndarray, text: str) -> np.ndarray:
            image_draw = np.copy(image)
            return cv.putText(image_draw, text, (50, 50), cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 1)

        first_row = np.hstack([
            add_text(self.image_rgb, "RGB"),
            add_text(utils.to_rgb(self.image_rgb_edges_r), "R"),
            add_text(utils.to_rgb(self.image_rgb_edges_g), "G"),
            add_text(utils.to_rgb(self.image_rgb_edges_b), "B")
        ])

        second_row = np.hstack([
            add_text(self.image_rgb, "YUV"),
            add_text(utils.to_rgb(self.image_yuv_edges_y), "Y"),
            add_text(utils.to_rgb(self.image_yuv_edges_u), "U"),
            add_text(utils.to_rgb(self.image_yuv_edges_v), "V")
        ])

        third_row = np.hstack([
            add_text(self.image_rgb, "LAB"),
            add_text(utils.to_rgb(self.image_lab_edges_l), "L"),
            add_text(utils.to_rgb(self.image_lab_edges_a), "A"),
            add_text(utils.to_rgb(self.image_lab_edges_b), "B")
        ])

        fourth_row = np.hstack([
            add_text(self.image_rgb, "HSV"),
            add_text(utils.to_rgb(self.image_hsv_edges_h), "H"),
            add_text(utils.to_rgb(self.image_hsv_edges_s), "S"),
            add_text(utils.to_rgb(self.image_hsv_edges_v), "V")
        ])

        self.image_visualization = np.vstack([first_row, second_row, third_row, fourth_row])


if __name__ == '__main__':
    files = glob.glob("D:/pharmapack/Enrollment/cropped/*.png")
    random.shuffle(files)

    for file in files:
        image = cv.imread(file, cv.IMREAD_COLOR)
        # image = utils.scale(image, 0.25)
        edges = Edges(image)
        h, s, v = cv.split(edges.image_hsv)
        to_edge = cv.bitwise_and(s, v)
        edges_img = Morph.extract_edges(to_edge)

        and_edges = cv.bitwise_and(edges.image_hsv_edges_v, edges.image_hsv_edges_s)

        to_disp = np.hstack([
            image,
            utils.to_rgb(edges_img),
            utils.to_rgb(and_edges)
        ])

        utils.display(to_disp)
        pass