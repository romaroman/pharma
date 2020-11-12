from typing import List, Any, Dict, NoReturn, Tuple

import pickle
import copyreg
from redis import Redis
import cv2 as cv
import numpy as np

import utils


def _pickle_keypoints(point):
    return cv.KeyPoint, (
        *point.pt, point.size, point.angle,
        point.response, point.octave, point.class_id
    )

redis_descriptors_db = Redis(db=3)
copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoints)


class DescriptorType(utils.CustomEnum):

    AKAZE = "AKAZE",
    SIFT = "SIFT",
    SURF = "SURF",
    ORB = "ORB",
    BRISK = "BRISK",
    KAZE = "KAZE",

    @classmethod
    def get_extractor_by_descriptor_type(cls, descriptor_type: 'DescriptorType') -> Any:
        if descriptor_type is cls.AKAZE:
            return cv.AKAZE_create()
        elif descriptor_type is cls.SIFT:
            return cv.SIFT_create()
        elif descriptor_type is cls.SURF:
            return cv.xfeatures2d.SURF_create()
        elif descriptor_type is cls.ORB:
            return cv.ORB_create()
        elif descriptor_type is cls.BRISK:
            return cv.BRISK_create()
        elif descriptor_type is cls.KAZE:
            return cv.KAZE_create()


class Extractor:

    def __init__(self, descriptor_types: List[DescriptorType]) -> NoReturn:
        self.extractors: Dict[DescriptorType, Any] = dict()

        for descriptor_type in descriptor_types:
            self.extractors[descriptor_type] = DescriptorType.get_extractor_by_descriptor_type(descriptor_type)

    def extract(self, image: np.ndarray) -> Dict[DescriptorType, Tuple[List[cv.KeyPoint], np.ndarray]]:
        # results = dict()
        # for descriptor_type, extractor in self.extractors.items():
        #     results[descriptor_type] =
        return dict([
            (descriptor_type, extractor.detectAndCompute(image, None))
            for descriptor_type, extractor in self.extractors.items()
        ])

    @staticmethod
    def save_descriptors(
            descriptors: Dict[DescriptorType, Tuple[List[cv.KeyPoint], np.ndarray]],
            filename: str
    ) -> NoReturn:
        redis_descriptors_db.append(filename, pickle.dumps(descriptors))

    @staticmethod
    def load_descriptors(filename: str) -> Dict[DescriptorType, Tuple[List[cv.KeyPoint], np.ndarray]]:
        return pickle.loads(redis_descriptors_db.get(filename))