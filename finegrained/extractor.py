from typing import List, Any, Dict, Tuple, Union

import cv2 as cv
import numpy as np

from common.enums import Descriptor


class Extractor:

    __akaze = cv.AKAZE_create()
    __sift = cv.SIFT_create()
    # __surf = cv.xfeatures2d.SURF_create()
    __orb = cv.ORB_create()
    __brisk = cv.BRISK_create()
    __kaze = cv.KAZE_create()

    @classmethod
    def __get_extractor(cls, descriptor: Descriptor) -> Any:
        return {
            Descriptor.AKAZE: cls.__akaze,
            Descriptor.SIFT: cls.__sift,
            # Descriptor.SURF: cls.__surf,
            Descriptor.ORB: cls.__orb,
            Descriptor.BRISK: cls.__brisk,
            Descriptor.KAZE: cls.__kaze,
        }.get(descriptor)

    @classmethod
    def extract_descriptor(
            cls,
            image: np.ndarray,
            descriptor: Descriptor,
            image_mask: Union[None, np.ndarray] = None
    ) -> Tuple[List[cv.KeyPoint], np.ndarray]:
        return cls.__get_extractor(descriptor).detectAndCompute(image, image_mask)

    @classmethod
    def extract_descriptors(
            cls,
            image: np.ndarray,
            descriptors: List[Descriptor],
            image_mask: Union[None, np.ndarray] = None
    ) -> Dict[Descriptor, Tuple[List[cv.KeyPoint], np.ndarray]]:
        return dict([
            (descriptor, cls.extract_descriptor(image, descriptor, image_mask))
            for descriptor in descriptors
        ])
