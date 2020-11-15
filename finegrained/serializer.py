import copyreg
import pickle
from pathlib import Path
from typing import NoReturn, Dict, Any, Union, List, Tuple

import cv2 as cv
import numpy as np
from redis import Redis

from common import config
from common.enums import Descriptor


def __pickle_keypoints(point):
    return cv.KeyPoint, (
        *point.pt, point.size, point.angle,
        point.response, point.octave, point.class_id
    )


copyreg.pickle(cv.KeyPoint().__class__, __pickle_keypoints)


class Serializer:

    __db_reference = Redis(db=3)
    __db_verification = Redis(db=4)

    @classmethod
    def __get_db(cls, as_reference: bool) -> Redis:
        return cls.__db_reference if as_reference else cls.__db_verification

    @classmethod
    def __get_path(cls, as_reference: bool) -> Path:
        src_path = config.general.dir_source / 'Detections'
        return src_path / 'Reference' if as_reference else src_path / 'Verification'

    @classmethod
    def save_detection_to_redis(
            cls,
            keypoints: Tuple[List, np.ndarray],
            identifier: str,
            as_reference: bool = False
    ) -> NoReturn:
        cls.__get_db(as_reference).append(identifier, pickle.dumps(keypoints))

    @classmethod
    def save_detections_to_redis(cls, detections: Dict, identifier: str, as_reference: bool = False) -> NoReturn:
        for descriptor, keypoints in detections.items():
            cls.save_detection_to_redis(keypoints, f"{descriptor.blob()}:{identifier}", as_reference)

    @classmethod
    def load_detection_from_redis(cls, identifier: str, as_reference: bool = False) -> Dict:
        db = cls.__get_db(as_reference)
        # keys = db.keys(f'*:{key}')

        detections = dict()
        # for key in keys:
        #     key_str =
        descriptor = Descriptor[identifier.split(':')[0]]
        detections[descriptor] = pickle.loads(db.get(identifier))

        return detections

    @classmethod
    def load_detections_from_redis(cls, pattern: str, as_reference: bool = False) -> Union[Any, None]:
        keys = cls.__get_db(as_reference).keys(pattern)

        return [cls.load_detection_from_redis(key.decode("utf-8"), as_reference) for key in keys] if keys else None

    @classmethod
    def save_detections_to_file(cls, detections: Dict, identifier: str, as_reference: bool = False) -> NoReturn:
        dst_folder = cls.__get_path(as_reference)

        for descriptor, keypoints in detections.items():
            dst_path = dst_folder / f"{descriptor.blob()}:{identifier}.pkl"
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            with open(dst_path, 'wb') as file_dst:
                pickle.dump(keypoints, file_dst)

    @classmethod
    def load_detections_from_file(cls, identifier: str, as_reference: bool = False) -> Dict:
        src_path = cls.__get_path(as_reference)

        detections = dict()
        for file in src_path.glob(f'*{identifier}*.pkl'):
            descriptor = Descriptor[file.stem.split(':')[0]]

            with open(file, 'rb') as file_src:
                detections[descriptor] = pickle.load(file_src)

        return detections
