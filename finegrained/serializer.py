import re
import pickle
import copyreg

from typing import NoReturn, Dict, Any, Union

import cv2 as cv
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
    __db_enrollment = Redis(db=4)

    @classmethod
    def __get_db(cls, reference: bool) -> Redis:
        return cls.__db_reference if reference else cls.__db_enrollment

    @classmethod
    def save_to_redis(cls, descriptors: Dict, key: str, as_reference: bool = False) -> NoReturn:
        cls.__get_db(as_reference).append(key, pickle.dumps(descriptors))

    @classmethod
    def load_from_redis_by_key(cls, key: str, as_reference: bool = False) -> Dict:
        return pickle.loads(cls.__get_db(as_reference).get(key))

    @classmethod
    def load_from_redis_by_pattern(cls, pattern: str, as_reference: bool = False) -> Union[Any, None]:
        keys = cls.__get_db(as_reference).keys(pattern)
        return cls.load_from_redis_by_key(keys[0], as_reference) if len(keys) == 1 else None

    @classmethod
    def save_file(cls, descriptors: Dict, filename: str) -> NoReturn:
        for descriptor, keypoints in descriptors.items():
            dst_path = config.general.dir_source / 'Descriptors' / str(descriptor) / f"{filename}.pkl"
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            with open(dst_path, 'wb') as file_dst:
                pickle.dump(keypoints, file_dst)

    @classmethod
    def load_file(cls, filename: str) -> Dict:
        src_path = config.general.dir_source / 'Descriptors'

        descriptors = dict()
        for file in src_path.glob(f'**/{filename}.pkl'):
            descriptor = Descriptor(file.parent.stem)

            with open(file, 'rb') as file_src:
                descriptors[descriptor] = pickle.load(file_src)

        return descriptors
