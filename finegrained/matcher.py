import re
from typing import List, NoReturn, Dict, Tuple

import cv2 as cv
import numpy as np
import pydegensac

from common import config
from finegrained.serializer import Serializer
from segmentation.annotation import Annotation

import utils


class Matcher:

    __brute = cv.BFMatcher_create()
    __flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    @classmethod
    def match_brute_candidate(cls):
        pass

    @classmethod
    def match_flann_candidate(
            cls,
            keypoints_verification: Tuple[List[cv.KeyPoint], np.ndarray],
            keypoints_reference: Tuple[List[cv.KeyPoint], np.ndarray],
            image_ver: np.ndarray,
            image_ref: np.ndarray
    ):
        kp1, des1 = keypoints_verification
        kp2, des2 = keypoints_reference

        matches = cls.__flann.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)

        matches_mask = [[0, 0] for i in range(len(matches))]

        matches_good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matches_mask[i] = [1, 0]
                matches_good.append(m)

        # draw_params = dict(matchColor=(0, 255, 0),
        #                    singlePointColor=(255, 0, 0),
        #                    matchesMask=matches_mask,
        #                    flags=cv.DrawMatchesFlags_DEFAULT)
        # img3 = cv.drawMatchesKnn(image_ver, kp1, image_ref, kp2, matches, None, **draw_params)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches_good]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches_good]).reshape(-1, 2)

        try:
            # H, mask = pydegensac.findHomography(src_pts, dst_pts, 3.0)
            F, Fmask = pydegensac.findFundamentalMatrix(src_pts, dst_pts, 3.0)
            return sum(Fmask)
        except:
            return 0
