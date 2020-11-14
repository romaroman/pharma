from typing import List, NoReturn, Dict, Tuple

import cv2 as cv
import numpy as np
import pydegensac

import utils


class Matcher:

    brute = cv.BFMatcher_create()
    flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    @classmethod
    def match(
            cls,
            keypoints_ver: Tuple[List[cv.KeyPoint], np.ndarray],
            keypoints_ref: Tuple[List[cv.KeyPoint], np.ndarray],
            image_ver: np.ndarray,
            image_ref: np.ndarray,
            matcher=flann
    ):
        kp_ver, des_ver = keypoints_ver
        kp_ref, des_ref = keypoints_ref

        matches = matcher.knnMatch(des_ver.astype(np.float32), des_ref.astype(np.float32), k=2)

        matches_mask = [[0, 0] for i in range(len(matches))]

        matches_good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matches_mask[i] = [1, 0]
                matches_good.append(m)

        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=matches_mask,
            flags=cv.DrawMatchesFlags_DEFAULT
        )
        image_visualization = cv.drawMatchesKnn(image_ver, kp_ver, image_ref, kp_ref, matches, None, **draw_params)

        src_pts = np.float32([kp_ver[m.queryIdx].pt for m in matches_good]).reshape(-1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches_good]).reshape(-1, 2)

        try:
            # H, mask = pydegensac.findHomography(src_pts, dst_pts, 3.0)
            F, Fmask = pydegensac.findFundamentalMatrix(src_pts, dst_pts, 3.0)
            return [len(kp_ver), len(kp_ref), len(matches_good), sum(Fmask)], image_visualization
        except:
            return [len(kp_ver), len(kp_ref), len(matches_good), 0], image_visualization
