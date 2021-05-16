from typing import List, Tuple

import cv2 as cv
import numpy as np
import pydegensac
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

import pyutils as pu


class Matcher:

    brute = cv.BFMatcher_create()
    flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    @classmethod
    def match(
            cls,
            keypoints_ver: Tuple[List[cv.KeyPoint], np.ndarray],
            keypoints_ref: Tuple[List[cv.KeyPoint], np.ndarray],
            # image_ver: np.ndarray,
            # image_ref: np.ndarray,
            matcher=flann
    ):
        kp_ver, des_ver = keypoints_ver
        kp_ref, des_ref = keypoints_ref

        matches = matcher.knnMatch(des_ver.astype(np.float32), des_ref.astype(np.float32), k=2)

        matches_mask = [[0, 0] for _ in range(len(matches))]

        matches_good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matches_mask[i] = [1, 0]
                matches_good.append(m)

        # draw_params = dict(
        #     matchColor=(0, 255, 0),
        #     singlePointColor=(255, 0, 0),
        #     matchesMask=matches_mask,
        #     flags=cv.DrawMatchesFlags_DEFAULT
        # )
        # image_visualization = cv.drawMatchesKnn(image_ver, kp_ver, image_ref, kp_ref, matches, None, **draw_params)

        src_pts = np.float32([kp_ver[m.queryIdx].pt for m in matches_good]).reshape(-1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches_good]).reshape(-1, 2)

        ransac_matches_amount = np.nan
        # mse_score = np.nan
        # ssim_score = np.nan
        try:
            F, Fmask = pydegensac.findFundamentalMatrix(src_pts, dst_pts, 3.0)
            ransac_matches_amount = sum(Fmask)

            # H, mask = pydegensac.findHomography(src_pts, dst_pts, 3.0)
            # image_ver_warped = cv.warpPerspective(image_ver, H, pu.swap_dimensions(image_ref.shape))
            # mse_score = mean_squared_error(image_ver_warped, image_ref)
            # ssim_score = ssim(image_ver_warped, image_ref, data_range=image_ref.max() - image_ref.min(), multichannel=True)
        except:
            pass
        finally:
            return [
                len(kp_ver),
                len(kp_ref),
                len(matches),
                len(matches_good),
                ransac_matches_amount,
                # mse_score,
                # ssim_score
            ]
