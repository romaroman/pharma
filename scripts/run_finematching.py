from itertools import product
from typing import Tuple

import cv2 as cv
import numpy as np
import pandas as pd
from numpy.distutils.command.config import config
from p_tqdm import p_map

import utils
from common import config
from finegrained import Serializer, Matcher


def create_image_ver_draw(id: str):
    image_mask_filled = cv.imread(
        str(config.general.dir_source / 'MasksUnaligned' / f'{id}.png'), 0
    )
    filename = id.split(':')[-1]
    image_orig = cv.imread(
        str(config.general.dir_source / str(config.general.database) / 'cropped' / f'{filename}.png')
    )
    contours, _ = cv.findContours(image_mask_filled, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    image_ver_mask = np.zeros_like(image_orig)
    image_ver_mask = cv.drawContours(image_ver_mask, contours, -1, (0, 255, 255), 3)
    return cv.add(image_orig, image_ver_mask)


def find_image_ref_orig(pattern: str):
    files_possible = list((config.general.dir_source / 'MasksUnaligned').glob(f'{pattern}*az360*'))

    if not files_possible:
        return None

    file = files_possible[0]

    filename = file.stem.split(':')[-1]
    image_orig = cv.imread(
        str(config.general.dir_source / str(config.general.database) / 'cropped' / f'{filename}.png')
    )
    return image_orig


def create_image_ref_draw(pattern: str):

    files_possible = list((config.general.dir_source / 'MasksUnaligned').glob(f'{pattern}*az360*'))

    if not files_possible:
        return None

    file = files_possible[0]

    image_mask_filled = cv.imread(str(file), 0)
    filename = file.stem.split(':')[-1]
    image_orig = cv.imread(
        str(config.general.dir_source / str(config.general.database) / 'cropped' / f'{filename}.png')
    )
    contours, _ = cv.findContours(image_mask_filled, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    image_ver_mask = np.zeros_like(image_orig)
    image_ver_mask = cv.drawContours(image_ver_mask, contours, -1, (0, 255, 255), 3)
    return cv.add(image_orig, image_ver_mask)


def match_single(df_iterrow: Tuple[int, pd.Series]):
    index, row = df_iterrow
    alg = row.segmentation_algorithm

    filename_ver = f"PFP_{row.sample_actual}_side1"
    id_ver = f"{row.segmentation_algorithm}:{filename_ver}"

    image_ver_orig = cv.imread(
        str(config.general.dir_source / str(config.general.database) / 'cropped' / f'{filename_ver}.png')
    )
    # image_ver_draw = create_image_ver_draw(id_ver)

    detections_ver = Serializer.load_detections_from_file(identifier=id_ver, as_reference=False)

    package_candidates = [v for k, v in row.to_dict().items() if k.startswith('package_candidate')]

    results_single = []
    for phone, package_candidate in product([1, 2, 3], package_candidates):
        result_base = [row.sample_actual, alg, row.package_actual, phone, package_candidate]

        try:
            id_ref = f'{row.segmentation_algorithm}:*Ph{phone}*{package_candidate}'
            # image_ref_draw = create_image_ref_draw(id_ref)
            image_ref_orig = find_image_ref_orig(id_ref)

            detections_ref = Serializer.load_detections_from_file(identifier=id_ref, as_reference=True)

            for descriptor in detections_ver.keys():
                scores = Matcher.match(detections_ver[descriptor], detections_ref[descriptor], image_ver_orig, image_ref_orig)

                # for ii, (res, label) in enumerate(
                #         zip(scores, ["ver keypoints", "ref keypoints", "matches", "good matches", "ransac matches"]),
                #         start=1
                # ):
                #     image_visualization = utils.add_text(
                #         image_visualization, text=f"{label}:{res}", point=(25, 35 * ii), scale=1, color=(0, 0, 255)
                #     )

                # dst_path_vis = config.general.dir_source / 'FineMatchingVisualization' / row.package_actual / row.sample_actual \
                #                / f"Alg:{alg}_Phone{phone}:_Desc:{descriptor.blob()}_Cand:{package_candidate}.png"
                # dst_path_vis.parent.mkdir(exist_ok=True, parents=True)
                # cv.imwrite(str(dst_path_vis), image_visualization)
                results_single.append(
                    result_base + [descriptor.blob()] + scores
                )
        except:
            results_single.append(result_base)

    return results_single


if __name__ == '__main__':
    utils.suppress_warnings()

    df = pd.read_csv(config.general.dir_source / 'candidates_prepared.csv', index_col=None)
    rows = list(df.iterrows())

    # for row in rows:
    #     match_single(row)

    results = p_map(match_single, rows, num_cpus=44)

    results_flattenned = []
    for result_complex in results:
        for results_s in result_complex:
            results_flattenned.append(results_s)

    df_result = pd.DataFrame(results_flattenned)
    df_result.to_csv('finematching_result.csv')
