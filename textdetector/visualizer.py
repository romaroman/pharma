import os
from pathlib import Path
from typing import NoReturn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from enums import EvalMetric


def convert_df_cell_to_np_array(string: str) -> np.ndarray:
    return np.array(eval(' '.join([p for p in string.split(' ') if p != '']).replace('[ ', '[').replace(' ', ',')))


def scatter_eval_by_pkg_class(file_path: Path) -> NoReturn:
    df = pd.read_csv(file_path, index_col=False)
    df = df.rename(columns={'fi_package_class': 'evr_package_class', 'fi_phone': 'evr_phone'})
    df = df.drop(columns=df.columns[df.isna().all()].tolist())
    df = df[df.columns[pd.Series(df.columns).str.startswith('evr')]]

    metrics = [metric for metric in EvalMetric.to_list()]
    algorithms = ['MI1', 'MI2', 'MSER', 'MSER+MI1+MI2']

    dst_folder = Path('/data/3tb/pharmapack/scatter')

    # fig, axes = plt.subplots(l
    # en(algorithms), len(metrics), figsize=((len(algorithms)) * 6, len(metrics) * 6),
    #                          sharex=True)
    processed = [float(p.name) for p in Path("/data/3tb/pharmapack/scatter/").glob("*")]
    for pclass in df['evr_package_class'].unique().tolist():
        if pclass in processed:
            continue

        df_pclass = df[df['evr_package_class'] == pclass]
        for ai, algorithm in enumerate(algorithms, start=0):
            df_evr_alg = df_pclass[f'evr_{algorithm}']
            df_evr_alg = df_evr_alg.dropna()
            # _pclass['evr_phone']
            s_array = np.array([convert_df_cell_to_np_array(cell) for cell in df_evr_alg])
            for mi, metric in enumerate(metrics, start=0):
                plt.figure(figsize=(10, 15))
                plt.xlabel('Sample')
                plt.ylabel('Score')
                plt.title(f'{algorithm} {metric}')
                for i, array in enumerate(s_array, start=0):
                    try:
                        values = array[:, mi + 2].clip(min=0)
                        xs = [i + 1] * len(values)
                        plt.scatter(xs, values, c='tab:blue', alpha=0.1)
                    except:
                        pass
                # plt.show()

                save_path = dst_folder / str(int(pclass)).zfill(4) / f"{algorithm}_{metric}.png"
                os.makedirs(str(save_path.parent.resolve()), exist_ok=True)
                plt.savefig(save_path)
                plt.close()

                print(str(save_path))


if __name__ == '__main__':
    csv_path = Path('/home/rchaban/session_pd_2020-09-18_21-02.csv')
    scatter_eval_by_pkg_class(csv_path)
