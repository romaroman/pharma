import functools
import os
import argparse
from pathlib import Path
import multiprocessing as mlt
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from textdetector.enums import EvalMetric


parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
parser.add_argument('dst_folder', type=str)

args = parser.parse_args()

def convert_df_cell_to_np_array(string: str) -> np.ndarray:
    return np.array(eval(' '.join([p for p in string.split(' ') if p != '']).replace('[ ', '[').replace(' ', ',')))


def process_single_class(df_pclass_and_pclass: Tuple[pd.DataFrame, int]):
    df_pclass, pclass = df_pclass_and_pclass

    fig, axes = plt.subplots(len(algorithms), len(metrics), figsize=((len(metrics)) * 6, len(algorithms) * 6), sharex=True)

    dst_path = dst_folder / f"{str(int(pclass)).zfill(4)}.png"

    for ai, algorithm in enumerate(algorithms, start=0):
        df_evr_alg = df_pclass[f'evr_{algorithm}']
        df_evr_alg = df_evr_alg.dropna()
        scores_region = np.array([convert_df_cell_to_np_array(cell) for cell in df_evr_alg])

        df_evm_alg = df_pclass[f'evm_{algorithm}']
        df_evm_alg = df_evm_alg.dropna()
        scores_mask = np.array([convert_df_cell_to_np_array(cell) for cell in df_evm_alg])

        for mi, metric in enumerate(metrics, start=0):
            ax = axes[ai][mi]

            ax.set_xlabel('Sample')
            ax.set_ylabel('Score')
            ax.set_title(f'{algorithm} {metric}')

            xmax_line = scores_region.shape[0]
            reg_means = []
            reg_medians = []

            for i, array in enumerate(scores_region[:10], start=0):
                try:
                    values = array[:, mi + 2].clip(min=0)
                    xmax = len(values)

                    xs = [i + 1] * xmax
                    ax.scatter(xs, values, c='tab:blue', alpha=0.15)

                    mean = np.mean(values)
                    reg_means.append(mean)
                    ax.scatter(i + 1, mean, c='tab:orange')

                    median = np.median(values)
                    reg_medians.append(median)
                    ax.scatter(i + 1, median, c='tab:red')

                    ax.scatter(i + 1, scores_mask[i][mi], c='tab:green')
                except:
                    pass

            ax.hlines(y=np.mean(reg_means), xmin=1, xmax=xmax_line, colors='tab:orange', linestyles='--')
            ax.hlines(y=np.mean(reg_medians), xmin=1, xmax=xmax_line, colors='tab:red', linestyles='--')
            ax.hlines(y=np.mean(scores_mask[:, mi]), xmin=1, xmax=xmax_line, colors='tab:green', linestyles='--')

    os.makedirs(str(dst_path.parent.resolve()), exist_ok=True)
    plt.savefig(dst_path)
    print(str(dst_path))

    plt.close()
    plt.cla()
    plt.clf()



if __name__ == '__main__':

        df = pd.read_csv(args.csv_path, index_col=False)
        df = df.rename(columns={'fi_package_class': 'evr_package_class', 'fi_phone': 'evr_phone'})
        df = df.drop(columns=df.columns[df.isna().all()].tolist())
        df = df.drop(columns=df.columns[pd.Series(df.columns).str.startswith('fi')])
        df = df.drop(columns=df.columns[pd.Series(df.columns).str.startswith('ses')])

        metrics = [metric for metric in EvalMetric.to_list()]
        algorithms = ['MI1', 'MI2', 'MSER', 'MSER+MI1+MI2']

        dst_folder = Path(args.dst_folder)
        processed = [float(p.name) for p in dst_folder.glob("*")]

        dfs_ps = []
        for pclass in df['evr_package_class'].unique().tolist():
            if pclass in processed:
                continue

            dfs_ps.append((df[df['evr_package_class'] == pclass], pclass))


        with mlt.Pool(processes=mlt.cpu_count()) as pool:
            results = pool.map(process_single_class, dfs_ps)
            pool.close()
