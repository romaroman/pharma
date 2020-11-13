import os
import re
import glob
from pathlib import Path

import cv2 as cv
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from textdetector.enums import DetectionAlgorithm, EvalMetric
import utils


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['ses_status'] == 'success']
    df = df.drop(columns=df.columns[df.isna().all()].tolist())
    df = df.sample(frac=0.1, random_state=1)
    df = df.rename(columns={'fi_package_class': 'er_package_class'})
    df = df[df.columns[pd.Series(df.columns).str.startswith('er')]]
    df = df.fillna(df.mean())
    return df


def diff_TXT_and_ALL(df: pd.DataFrame) -> pd.DataFrame:
    df_res = pd.DataFrame()
    for col in df.columns:
        col = str(col)
        if col.find('ALL') != -1 and col.find('_R') == -1:
            df_res[col.replace('ALL', 'DIFF')] = df[col] - df[col.replace('ALL', 'TXT')]

    return df_res


def plot_distribution(data: np.ndarray):
    sns_plot = sns.displot(data, binwidth=0.01)
    fig = sns_plot.get_figure()
    fig.savefig("output.png")


if __name__ == '__main__':
    file_path = 'data/eval_2020-13-09.csv'
    # ts = utils.get_str_timestamp()
    save_path = f'/data/500gb/pharmapack/output_visualization/'

    if not Path(save_path).exists():
        os.makedirs(save_path)

    save_path += '{metric}.png'

    df = pd.read_csv(file_path, index_col=None)
    df = prepare_df(df)
    df_diff = diff_TXT_and_ALL(df)
    df_comb = pd.concat([df, df_diff])

    df_mean = df.groupby('er_package_class').mean().reset_index()
    df_std = df.groupby('er_package_class').std().reset_index()
    df_median = df.groupby('er_package_class').median().reset_index()

    metrics = [metric.blob() for metric in EvalMetric if metric.blob() != 'R']
    algorithms = [col.split('_')[1] for col in df.columns.to_list() if col.find('ALL_SNS') != -1]
    modes = ['ALL', 'TXT', 'DIFF', 'MEAN', 'STD', 'MEDIAN']
    nbins = 50

    y = df['er_package_class']
    # for metric in metrics:
    #     fig, axes = plt.subplots(len(modes), len(algorithms),
    #                              figsize=((len(algorithms)) * 6, len(modes) * 6), sharex=True)
    #     for mi, mode in enumerate(modes, start=0):
    #         for ai, algorithm in enumerate(algorithms, start=0):
    #             axis = axes[mi][ai]
    #             if mode in ['ALL', 'TXT', 'DIFF']:
    #                 col = f'er_{algorithm}_{mode}_{metric}'
    #                 axis.hist(df_comb[col].to_numpy(), bins=30)
    #             else:
    #                 col = f'er_{algorithm}_ALL_{metric}'
    #                 if mode == 'MEAN':
    #                     axis.hist(df_mean[col].to_numpy(), bins=nbins)
    #                 elif mode == 'STD':
    #                     axis.hist(df_std[col].to_numpy(), bins=nbins)
    #                 elif mode == 'MEDIAN':
    #                     axis.hist(df_median[col].to_numpy(), bins=nbins)
    #
    #             if mi == 0:
    #                 axis.set_title(f'{mode} {algorithm}')
    #             else:
    #                 axis.set_title(algorithm)
    #
    #     fig.savefig(save_path.format(metric=metric))

    pass
