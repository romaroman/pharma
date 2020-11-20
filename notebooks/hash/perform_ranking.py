import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import mode
from tqdm.notebook import tqdm
from p_tqdm import p_map
import warnings
import traceback
from itertools import chain

warnings.filterwarnings("ignore")
fill_value = 100


def calc_one(df):
    sample_actual = df.iloc[0].sample_actual
    package_actual = df.iloc[0].package_actual
    
    results = []
    for alg in df.algorithm.unique():
        try:
            df_piv = df[df.algorithm == alg].pivot_table(
                index='package_predicted', columns='descriptor_actual', values='distance', fill_value=np.nan,
                aggfunc=[np.mean, np.median, 'size', sum, min]#, ]
            )
            df_piv = df_piv.reset_index()

            df_rank = pd.DataFrame(columns=df_piv.columns)

            df_rank['package_predicted'] = df_piv['package_predicted']

            rank_ascending = ['mean', 'median', 'sum', 'min'] # 
            df_rank.loc[:, rank_ascending] = df_piv[rank_ascending].rank(method='max', na_option='bottom', ascending=True)

            rank_descending = ['size']
            df_rank.loc[:, rank_descending] = df_piv[rank_descending].rank(method='max', na_option='bottom', ascending=False)

            guess_rankings = dict()

            for desc_index in df_piv.columns.levels[1].to_list()[:-1]:
                df_rank.loc[:, ('rank_sum', desc_index)] = df_rank.loc[:, (rank_ascending + rank_descending, desc_index)].sum(axis=1)
                guesses = df_rank.loc[df_rank.loc[:, ('rank_sum', desc_index)].nsmallest(10).index, 'package_predicted'].to_list()

                for rank, guess in enumerate(guesses, start=1):
                    if guess in guess_rankings.keys():
                        guess_rankings[guess].append(rank)
                    else:
                        guess_rankings[guess] = [rank]
                        
            top_guesses = [s[0] for s in sorted(guess_rankings.items(), key=lambda x: len(x[1]), reverse=True)][:10]
            results.append([sample_actual, package_actual, alg, ''] + top_guesses)
        except:
            results.append([sample_actual, package_actual, alg, str(traceback.format_exc(limit=10))])

    return results


if __name__ == '__main__':
    csv_path = Path('/ndata/chaban/pharmapack/CSVs/lopq/LOPQ:BOTH:resnet50:512:20.csv')
    df_combined = pd.read_csv(csv_path, index_col=None)
#     df_combined = df_combined[(df_combined.algorithm == 'MI1') & (df_combined.distance < 8)]
#     df_combined = df_combined[(df_combined.distance < 10)]
    dfs = [x for _, x in df_combined.groupby('sample_actual')]
    results = p_map(calc_one, dfs, num_cpus=44)
    results_flattened = []
    for result in results:
        for r in result:
            results_flattened.append(r)
#     results = calc_one(dfs[0])
    pd.DataFrame(results_flattened).to_csv(str(csv_path).replace('.csv', ':rank.csv'), index=False)