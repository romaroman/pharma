import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import mode
from tqdm.notebook import tqdm
from p_tqdm import p_map


fill_value = 100
def calc_one(path):
    actual = path.stem[4:13]

    df = pd.read_csv(path, index_col=None)
    df = df[df.score < 5]
    
    if df.shape[0] == 0 or df.shape[1] == 0:
        return [path.stem, actual, "", False] 
    
    df_piv = df.pivot_table(index='id_predicted', columns='index_actual', values='score', fill_value=fill_value, aggfunc=lambda x: [s for s in x])

    scores = df_piv.to_numpy().T
    
    d = dict()
    for i, score in enumerate(scores, start=1):
        score = [s if s != fill_value else [fill_value] for s in score]
        metrics = [
            (si, len(x), np.mean(x), sum(x), np.median(x)) for si, x in enumerate(score)
        ]
        metric_len = max(metrics, key=lambda x: x[1])[0]
        metric_mean = min(metrics, key=lambda x: x[2])[0]
        metric_sum = min(metrics, key=lambda x: x[3])[0]
        metric_median = min(metrics, key=lambda x: x[4])[0]

        m = mode([metric_len, metric_len, metric_mean, metric_sum, metric_median]).mode[0]
        if m in d.keys():
            d[m] +=1
        else:
            d[m] = 1
    predicted = df_piv.index[max(d, key=d.get)]
    
    return [path.stem, actual, predicted, actual == predicted] 


if __name__ == '__main__':
    items = list(Path('/home/chaban/pharmapack-recognition/separate/lopq/').glob('**/*.csv'))
    res = p_map(calc_one, items, num_cpus=44)
    pd.DataFrame(res).to_csv('accuracy_lopq_fs5.csv')