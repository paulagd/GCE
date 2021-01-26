import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import ast

from IPython import embed


if __name__ == '__main__':

    data_path = f'./data/drugs/'

    df = pd.read_csv(f'{data_path}train_data_allcontext_sideeffect.csv', index_col=0)
    embed()
    df.groupby('user')['activity_count'].agg('count')

    ax = df['user'].hist()
    fig = ax.get_figure()
    fig.savefig('/path/to/figure.pdf')

        


