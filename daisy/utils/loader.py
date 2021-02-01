import os
import gc
import re
import json
import random
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from daisy.utils.data import incorporate_in_ml100k
from scipy.sparse import csr_matrix

from collections import defaultdict
from IPython import embed


def convert_unique_idx(df, col):
    column_dict = {x: i for i, x in enumerate(df[col].unique())}
    df[col] = df[col].apply(column_dict.get)
    assert df[col].min() == 0
    assert df[col].max() == len(column_dict) - 1
    return df


def cut_down_data_half(df):
    cut_df = pd.DataFrame([])
    for u in np.unique(df.user):
        aux = df[df['user'] == u].copy()
        cut_df = cut_df.append(df.sample(int(len(aux) / 2)))
    return cut_df


def filter_users_and_items(df, num_users=None, freq_items=None, top_items=None, keys=['user', 'item']):
    '''
        Reduces the dataframe to a number of users = num_users and it filters the items by frequency
    '''
    if num_users is not None:
        # df = df[df['user_id'].isin(np.unique(df.user_id)[:num_users])]
        df = df[df[keys[0]].isin(np.unique(df[keys[0]])[:num_users])]

    # Get top5k books
    if top_items is not None:
        top5k_books = df[keys[1]].value_counts()[:top_items].index
        df = df[df[keys[1]].isin(top5k_books)]

    if freq_items is not None:
        frequent_items = df['item'].value_counts()[df['item'].value_counts() > freq_items].index
        df = df[df[keys[1]].isin(frequent_items)]

    return df


def run_statistics(df, src):

    path = f'histograms/{src}'
    bins = 30
    os.makedirs(path, exist_ok=True)
    f = open(os.path.join(path, "information.txt"), "w+")
    f.write("Information:\n")
    f.write("==========================\n")
    f.write(f"Interactions: {len(df)}\n")
    f.write(f"#users = {df['user'].nunique()}\n")
    f.write(f"#items = {df['item'].nunique()}\n")
    f.close()

    for key in ['user', 'item']:
        # OPCIÓ A: HISTOGRAMA
        a = pd.DataFrame(df.groupby([key])[key].count())
        a.columns = ['value_counts']
        a.reset_index(level=[0], inplace=True)
        dims = (15, 5)
        fig, ax = plt.subplots(figsize=dims)
        a["value_counts"].hist(bins=200)
        # fig.savefig('hist.jpg')
        fig.savefig(os.path.join(path, f'{src}_histogram_{key}_bins={bins}.png'))
        fig.clf()

        # OPCIÓ : BARPLOT
        # a = pd.DataFrame(df_year.groupby(['user'])['user'].count())
        # a.columns = ['value_counts']
        # a.reset_index(level=[0], inplace=True)
        # dims = (15, 5)
        # fig, ax = plt.subplots(figsize=dims)
        # sns.set_style("darkgrid")
        # sns.barplot(ax=ax, x="user", y="value_counts", data=a, palette="Blues_d")
        # ax.set(xlabel="User", ylabel="Value Counts")
        # plt.xticks(rotation=45)
        # plt.show()
        # fig.savefig('data.jpg')


def load_rate(src='ml-100k', prepro='origin', binary=True, pos_threshold=None, level='ui', context=False,
              gce_flag=False, cut_down_data=False, side_info=False, context_type='', context_as_userfeat=False,
              flag_run_statistics=False, remove_top_users=0, remove_on='item'):
    """
    Method of loading certain raw data
    Parameters
    ----------
    src : str, the name of dataset
    prepro : str, way to pre-process raw data input, expect 'origin', f'{N}core', f'{N}filter', N is integer value
    binary : boolean, whether to transform rating to binary label as CTR or not as Regression
    pos_threshold : float, if not None, treat rating larger than this threshold as positive sample
    level : str, which level to do with f'{N}core' or f'{N}filter' operation (it only works when prepro contains 'core' or 'filter')

    Returns
    -------
    df : pd.DataFrame, rating information with columns: user, item, rating, (options: timestamp)
    user_num : int, the number of users
    item_num : int, the number of items
    """
    df = pd.DataFrame()
    # import mat73
    # a = mat73.loadmat('data/gen-disease/genes_phenes.mat')
    # which dataset will use
    if src == 'ml-100k':
        df = pd.read_csv(f'./data/{src}/u.data', sep='\t', header=None,
                         names=['user', 'item', 'rating', 'timestamp'], engine='python')
        if cut_down_data:
            df = cut_down_data_half(df)  # from 100k to 49.760 interactions

    elif src == 'drugs':
        union = False
        if union == True:
            df = pd.read_csv(f'./data/{src}/train_data_contextUNION_sideeffect.csv', engine='python', index_col=0)
            df.drop(columns=['context'], inplace=True)
            df.rename(columns={'drug': 'user', 'disease': 'item',
                               'context_union': 'context',
                               'proteins': 'item-feat', 'side_effect': 'user-feat'}, inplace=True)
        else:
            df = pd.read_csv(f'./data/{src}/train_data_allcontext_sideeffect.csv', engine='python', index_col=0)
            df.rename(columns={'drug': 'user', 'disease': 'item',
                               # 'proteins_drug': 'user-feat',
                               'proteins': 'item-feat', 'side_effect': 'user-feat'}, inplace=True)
        if context_as_userfeat:
            df = df[['user', 'item', 'user-feat', 'item-feat']]
        else:
            df = df[['user', 'item', 'context', 'user-feat']]

        df['timestamp'] = 1
        df['rating'] = 1
        df['array_context_flag'] = True

    elif src == 'ml-1m':
        df = pd.read_csv(f'./data/{src}/ratings.dat', sep='::', header=None, 
                         names=['user', 'item', 'rating', 'timestamp'], engine='python')
        # only consider rating >=4 for data density
        # df = df.query('rating >= 4').reset_index(drop=True).copy()

    elif src == 'ml-10m':
        df = pd.read_csv(f'./data/{src}/ratings.dat', sep='::', header=None, 
                         names=['user', 'item', 'rating', 'timestamp'], engine='python')
        # df = df.query('rating >= 4').reset_index(drop=True).copy()

    elif src == 'ml-20m':
        df = pd.read_csv(f'./data/{src}/ratings.csv')
        df.rename(columns={'userId': 'user', 'movieId': 'item'}, inplace=True)
        # df = df.query('rating >= 4').reset_index(drop=True)

    elif src == 'books':
        if not os.path.exists(f'./data/{src}/preprocessed_books_complete_timestamp.csv'):
            df = pd.read_csv(f'./data/{src}/preprocessed_books_complete.csv', sep=',', engine='python')
            df.rename(columns={'user_id': 'user', 'book_id': 'item', 'date_added': 'timestamp'}, inplace=True)

            df = convert_unique_idx(df, 'user')
            df = convert_unique_idx(df, 'item')
            df['rating'] = 1.0
            # if type(df['timestamp'][0]) == 'str':
            df['date'] = pd.to_datetime(df['timestamp'])
            df['timestamp'] = pd.to_datetime(df['date'], utc=True)
            df['timestamp'] = df.timestamp.astype('int64') // 10 ** 9
            df.to_csv(f'./data/{src}/preprocessed_books_complete_timestamp.csv', sep=',', index=False)
        else:
            df = pd.read_csv(f'./data/{src}/preprocessed_books_complete_timestamp.csv', sep=',', engine='python')
        del df['date']
        # reduce users to 3000 and filter items by clicked_frequency > 10
        df = filter_users_and_items(df, num_users=4000, freq_items=50, top_items=5000, keys=['user', 'item'])  # 35422 books

    elif src == 'music':
        df = pd.read_csv(f'./data/{src}-context/train.csv')
        if side_info:
            # ['user_id', 'track_id', 'hashtag', 'created_at', 'score', 'lang', 'tweet_lang', 'time_zone',
            # 'instrumentalness', 'liveness', 'speechiness', 'danceability', 'valence', 'loudness', 'tempo',
            # 'acousticness', 'energy', 'mode', 'key', 'rating']
            df.rename(columns={'user_id': 'user', 'track_id': 'item', 'created_at': 'timestamp', 'speechiness': 'side_info'},
                      inplace=True)
            df = df[['user', 'item', 'timestamp', 'side_info']]
            # PREPROCESS SPEECHINESS # VALUE 10 FOR NON EXISTING FEATURE
            df['side_info'] = df['side_info'].round(1)
            df['side_info'] = df['side_info']*10
            df['side_info'] = df['side_info'].fillna(10)
            df['side_info'] = df['side_info'].astype(int)
        else:
            df.rename(columns={'user_id': 'user', 'track_id': 'item', 'created_at': 'timestamp'}, inplace=True)
            df = df[['user', 'item', 'timestamp']]
        # df = df.query('rating >= 4').reset_index(drop=True)
        df = convert_unique_idx(df, 'user')
        df = convert_unique_idx(df, 'item')
        df = filter_users_and_items(df, num_users=3000, freq_items=20, keys=['user', 'item'])  # 18508 songs - 3981 users
        # FILTER USERS WHITH LESS THAN 4 INTERACTIONS
        df_aux = df.groupby('user').count().reset_index()[['user', 'item']]
        indexes = df_aux[df_aux['item'] >= 3]['user'].index
        df = df[df['user'].isin(indexes)]

        df['rating'] = 1.0
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['timestamp'] = df.timestamp.astype('int64') // 10 ** 9
        prepro = 'origin'

    elif src == 'frappe':
        df1 = pd.read_csv(f'./data/{src}/{src}_xin/train.csv', sep=',', header=None)
        df2 = pd.read_csv(f'./data/{src}/{src}_xin/test.csv', sep=',', header=None)
        df = pd.concat([df1, df2])

        df['item'] = df[1].apply(lambda x: x.split('-')[0])
        df['i-feature'] = df[1].apply(lambda x: x.split('-')[1])

        df['user'] = df[0].apply(lambda x: x.split('-')[0])
        df['user-context'] = df[0].apply(lambda x: '-'.join(x.split('-')[1:]))

        # http://web.archive.org/web/20180422190150/http://baltrunas.info/research-menu/frappe
        # columNames = ['labels', 'user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather',
        #               'country', 'city']
        context_type = ['daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city']
        # df = pd.read_csv(f'./data/{src}/{src}.csv', sep=' ', engine='python', names=columNames)
        df = pd.read_csv(f'./data/{src}/{src}.csv', sep='\t')

        # TODO: select one context
        if context:
            # context_type
            df = df[['user', 'item']+context_type]
            for context_aux in context_type:
                df = convert_unique_idx(df, context_aux)
        else:
            df = df[['user', 'item']]
        # treat weight as interaction, as 1
        df['rating'] = 1.0
        df['timestamp'] = 1

        # fake timestamp column
    elif src == 'netflix':
        cnt = 0
        tmp_file = open(f'./data/{src}/training_data.csv', 'w')
        tmp_file.write('user,item,rating,timestamp' + '\n')
        for f in os.listdir(f'./data/{src}/training_set/'):
            cnt += 1
            if cnt % 5000 == 0:
                print(f'Finish Process {cnt} file......')
            txt_file = open(f'./data/{src}/training_set/{f}', 'r')
            contents = txt_file.readlines()
            item = contents[0].strip().split(':')[0]
            for val in contents[1:]:
                user, rating, timestamp = val.strip().split(',')
                tmp_file.write(','.join([user, item, rating, timestamp]) + '\n')
            txt_file.close()

        tmp_file.close()

        df = pd.read_csv(f'./data/{src}/training_data.csv')
        df['rating'] = df.rating.astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    elif src == 'lastfm':

        # if not os.path.exists(f'./data/{src}/1k_dataset.csv'):
        #
        #     df = pd.read_csv(f'./data/{src}-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv',
        #                      sep='\t', names=['user', 'timestamp', 'artist', 'artist_name', 'item', 'item_name'],
        #                      engine='python')
        #     df['rating'] = 1.0
        #     df = df[['user', 'item', 'artist', 'rating', 'timestamp']]
        #     df = filter_users_and_items(df, num_users=1000, freq_items=20, keys=['user', 'item'])  # 35422 books
        #
        #     df.to_csv(f'./data/{src}/1k_dataset.csv', sep=',', index=False)
        #
        # else:
        #     df = pd.read_csv(f'./data/{src}/1k_dataset.csv', sep=',')
        #     df.rename(columns={'artist': 'side_info'}, inplace=True)

        # XIN_XIN: user_artists.dat     1.000 (414) , 20.301 (14.387) , 214.574 (SI)
        # xin_file = f'./data/{src}/dataset.csv'
        # if not os.path.exists(xin_file):
        #     df1 = pd.read_csv(f'./data/{src}/train.csv', sep=',', names=['user', 'item', 'rating', 'timestamp'])
        #     df2 = pd.read_csv(f'./data/{src}/test.csv', sep=',', names=['user', 'item', 'rating', 'timestamp'])
        #     df = pd.concat([df1, df2])
        #
        #     df['user-feat'] = df['user'].apply(lambda x: x.split('-')[1])  #previous item
        #     df['user'] = df['user'].apply(lambda x: x.split('-')[0])
        #     df['item-feat'] = df['item'].apply(lambda x: x.split('-')[1])  #artist
        #     df['item'] = df['item'].apply(lambda x: x.split('-')[0])
        #
        #     df.to_csv(f'./data/{src}/dataset.csv', sep=',', index=False)
        # else:
        #     df = pd.read_csv(f'./data/{src}/dataset.csv', sep=',')

        file_path = f'./data/{src}-dataset-1K/'

        if not os.path.exists(f'{file_path}dataset.csv'):

            df = pd.read_csv(f'{file_path}userid-timestamp-artid-artname-traid-traname.tsv', sep='\t',
                             names=['userid', 'timestamp', 'artist-id', 'artist-name', 'track-id', 'track-name'],
                             error_bad_lines=False)
            df['rating'] = 1

            df.rename(columns={'userid': 'user', 'track-id': 'item', 'artist-id': 'item-feat'}, inplace=True)
            df = df.dropna(subset=['user', 'item'])

            df = convert_unique_idx(df, 'user')
            df = convert_unique_idx(df, 'item')
            df = convert_unique_idx(df, 'item-feat')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = filter_users_and_items(df, num_users=1000, freq_items=20, keys=['user', 'item'])
            df.to_csv(f'{file_path}/dataset.csv', sep=',', index=False)
        else:
            df = pd.read_csv(f'{file_path}/dataset.csv', sep=',')
            selected_users = df['user'].value_counts()[df['user'].value_counts() > 20].index
            df = df[df['user'].isin(selected_users)]
            year = 2009
            if not os.path.exists(f'{file_path}/dataset_{year}.csv'):
                # TODO: TAKE SPECIFIC YEAR
                df_year = pd.DataFrame([])
                for user in range(np.max(df['user'])):
                    u = df[df['user'] == user].sort_values(by='timestamp', ascending=False)
                    try:
                        u['timestamp'] = pd.to_datetime(u['timestamp'], unit='ns')
                    except:
                        u['timestamp'] = pd.to_datetime(u['timestamp'], unit='s')

                    include = u[u['timestamp'].dt.year == year]
                    df_year = df_year.append(include)

                selected_items = df_year['item'].value_counts()[df_year['item'].value_counts() > 20].index
                df_year = df_year[df_year['item'].isin(selected_items)]

                selected_users = df_year['user'].value_counts()[df_year['user'].value_counts() > 20].index
                df_year = df_year[df_year['user'].isin(selected_users)]
                df_year.to_csv(f'{file_path}/dataset_{year}.csv', sep=',', index=False)
                df = df_year.copy()

                '''
                year --> 2009
                user            774
                item           4985
                item-feat       794
                rating            1
                timestamp    412008
                '''

            else:
                print('LOADED POST-PROCESSED DB')
                df = pd.read_csv(f'{file_path}/dataset_2009.csv', sep=',')

            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df['timestamp'] = df.timestamp.astype('int64') // 10 ** 9

            selected_users = df['user'].value_counts()[df['user'].value_counts() < 9000].index
            df = df[df['user'].isin(selected_users)]
            df = df[['user', 'item', 'item-feat', 'rating', 'timestamp']]
            df = filter_users_and_items(df, num_users=1000, freq_items=20, top_items=5000, keys=['user', 'item'])  # 35422 books

    elif src == 'bx':
        df = pd.read_csv(f'./data/{src}/BX-Book-Ratings.csv', delimiter=";", encoding="latin1")
        df.rename(columns={'User-ID': 'user', 'ISBN': 'item', 'Book-Rating': 'rating'}, inplace=True)
        # fake timestamp column
        df['timestamp'] = 1

    elif src == 'pinterest':
        # TODO this dataset has wrong source URL, we will figure out in future
        pass

    elif src == 'amazon-cloth':
        df = pd.read_csv(f'./data/{src}/ratings_Clothing_Shoes_and_Jewelry.csv', 
                         names=['user', 'item', 'rating', 'timestamp'])

    elif src == 'amazon-electronic':
        df = pd.read_csv(f'./data/{src}/ratings_Electronics.csv', 
                         names=['user', 'item', 'rating', 'timestamp'])

    elif src == 'amazon-book':
        df = pd.read_csv(f'./data/{src}/ratings_Books.csv', 
                         names=['user', 'item', 'rating', 'timestamp'], low_memory=False)
        df = df[df['timestamp'].str.isnumeric()].copy()
        df['timestamp'] = df['timestamp'].astype(int)

    elif src == 'amazon-music':
        df = pd.read_csv(f'./data/{src}/ratings_Digital_Music.csv', 
                         names=['user', 'item', 'rating', 'timestamp'])

    elif src == 'epinions':
        d = sio.loadmat(f'./data/{src}/rating_with_timestamp.mat')
        prime = []
        for val in d['rating_with_timestamp']:
            user, item, rating, timestamp = val[0], val[1], val[3], val[5]
            prime.append([user, item, rating, timestamp])
        df = pd.DataFrame(prime, columns=['user', 'item', 'rating', 'timestamp'])
        del prime
        df = convert_unique_idx(df, 'user')
        df = convert_unique_idx(df, 'item')
        gc.collect()

    elif src == 'yelp':
        json_file_path = f'./data/{src}/yelp_academic_dataset_review.json'
        prime = []
        for line in open(json_file_path, 'r', encoding='UTF-8'):
            val = json.loads(line)
            prime.append([val['user_id'], val['business_id'], val['stars'], val['date']])
        df = pd.DataFrame(prime, columns=['user', 'item', 'rating', 'timestamp'])
        # df['timestamp'] = pd.to_datetime(df.timestamp)
        df['timestamp'] = pd.to_datetime(df.timestamp).astype(int)
        del prime
        gc.collect()

    elif src == 'citeulike':
        user = 0
        dt = []
        for line in open(f'./data/{src}/users.dat', 'r'):
            val = line.split()
            for item in val:
                dt.append([user, item])
            user += 1
        df = pd.DataFrame(dt, columns=['user', 'item'])
        # fake timestamp column
        df['timestamp'] = 1

    else:
        raise ValueError('Invalid Dataset Error')

    # set rating >= threshold as positive samples
    if pos_threshold is not None:
        df = df.query(f'rating >= {pos_threshold}').reset_index(drop=True)

    # reset rating to interaction, here just treat all rating as 1
    if binary:
        df['rating'] = 1.0

    # which type of pre-dataset will use
    if prepro == 'origin':
        pass
        if remove_top_users > 0:
            if remove_on == 'user':
                unknown_item = df['item'].nunique()
                a = df['user'].value_counts()
                n = int(len(a) * (remove_top_users / 100))
                users_removed = a.head(n).index
                a.drop(a.head(n).index, inplace=True)
                new_df = df[df['user'].isin(a.index)]

                for u in users_removed:
                    aux = df[df['user'] == u][:1].copy()
                    # aux['item'] = unknown_item
                    new_df = new_df.append(aux, ignore_index=True)

                # IDEA: generate fake interactions with items that disapeared
                # def do(df, new_df):
                #     return [item for item in df['item'].unique() if not item in new_df['item'].unique()]
                missing_items = [item for item in df['item'].unique() if not item in new_df['item'].unique()]
                for i in missing_items:
                    aux = df[df['item'] == i][:1].copy()
                    # aux['user'] = 1
                    new_df = new_df.append(aux, ignore_index=True)
                df = new_df.copy()

            elif remove_on == 'item':
                unknown_user = 1
                a = df['item'].value_counts()
                n = int(len(a) * (remove_top_users / 100))
                items_removed = a.head(n).index
                a.drop(a.head(n).index, inplace=True)
                new_df = df[df['item'].isin(a.index)]

                for i in items_removed:
                    aux = df[df['item'] == i][:1].copy()
                    # aux['user'] = unknown_user
                    new_df = new_df.append(aux, ignore_index=True)
                df = new_df.copy()

            else:
                pass

    elif prepro.endswith('filter'):
        pattern = re.compile(r'\d+')
        filter_num = int(pattern.findall(prepro)[0])

        tmp1 = df.groupby(['user'], as_index=False)['item'].count()
        tmp1.rename(columns={'item': 'cnt_item'}, inplace=True)
        tmp2 = df.groupby(['item'], as_index=False)['user'].count()
        tmp2.rename(columns={'user': 'cnt_user'}, inplace=True)
        df = df.merge(tmp1, on=['user']).merge(tmp2, on=['item'])
        if level == 'ui':    
            df = df.query(f'cnt_item >= {filter_num} and cnt_user >= {filter_num}').reset_index(drop=True).copy()
        elif level == 'u':
            df = df.query(f'cnt_item >= {filter_num}').reset_index(drop=True).copy()
        elif level == 'i':
            df = df.query(f'cnt_user >= {filter_num}').reset_index(drop=True).copy()        
        else:
            raise ValueError(f'Invalid level value: {level}')

        df.drop(['cnt_item', 'cnt_user'], axis=1, inplace=True)
        del tmp1, tmp2
        gc.collect()

    elif prepro.endswith('core'):
        pattern = re.compile(r'\d+')
        core_num = int(pattern.findall(prepro)[0])

        def filter_user(df):
            tmp = df.groupby(['user'], as_index=False)['item'].count()
            tmp.rename(columns={'item': 'cnt_item'}, inplace=True)
            df = df.merge(tmp, on=['user'])
            df = df.query(f'cnt_item >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_item'], axis=1, inplace=True)

            return df

        def filter_item(df):
            tmp = df.groupby(['item'], as_index=False)['user'].count()
            tmp.rename(columns={'user': 'cnt_user'}, inplace=True)
            df = df.merge(tmp, on=['item'])
            df = df.query(f'cnt_user >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_user'], axis=1, inplace=True)

            return df

        if level == 'ui':
            while 1:
                df = filter_user(df)
                df = filter_item(df)
                chk_u = df.groupby('user')['item'].count()
                chk_i = df.groupby('item')['user'].count()
                if len(chk_i[chk_i < core_num]) <= 0 and len(chk_u[chk_u < core_num]) <= 0:
                    break
        elif level == 'u':
            df = filter_user(df)
        elif level == 'i':
            df = filter_item(df)
        else:
            raise ValueError(f'Invalid level value: {level}')

        gc.collect()

    else:
        raise ValueError('Invalid dataset preprocess type, origin/Ncore/Nfilter (N is int number) expected')
    # encoding user_id and item_id
    unique_original_items = np.unique(df['item'])

    df['user'] = pd.Categorical(df['user']).codes
    df['item'] = pd.Categorical(df['item']).codes

    if 'user-feat' in df.columns:
        df['user-feat'] = pd.Categorical(df['user-feat']).codes
    if 'item-feat' in df.columns:
        df['item-feat'] = pd.Categorical(df['item-feat']).codes

    user_num = df['user'].nunique()
    item_num = df['item'].nunique()
    if flag_run_statistics:
        run_statistics(df, src)
        exit()
    # ####################################################################
    # if side_info and src == 'ml-100k':
    #     si = pd.read_csv(f'./data/{src}/side-information.csv', index_col=0)
    #     si.rename(columns={'id': 'item', 'genres': 'side_info'}, inplace=True)
    #     si = si[['item', 'side_info']]
    #     # if df['item'].min() > 0:  # Reindex items
    #     si_extension = incorporate_in_ml100k(si, X.shape[0], unique_original_items, user_num)
    #     # TODO: INCORPORATE si_extension to X
    #     X_gender_mx = csr_matrix(si_extension.values)
    ####################################################################

    print(f'Finish loading [{src}]-[{prepro}] dataset with [context == {context}] and [GCE == {gce_flag}]')
    # embed()
    # print(df.nunique())
    return df, user_num, item_num, unique_original_items


def add_last_clicked_item_context(df, dataset='', random_context=False):
    df['context'] = df[df.columns[2]] if dataset == 'frappe' else df['rating']
    timestamp_flag = False if dataset == 'frappe' else True
    df = df[['user', 'item', 'context', 'rating', 'timestamp']]
    data = df.to_numpy().astype(int)
    assert data[:, 1].min() == data[:, 0].max() + 1
    # let space for film UNKNOWN  (one more index than the last film)
    empty_film_idx = data[:, 1].max() + 1
    assert data[:, 1].max() + 1 == empty_film_idx
    if random_context:
        sorted_data = data.copy()
        # add 2 to max dimension to take into account the last number of range + empty_film_idx
        aux = np.random.randint(df['item'].min(), df['item'].max() + 2, size=len(data[:, 2]))
        sorted_data[:, 2] = aux
    else:
        sorted_data = data[data[:, -1].argsort()]

        if not timestamp_flag:
            data[:, 2] = data[:, 2] + (empty_film_idx + 1)
            sorted_data = data.copy()
        else:
            for u in tqdm(np.unique(sorted_data[:, 0]), desc="mapping context"):
                aux = sorted_data[sorted_data[:, 0] == u]
                # if timestamp_flag:
                aux[:, 2] = np.insert(aux[:-1][:, 1], 0, empty_film_idx)
                sorted_data[sorted_data[:, 0] == u] = aux

    # # user_num == first item number
    # sorted_data[:, 2] = np.concatenate(([user_num], sorted_data[:-1][:, 1]))
    new_df = pd.DataFrame(data=sorted_data, columns=list(df.columns))
    return new_df


def get_ur(df, context=False, eval=False):
    """
    Method of getting user-rating pairs
    Parameters
    ----------
    df : pd.DataFrame, rating dataframe

    Returns
    -------
    ur : dict, dictionary stored user-items interactions
    """
    ur = defaultdict(set)
    for _, row in df.iterrows():
        if context and not eval:
            if 'array_context_flag' in df.columns:
                lst = np.append(int(row['user']), row['context']).tolist()
                ur[tuple(lst)].add(int(row['item']))
            else:
                ur[int(row['user']), int(row['context'])].add(int(row['item']))
        else:
            ur[int(row['user'])].add(int(row['item']))
    return ur


def get_ir(df):
    """
    Method of getting item-rating pairs
    Parameters
    ----------
    df : pd.DataFrame, rating dataframe

    Returns
    -------
    ir : dict, dictionary stored item-users interactions
    """
    ir = defaultdict(set)
    for _, row in df.iterrows():
        ir[int(row['item'])].add(int(row['user']))

    return ir


def build_feat_idx_dict(df:pd.DataFrame, 
                        cat_cols:list=['user', 'item'], 
                        num_cols:list=[]):
    """
    Method of encoding features mapping for FM
    Parameters
    ----------
    df : pd.DataFrame feature dataframe
    cat_cols : List, list of categorical column names
    num_cols : List, list of numeric column names

    Returns
    -------
    feat_idx_dict : Dictionary, dict with index-feature column mapping information
    cnt : int, the number of features
    """
    feat_idx_dict = {}
    idx = 0
    for col in cat_cols:
        feat_idx_dict[col] = idx
        idx = idx + df[col].max() + 1
    for col in num_cols:
        feat_idx_dict[col] = idx
        idx += 1
    print('Finish build feature index dictionary......')

    cnt = 0
    for col in cat_cols:
        for _ in df[col].unique():
            cnt += 1
    for _ in num_cols:
        cnt += 1
    print(f'Number of features: {cnt}')

    return feat_idx_dict, cnt


def convert_npy_mat(user_num, item_num, df):
    """
    method of convert dataframe to numoy matrix
    Parameters
    ----------
    user_num : int, the number of users
    item_num : int, the number of items
    df :  pd.DataFrame, rating dataframe

    Returns
    -------
    mat : np.matrix, rating matrix
    """
    mat = np.zeros((user_num, item_num))
    for _, row in df.iterrows():
        u, i, r = row['user'], row['item'], row['rating']
        mat[int(u), int(i)] = float(r)
    return mat


def build_candidates_set(test_ur, train_ur, item_pool, candidates_num=1000, context_flag=False):
    """
    method of building candidate items for ranking
    Parameters
    ----------
    test_ur : dict, ground_truth that represents the relationship of user and item in the test set
    train_ur : dict, this represents the relationship of user and item in the train set
    item_pool : the set of all items
    candidates_num : int, the number of candidates
    Returns
    -------
    test_ucands : dict, dictionary storing candidates for each user in test set
    """
    test_ucands = defaultdict(list)
    for k, v in test_ur.items():
        sample_num = candidates_num - len(v) if len(v) < candidates_num else 0
        if context_flag:
            user = k[0]
            # context = k[1:]
            # if isinstance(context, list):
            #     lst = np.append(int(row['user']), row['context']).tolist()
            #     tup = tuple(lst)
            # else:
            #     tup = (user, context)

            sub_item_pool = item_pool - v - train_ur[user]  # remove GT & interacted 
            sample_num = min(len(sub_item_pool), sample_num)
            if sample_num == 0:
                samples = random.sample(v, candidates_num)
                # test_ucands[(user, context)] = list(set(samples))
                test_ucands[k] = list(set(samples))
            else:
                samples = random.sample(sub_item_pool, sample_num)
                test_ucands[k] = list(v | set(samples))
        else:
            sub_item_pool = item_pool - v - train_ur[k]  # remove GT & interacted (with same context)
            sample_num = min(len(sub_item_pool), sample_num)
            if sample_num == 0:
                samples = random.sample(v, candidates_num)
                test_ucands[k] = list(set(samples))
            else:
                samples = random.sample(sub_item_pool, sample_num)
                test_ucands[k] = list(v | set(samples))
            
    return test_ucands
