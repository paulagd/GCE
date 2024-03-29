import numpy as np
import pandas as pd
import time, torch, os
from tqdm import tqdm
from daisy.utils.metrics import precision_at_k, recall_at_k, off_policy_at_k, hr_at_k, ndcg_at_k, mrr_at_k
from sklearn.model_selection import KFold, train_test_split, GroupShuffleSplit
from IPython import embed
from collections import defaultdict


def get_weight_file(args, all_files):
    params = f'lr={args.lr}_DO={int(args.dropout)}'
    params1 = f'bs={args.batch_size}'
    match_file = [file for file in all_files if file.endswith('__not_early_stopping.pkl') and
                  params in file and params1 in file]
    assert len(match_file) > 0
    return match_file[0]


def perform_evaluation(loaders, candidates, model, args, device, test_ur, s_time=None, writer=None, epoch=None,
                       minutes_train=None, seconds_train=None, tune=False, populary_dict=None, desc=None):
    if args.remove_top_users > 0 and not desc is None:
        # LOAD WEIGHTS TO MODEL
        filename_weights = f'weights/{args.dataset}/best_weights/{args.algo_name}'
        all_files = os.listdir(filename_weights)
        match_file = get_weight_file(args, all_files)
        # todo: build str with  bs, do, lr and model
        best_weight_path = os.path.join(filename_weights, match_file)
        checkpoint = torch.load(best_weight_path)
        # aux = 'best_epoch=6_mf_RANK_ALL_lr=0.0005_DO=0_bs=512_reindexed_UIC__100epochs__not_early_stopping.pkl'
        # checkpoint = torch.load(os.path.join(filename_weights,aux))

        model.load_state_dict(checkpoint["state_dict"])
        # checkpoint["state_dict"]['embeddings.weight'] = checkpoint["state_dict"]['embeddings.weight']
        # checkpoint["state_dict"]['embeddings.weight'] = checkpoint["state_dict"]['embeddings.weight']
        # checkpoint["state_dict"]['embeddings.weight'] = checkpoint["state_dict"]['embeddings.weight']
        model.to(device)
        print('LOADED BEST MODEL')

        if args.remove_on == 'user':
            # remove users candidates --> can
            keys = list(candidates.keys())
            user_keys = [k[0] for k in keys]
            idx_to_take = [i for i, u in enumerate(user_keys) if u not in populary_dict]
            new_keys = [keys[idx] for idx in idx_to_take]

            aux_cand = defaultdict(list)
            for tup in new_keys:
                aux_cand[tup] = list(candidates[tup])

            # candidates = [candidates[k] for k in new_keys]
            candidates = aux_cand

    model.eval()
    preds = {}
    for u in tqdm(candidates.keys(), disable=tune):
        # get top-N list with torch method
        for items in loaders[u]:
            # remove items[1] in k_popular_items --> remove tambe x items[0] i x items[2]
            user_u, item_i, context = items[0], items[1], items[2]
            if args.remove_top_users > 0 and args.remove_on == 'item':
                item_i = [item.item() for item in item_i if item not in populary_dict]
                item_i = torch.LongTensor(item_i)
                user_u = user_u[:len(item_i)]
                context = context[:len(item_i)]
            item_i = item_i.to(device)
            user_u = user_u.to(device)
            if args.context:
                if isinstance(context, list):
                    if len(context) > 1:
                        context = torch.stack(context, dim=1)
                        context = [torch.LongTensor(c).to(device) for c in context]
                    else:
                        # context = torch.LongTensor(context).to(device)
                        context = context[0].to(device)
                        # context = context[:len(item_i)]
                else:
                    context = context.to(device)
            else:
                context = None

            prediction = model.predict(user_u, item_i, context)
            _, indices = torch.topk(prediction, args.topk)
            top_n = torch.take(torch.tensor(candidates[u]), indices).cpu().numpy()

        preds[u] = top_n

    # convert rank list to binary-interaction
    rank_list_items = preds.copy()
    for u in preds.keys():
        preds[u] = [1 if i in test_ur[u] else 0 for i in preds[u]]

    for u in rank_list_items.keys():
        rank_list_items[u] = [i if i in test_ur[u] else 0 for i in rank_list_items[u]]

    # res = pd.DataFrame({'metric@K': ['pre', 'rec', 'hr', 'map', 'mrr', 'ndcg']})
    res = pd.DataFrame({'metric@K': ['hr', 'ndcg']})
    tmp_pred_10 = []
    for k in [10, 20, 30, 40, 50]:
        if k > args.topk:
            continue
        tmp_preds = preds.copy()
        tmp_preds = {key: rank_list[:k] for key, rank_list in tmp_preds.items()}
        rank_list_items = {key: rank_list[:k] for key, rank_list in rank_list_items.items()}
        # pre_k = np.mean([precision_at_k(r, k) for r in tmp_preds.values()])
        # rec_k = recall_at_k(tmp_preds, test_ur, k)
        hr_k = hr_at_k(tmp_preds, test_ur)
        # map_k = map_at_k(tmp_preds.values())
        # mrr_k = mrr_at_k(tmp_preds, k)
        # if populary_dict and not tune:
        #     embed()
        #     off_policy_k, off_policy_k_norm = off_policy_at_k(populary_dict, rank_list_items)
        # else:
        #     populary_dict = np.nan
        ndcg_k = np.mean([ndcg_at_k(r, k) for r in tmp_preds.values()])

        if (writer and not epoch is None) and not tune:
            writer.add_scalar(f'metrics/HR_@{k}', hr_k, epoch+1)
            writer.add_scalar(f'metrics/NDCG_@{k}', ndcg_k, epoch+1)
            if args.printall and epoch > 0:
                print(f'{epoch}\t@{k}\t{hr_k:.4f}\t{ndcg_k:.4f}')
            # writer.add_scalar(f'metrics/Discounted_HR@{k}', off_policy_k, epoch+1)
            # print(f'HR@{k}: {hr_k:.4f}  |  NDCG@{k}: {ndcg_k:.4f}')

        # res[k] = np.array([pre_k, rec_k, hr_k, map_k, mrr_k, ndcg_k])
        res[k] = np.array([hr_k, ndcg_k])
        if k == 10:
            tmp_pred_10 = np.array([hr_k, ndcg_k])
        if not (writer and not epoch is None) and not tune and not args.printall:
            if k == 10:
                print('--------------TEST METRICS ------------')
                print('+'*80)
                print('+'*80)
            # print(f'Precision@{k}: {pre_k:.4f}')
            # print(f'Recall@{k}: {rec_k:.4f}')
            print(f'HR@{k}: {hr_k:.4f}')
            # print(f'MAP@{k}: {map_k:.4f}')
            # print(f'MRR@{k}: {mrr_k:.4f}')
            print(f'NDCG@{k}: {ndcg_k:.4f}')
            # print(f'Discounted__HR@{k}: {off_policy_k}')

    if not (writer and not epoch is None) and not tune:
        print(f'TRAINING ELAPSED TIME: {minutes_train:.2f} min, {seconds_train:.4f}seconds')

        elapsed_time_total = time.time() - s_time
        hours, rem = divmod(elapsed_time_total, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'TOTAL ELAPSED TIME: {minutes:.2f} min, {seconds:.4f}seconds')

    return res, writer, tmp_pred_10
        

def split_test(df, test_method='fo', test_size=.2):
    """
    method of splitting data into training data and test data
    Parameters
    ----------
    df : pd.DataFrame raw data waiting for test set splitting
    test_method : str, way to split test set
                    'fo': split by ratio
                    'tfo': split by ratio with timestamp
                    'tloo': leave one out with timestamp
                    'loo': leave one out
                    'ufo': split by ratio in user level
                    'utfo': time-aware split by ratio in user level
    test_size : float, size of test set

    Returns
    -------
    train_set : pd.DataFrame training dataset
    test_set : pd.DataFrame test dataset

    """

    train_set, test_set = pd.DataFrame(), pd.DataFrame()
    if test_method == 'ufo':
        driver_ids = df['user']
        _, driver_indices = np.unique(np.array(driver_ids), return_inverse=True)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=2020)
        for train_idx, test_idx in gss.split(df, groups=driver_indices):
            train_set, test_set = df.loc[train_idx, :].copy(), df.loc[test_idx, :].copy()

    elif test_method == 'utfo':
        df = df.sort_values(['user', 'timestamp']).reset_index(drop=True)
        def time_split(grp):
            start_idx = grp.index[0]
            split_len = int(np.ceil(len(grp) * (1 - test_size)))
            split_idx = start_idx + split_len
            end_idx = grp.index[-1]

            return list(range(split_idx, end_idx + 1))

        test_index = df.groupby('user').apply(time_split).explode().values
        test_set = df.loc[test_index, :]
        train_set = df[~df.index.isin(test_index)]

    elif test_method == 'tfo':
        # df = df.sample(frac=1)
        df = df.sort_values(['timestamp']).reset_index(drop=True)
        split_idx = int(np.ceil(len(df) * (1 - test_size)))
        train_set, test_set = df.iloc[:split_idx, :].copy(), df.iloc[split_idx:, :].copy()

    elif test_method == 'fo':
        train_set, test_set = train_test_split(df, test_size=test_size, random_state=2019)

    elif test_method == 'tloo':
        # df = df.sample(frac=1)
        df = df.sort_values(['timestamp']).reset_index(drop=True)
        df['rank_latest'] = df.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
        train_set, test_set = df[df['rank_latest'] > 1].copy(), df[df['rank_latest'] == 1].copy()
        del train_set['rank_latest'], test_set['rank_latest']

    elif test_method == 'loo':
        # # slow method
        # test_set = df.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
        # test_key = test_set[['user', 'item']].copy()
        # train_set = df.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(test_key)).reset_index().copy()

        # # quick method
        test_index = df.groupby(['user']).apply(lambda grp: np.random.choice(grp.index))
        test_set = df.loc[test_index, :].copy()
        train_set = df[~df.index.isin(test_index)].copy()

    else:
        raise ValueError('Invalid data_split value, expect: loo, fo, tloo, tfo')

    train_set, test_set = train_set.reset_index(drop=True), test_set.reset_index(drop=True)

    return train_set, test_set


def split_validation(train_set, val_method='fo', fold_num=1, val_size=.1, list_output=True):
    """
    method of split data into training data and validation data.
    (Currently, this method returns list of train & validation set, but I'll change 
    it to index list or generator in future so as to save memory space) TODO

    Parameters
    ----------
    train_set : pd.DataFrame train set waiting for split validation
    val_method : str, way to split validation
                    'cv': combine with fold_num => fold_num-CV
                    'fo': combine with fold_num & val_size => fold_num-Split by ratio(9:1)
                    'tfo': Split by ratio with timestamp, combine with val_size => 1-Split by ratio(9:1)
                    'tloo': Leave one out with timestamp => 1-Leave one out
                    'loo': combine with fold_num => fold_num-Leave one out
                    'ufo': split by ratio in user level with K-fold
                    'utfo': time-aware split by ratio in user level
    fold_num : int, the number of folder need to be validated, only work when val_method is 'cv', 'loo', or 'fo'
    val_size: float, the size of validation dataset

    Returns
    -------
    train_set_list : List, list of generated training datasets
    val_set_list : List, list of generated validation datasets
    cnt : cnt: int, the number of train-validation pair

    """
    if val_method in ['tloo', 'tfo', 'utfo']:
        cnt = 1
    elif val_method in ['cv', 'loo', 'fo', 'ufo']:
        cnt = fold_num
    else:
        raise ValueError('Invalid val_method value, expect: cv, loo, tloo, tfo')
    if list_output:
        train_set_list, val_set_list = [], []
    else:
        train_set_list, val_set_list = pd.DataFrame(), pd.DataFrame()

    if val_method == 'ufo':
        driver_ids = train_set['user']
        _, driver_indices = np.unique(np.array(driver_ids), return_inverse=True)
        gss = GroupShuffleSplit(n_splits=fold_num, test_size=val_size, random_state=2020)
        for train_idx, val_idx in gss.split(train_set, groups=driver_indices):
            train_set_list.append(train_set.loc[train_idx, :])
            val_set_list.append(train_set.loc[val_idx, :])
    if val_method == 'utfo':
        train_set = train_set.sort_values(['user', 'timestamp']).reset_index(drop=True)
        def time_split(grp):
            start_idx = grp.index[0]
            split_len = int(np.ceil(len(grp) * (1 - val_size)))
            split_idx = start_idx + split_len
            end_idx = grp.index[-1]

            return list(range(split_idx, end_idx + 1))
        val_index = train_set.groupby('user').apply(time_split).explode().values
        val_set = train_set.loc[val_index, :]
        train_set = train_set[~train_set.index.isin(val_index)]
        train_set_list.append(train_set)
        val_set_list.append(val_set)
    if val_method == 'cv':
        kf = KFold(n_splits=fold_num, shuffle=False, random_state=2019)
        for train_index, val_index in kf.split(train_set):
            train_set_list.append(train_set.loc[train_index, :])
            val_set_list.append(train_set.loc[val_index, :])
    if val_method == 'fo':
        for _ in range(fold_num):
            train, validation = train_test_split(train_set, test_size=val_size)
            train_set_list.append(train)
            val_set_list.append(validation)
    elif val_method == 'tfo':
        # train_set = train_set.sample(frac=1)
        train_set = train_set.sort_values(['timestamp']).reset_index(drop=True)
        split_idx = int(np.ceil(len(train_set) * (1 - val_size)))
        train_set_list.append(train_set.iloc[:split_idx, :])
        val_set_list.append(train_set.iloc[split_idx:, :])
    elif val_method == 'loo':
        for _ in range(fold_num):
            val_index = train_set.groupby(['user']).apply(lambda grp: np.random.choice(grp.index))
            val_set = train_set.loc[val_index, :].reset_index(drop=True).copy()
            sub_train_set = train_set[~train_set.index.isin(val_index)].reset_index(drop=True).copy()

            train_set_list.append(sub_train_set)
            val_set_list.append(val_set)
    elif val_method == 'tloo':
        # train_set = train_set.sample(frac=1)
        train_set = train_set.sort_values(['timestamp']).reset_index(drop=True)

        train_set['rank_latest'] = train_set.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
        new_train_set = train_set[train_set['rank_latest'] > 1].copy()
        val_set = train_set[train_set['rank_latest'] == 1].copy()
        del new_train_set['rank_latest'], val_set['rank_latest']

        if list_output:
            train_set_list.append(new_train_set)
            val_set_list.append(val_set)
        else:
            train_set_list = new_train_set
            val_set_list = val_set

    return train_set_list, val_set_list, cnt


