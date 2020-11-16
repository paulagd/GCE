import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.utils.data as data

from daisy.utils.sampler import Sampler
from daisy.utils.parser import parse_args
from daisy.utils.splitter import split_test
from daisy.utils.data import PointData, PairData, UAEData, sparse_mx_to_torch_sparse_tensor
from daisy.utils.loader import load_rate, get_ur, convert_npy_mat, build_candidates_set, add_last_clicked_item_context
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, ndcg_at_k, mrr_at_k

from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import identity
from IPython import embed


if __name__ == '__main__':
    ''' all parameter part '''
    args = parse_args()

    if args.dataset == 'epinions':
        args.lr = 0.001

    # FIX SEED AND SELECT DEVICE
    seed = 1234
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = "cuda"
    else:
        device = "cpu"

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)

    # store running time in time_log file
    time_log = open('time_log.txt', 'a') 
    
    ''' Test Process for Metrics Exporting '''
    df, users, items = load_rate(args.dataset, args.prepro, binary=True, context=args.context)
    if args.reindex:
        df = df.astype(np.int64)
        df['item'] = df['item'] + users
        if args.context:
            df = add_last_clicked_item_context(df, users)
            df['context'] = df['context'] + items

            # check last number is positive
            assert df['item'].tail().values[-1] > 0

            # TODO: SHOULD I REINDEX THE CONTEXT?
            # np.max(data, axis=0) == array([      942,      2094,      2094,         1, 893286638])
            # users, items = np.max(data, axis=0)[:2]

    train_set, test_set = split_test(df, args.test_method, args.test_size)
    # temporary used for tuning test result
    # train_set = pd.read_csv(f'./experiment_data/train_{args.dataset}_{args.prepro}_{args.test_method}.dat')
    # test_set = pd.read_csv(f'./experiment_data/test_{args.dataset}_{args.prepro}_{args.test_method}.dat')
    df = pd.concat([train_set, test_set], ignore_index=True)

    # user_num = df['user'].nunique()
    # item_num = df['item'].nunique()
    dims = np.max(df.to_numpy().astype(int), axis=0) + 1
    if args.dataset in ['yelp']:
        train_set['timestamp'] = pd.to_datetime(train_set['timestamp'], unit='ns')
        test_set['timestamp'] = pd.to_datetime(test_set['timestamp'], unit='ns')

    # get ground truth
    test_ur = get_ur(test_set, context=args.context)
    total_train_ur = get_ur(train_set, context=args.context)
    # initial candidate item pool
    item_pool = set(range(dims[0], dims[1])) if args.reindex else set(range(dims[1]))
    candidates_num = args.cand_num

    print('='*50, '\n')
    # retrain model by the whole train set
    # format training data
    sampler = Sampler(
        dims,
        num_ng=args.num_ng, 
        sample_method=args.sample_method, 
        sample_ratio=args.sample_ratio,
        reindex=args.reindex
    )
    neg_set, adj_mx = sampler.transform(train_set, is_training=True, context=args.context)
    if args.gce:
        X = sparse_mx_to_torch_sparse_tensor(identity(adj_mx.shape[0])).to(device)
        # We retrieve the graph's edges and send both them and graph to device in the next two lines
        edge_idx, edge_attr = from_scipy_sparse_matrix(adj_mx)
        edge_idx = edge_idx.to(device)

    if args.algo_name in ['cdae', 'vae']:
        train_dataset = UAEData(dims[0], dims[1], train_set, test_set)
        training_mat = convert_npy_mat(dims[0], dims[1], train_set)
    else:
        if args.problem_type == 'pair':
            train_dataset = PairData(neg_set, is_training=True)
        else:
            train_dataset = PointData(neg_set, is_training=True, context=args.context)

    # if args.algo_name == 'mostpop':
    #     from daisy.model.PopRecommender import MostPop
    #     model = MostPop(n=100)
    if args.problem_type == 'point':
        user_num = dims[0]
        max_dim = dims[2] if args.context else dims[1]

        if args.algo_name == 'mf':
            from daisy.model.point.MFRecommender import PointMF
            model = PointMF(
                user_num, 
                max_dim,
                factors=args.factors,
                epochs=args.epochs,
                optimizer=args.optimizer,
                lr=args.lr,
                reg_1=args.reg_1,
                reg_2=args.reg_2,
                loss_type=args.loss_type,
                reindex=args.reindex,
                X=X if args.gce else None,
                GCE_flag=args.gce,
                A=edge_idx if args.gce else None,
                gpuid=args.gpu
            )
        elif args.algo_name == 'fm':
            from daisy.model.point.FMRecommender import PointFM
            model = PointFM(
                user_num, 
                max_dim,
                factors=args.factors,
                optimizer=args.optimizer,
                epochs=args.epochs,
                lr=args.lr,
                reg_1=args.reg_1,
                reg_2=args.reg_2,
                loss_type=args.loss_type,
                GCE_flag=args.gce,
                reindex=args.reindex,
                X=X if args.gce else None,
                A=edge_idx if args.gce else None,
                gpuid=args.gpu
            )
        elif args.algo_name == 'neumf':
            from daisy.model.point.NeuMFRecommender import PointNeuMF
            model = PointNeuMF(
                user_num, 
                max_dim,
                factors=args.factors,
                num_layers=args.num_layers,
                q=args.dropout,
                optimizer=args.optimizer,
                lr=args.lr,
                epochs=args.epochs,
                reg_1=args.reg_1,
                reg_2=args.reg_2,
                loss_type=args.loss_type,
                GCE_flag=args.gce,
                reindex=args.reindex,
                # X=X if args.gce else None,
                # A=edge_idx if args.gce else None,
                gpuid=args.gpu
            )
        elif args.algo_name == 'nfm':
            from daisy.model.point.NFMRecommender import PointNFM
            model = PointNFM(
                user_num,
                max_dim,
                factors=args.factors,
                optimizer=args.optimizer,
                act_function=args.act_func,
                num_layers=args.num_layers,
                batch_norm=args.no_batch_norm,
                q=args.dropout,
                epochs=args.epochs,
                lr=args.lr,
                reg_1=args.reg_1,
                reg_2=args.reg_2,
                loss_type=args.loss_type,
                GCE_flag=args.gce,
                reindex=args.reindex,
                X=X if args.gce else None,
                A=edge_idx if args.gce else None,
                gpuid=args.gpu
            )
        elif args.algo_name == 'deepfm':
            from daisy.model.point.DeepFMRecommender import PointDeepFM
            model = PointDeepFM(
                user_num,
                max_dim,
                factors=args.factors,
                act_activation=args.act_func,
                optimizer=args.optimizer,
                num_layers=args.num_layers,
                batch_norm=args.no_batch_norm,
                q=args.dropout,
                epochs=args.epochs,
                lr=args.lr,
                reg_1=args.reg_1,
                reg_2=args.reg_2,
                loss_type=args.loss_type,
                GCE_flag=args.gce,
                reindex=args.reindex,
                context_flag=args.context,
                X=X if args.gce else None,
                A=edge_idx if args.gce else None,
                gpuid=args.gpu,
            )
        elif args.algo_name == 'cdae':
            from daisy.model.CDAERecommender import CDAE
            model = CDAE(
                rating_mat=training_mat,
                factors=args.factors,
                act_activation=args.act_func,
                out_activation=args.out_func,
                epochs=args.epochs,
                lr=args.lr,
                q=args.dropout,
                reg_1=args.reg_1,
                reg_2=args.reg_2,
                loss_type=args.loss_type,
                gpuid=args.gpu,
            )
        elif args.algo_name == 'vae':
            from daisy.model.VAERecommender import VAE
            model = VAE(
                rating_mat=training_mat,
                q=args.dropout,
                epochs=args.epochs,
                lr=args.lr,
                reg_1=args.reg_1,
                reg_2=args.reg_2,
                loss_type=args.loss_type,
                gpuid=args.gpu
            )
        else:
            raise ValueError('Invalid algorithm name')
    elif args.problem_type == 'pair':
        if args.algo_name == 'mf':
            from daisy.model.pair.MFRecommender import PairMF
            model = PairMF(
                user_num, 
                item_num,
                factors=args.factors,
                epochs=args.epochs,
                lr=args.lr,
                reg_1=args.reg_1,
                reg_2=args.reg_2,
                loss_type=args.loss_type,
                GCE_flag=args.gce,
                reindex=args.reindex,
                X=X if args.gce else None,
                A=edge_idx if args.gce else None,
                gpuid=args.gpu
            )
        elif args.algo_name == 'fm':
            from daisy.model.pair.FMRecommender import PairFM
            model = PairFM(
                user_num, 
                item_num,
                factors=args.factors,
                epochs=args.epochs,
                lr=args.lr,
                reg_1=args.reg_1,
                reg_2=args.reg_2,
                loss_type=args.loss_type,
                GCE_flag=args.gce,
                reindex=args.reindex,
                X=X if args.gce else None,
                A=edge_idx if args.gce else None,
                gpuid=args.gpu
            )
        elif args.algo_name == 'neumf':
            from daisy.model.pair.NeuMFRecommender import PairNeuMF
            model = PairNeuMF(
                user_num, 
                item_num,
                factors=args.factors,
                num_layers=args.num_layers,
                q=args.dropout,
                lr=args.lr,
                epochs=args.epochs,
                reg_1=args.reg_1,
                reg_2=args.reg_2,
                loss_type=args.loss_type,
                gpuid=args.gpu
            )
        elif args.algo_name == 'nfm':
            from daisy.model.pair.NFMRecommender import PairNFM
            model = PairNFM(
                user_num, 
                item_num,
                factors=args.factors,
                act_function=args.act_func,
                num_layers=args.num_layers,
                batch_norm=args.no_batch_norm,
                q=args.dropout,
                epochs=args.epochs,
                lr=args.lr,
                reg_1=args.reg_1,
                reg_2=args.reg_2,
                loss_type=args.loss_type,
                GCE_flag=args.gce,
                reindex=args.reindex,
                X=X if args.gce else None,
                A=edge_idx if args.gce else None,
                gpuid=args.gpu
            )
        else:
            raise ValueError('Invalid algorithm name')
    else:
        raise ValueError('Invalid problem type')

    # if args.algo_name == 'mostpop':
    #     train_loader = train_dataset
    #     args.num_workers = 0
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # build recommender model
    s_time = time.time()
    # TODO: refactor train
    if args.problem_type == 'pair':
        # model.fit(train_loader)
        from daisy.model.pair.train import train
        train(args, model, train_loader, device, args.context)
    elif args.problem_type == 'point':
        from daisy.model.point.train import train
        train(args, model, train_loader, device, args.context)
    else:
        raise ValueError()
    # model.fit(train_loader)
    elapsed_time = time.time() - s_time

    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    time_log.write(f'{args.dataset}_{args.prepro}_{args.test_method}_{args.problem_type}{args.algo_name}'
                   f'_{args.loss_type}_{args.sample_method}_GCE={args.gce},  {minutes:.2f} min, {seconds:.4f}seconds' + '\n')
    time_log.close()

    print('Start Calculating Metrics......')

    test_ucands = build_candidates_set(test_ur, total_train_ur, item_pool, candidates_num)

    # get predict result
    print('')
    print('Generate recommend list...')
    print('')
    preds = {}
    if args.algo_name in ['vae', 'cdae'] and args.problem_type == 'point':
        for u in tqdm(test_ucands.keys()):
            pred_rates = [model.predict(u, i) for i in test_ucands[u]]
            rec_idx = np.argsort(pred_rates)[::-1][:args.topk]
            top_n = np.array(test_ucands[u])[rec_idx]
            preds[u] = top_n
    else:
        for u in tqdm(test_ucands.keys()):
            # build a test MF dataset for certain user u to accelerate
            if args.context:
                tmp = pd.DataFrame({
                    'user': [u[0] for _ in test_ucands[u]],
                    'item': test_ucands[u],
                    'context': [u[1] for _ in test_ucands[u]],
                    'rating': [0. for _ in test_ucands[u]],  # fake label, make nonsense
                })
            else:
                tmp = pd.DataFrame({
                    'user': [u for _ in test_ucands[u]],
                    'item': test_ucands[u],
                    'rating': [0. for _ in test_ucands[u]], # fake label, make nonsense
                })
            tmp_neg_set = sampler.transform(tmp, is_training=False, context=args.context)
            tmp_dataset = PairData(tmp_neg_set, is_training=False, context=args.context)
            tmp_loader = data.DataLoader(
                tmp_dataset,
                batch_size=candidates_num, 
                shuffle=False, 
                num_workers=0
            )
            # get top-N list with torch method 
            for items in tmp_loader:
                user_u, item_i, context = items[0], items[1], items[2]
                user_u = user_u.to(device)
                item_i = item_i.to(device)
                context = context.to(device) if args.context else None

                prediction = model.predict(user_u, item_i, context)
                _, indices = torch.topk(prediction, args.topk)
                top_n = torch.take(torch.tensor(test_ucands[u]), indices).cpu().numpy()

            preds[u] = top_n

    # convert rank list to binary-interaction
    for u in preds.keys():
        preds[u] = [1 if i in test_ur[u] else 0 for i in preds[u]]
    # process topN list and store result for reporting KPI
    print('Save metric@k result to res folder...')
    result_save_path = f'./res/{args.dataset}/{args.prepro}/{args.test_method}/'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    res = pd.DataFrame({'metric@K': ['pre', 'rec', 'hr', 'map', 'mrr', 'ndcg']})
    for k in [1, 5, 10, 20, 30, 50]:
        if k > args.topk:
            continue
        tmp_preds = preds.copy()        
        tmp_preds = {key: rank_list[:k] for key, rank_list in tmp_preds.items()}

        pre_k = np.mean([precision_at_k(r, k) for r in tmp_preds.values()])
        rec_k = recall_at_k(tmp_preds, test_ur, k)
        hr_k = hr_at_k(tmp_preds, test_ur)
        map_k = map_at_k(tmp_preds.values())
        mrr_k = mrr_at_k(tmp_preds, k)
        ndcg_k = np.mean([ndcg_at_k(r, k) for r in tmp_preds.values()])

        if k == 10:
            # print(f'Precision@{k}: {pre_k:.4f}')
            # print(f'Recall@{k}: {rec_k:.4f}')
            print(f'HR@{k}: {hr_k:.4f}')
            # print(f'MAP@{k}: {map_k:.4f}')
            # print(f'MRR@{k}: {mrr_k:.4f}')
            print(f'NDCG@{k}: {ndcg_k:.4f}')

        res[k] = np.array([pre_k, rec_k, hr_k, map_k, mrr_k, ndcg_k])

    common_prefix = f'with_{args.sample_ratio}{args.sample_method}'
    algo_prefix = f'{args.loss_type}_{args.problem_type}_{args.algo_name}'

    res.to_csv(
        f'{result_save_path}{algo_prefix}_{common_prefix}_GCE={args.gce}_kpi_results.csv',
        index=False
    )

    print('+'*80)
    print('+'*80)
    print(f'TRAINING ELAPSED TIME: {minutes:.2f} min, {seconds:.4f}seconds')

    elapsed_time_total = time.time() - s_time
    hours, rem = divmod(elapsed_time_total, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f'TOTAL ELAPSED TIME: {minutes:.2f} min, {seconds:.4f}seconds')
