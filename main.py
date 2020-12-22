import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from daisy.utils.sampler import Sampler
from daisy.utils.parser import parse_args
from daisy.utils.splitter import split_test, split_validation, perform_evaluation
from daisy.utils.data import PointData, PairData, UAEData, sparse_mx_to_torch_sparse_tensor
from daisy.utils.loader import load_rate, get_ur, convert_npy_mat, build_candidates_set, add_last_clicked_item_context

from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import identity
from IPython import embed


def build_evaluation_set(test_ur, total_train_ur, item_pool, candidates_num, sampler, context_flag=False, tune=False):
    test_ucands = build_candidates_set(test_ur, total_train_ur, item_pool, candidates_num, context_flag=context_flag)

    # get predict result
    print('')
    print('Generate recommend list...')
    print('')
    loaders = {}
    for u in tqdm(test_ucands.keys(), disable=tune):
        # build a test MF dataset for certain user u to accelerate
        if context_flag:
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
                'rating': [0. for _ in test_ucands[u]],  # fake label, make nonsense
            })
        tmp_neg_set = sampler.transform(tmp, is_training=False, context=context_flag)
        tmp_dataset = PairData(tmp_neg_set, is_training=False, context=context_flag)
        tmp_loader = data.DataLoader(
            tmp_dataset,
            batch_size=candidates_num,
            shuffle=False,
            num_workers=0
        )
        loaders[u] = tmp_loader

    return loaders, test_ucands


if __name__ == '__main__':
    ''' all parameter part '''
    args = parse_args()

    # for visualization
    date = datetime.now().strftime('%y%m%d%H%M%S')
    if args.logs:
        if len(args.logsname) == 0:
            string = "reindexed" if args.reindex and not args.gce else "graph"
            context_folder = "context" if args.context else "no_context"
            loss = 'BPR' if args.loss_type == "BPR" else "CL"
            sampling = 'neg_sampling_each_epoch' if args.neg_sampling_each_epoch else ""
            writer = SummaryWriter(log_dir=f'logs/{args.dataset}/{context_folder}/'
            f'logs_{loss}_lr={args.lr}_DO={args.dropout}_{args.algo_name}_{string}_{args.epochs}epochs_{sampling}_{date}/')
        else:
            writer = SummaryWriter(log_dir=f'logs/{args.dataset}/logs_{args.logsname}_{date}/')
    else:
        writer = SummaryWriter(log_dir=f'logs/nologs/logs/')

    if args.dataset == 'epinions':
        args.lr = 0.001

    # p = multiprocessing.Pool(args.num_workers)
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
    
    ''' LOAD DATA AND ADD CONTEXT IF NECESSARY '''
    df, users, items = load_rate(args.dataset, args.prepro, binary=True, context=args.context, gce_flag=args.gce,
                                 cut_down_data=args.cut_down_data)
    if args.reindex:
        df = df.astype(np.int64)
        df['item'] = df['item'] + users
        if args.context:
            df = add_last_clicked_item_context(df, args.dataset)
            # check last number is positive
            assert df['item'].tail().values[-1] > 0

    ''' SPLIT DATA '''
    train_set, test_set = split_test(df, args.test_method, args.test_size)
    train_set, val_set, _ = split_validation(train_set, val_method=args.test_method, list_output=False)

    # temporary used for tuning test result
    # train_set = pd.read_csv(f'./experiment_data/train_{args.dataset}_{args.prepro}_{args.test_method}.dat')
    # test_set = pd.read_csv(f'./experiment_data/test_{args.dataset}_{args.prepro}_{args.test_method}.dat')
    df = pd.concat([train_set, test_set], ignore_index=True)
    dims = np.max(df.to_numpy().astype(int), axis=0) + 1
    if args.dataset in ['yelp']:
        train_set['timestamp'] = pd.to_datetime(train_set['timestamp'], unit='ns')
        test_set['timestamp'] = pd.to_datetime(test_set['timestamp'], unit='ns')

    ''' GET GROUND-TRUTH AND CANDIDATES '''
    # get ground truth
    test_ur = get_ur(test_set, context=args.context, eval=False)
    val_ur = get_ur(val_set, context=args.context, eval=False)
    
    total_train_ur = get_ur(train_set, context=args.context, eval=True)
    # initial candidate item pool
    item_pool = set(range(dims[0], dims[1])) if args.reindex else set(range(dims[1]))
    candidates_num = args.cand_num

    print('='*50, '\n')

    ''' FORMAT DATA AND CHOOSE MODEL '''
    sampler = Sampler(
        dims,
        num_ng=args.num_ng, 
        sample_method=args.sample_method, 
        sample_ratio=args.sample_ratio,
        reindex=args.reindex
    )
    neg_set, adj_mx = sampler.transform(train_set, is_training=True, context=args.context, pair_pos=None)
    if args.gce:
        # embed()
        if args.mh > 1:
            print(f'[ MULTI HOP {args.mh} ACTIVATED ]')
            adj_mx = adj_mx.__pow__(int(args.mh))
        X = sparse_mx_to_torch_sparse_tensor(identity(adj_mx.shape[0])).to(device)
        # We retrieve the graph's edges and send both them and graph to device in the next two lines
        edge_idx, edge_attr = from_scipy_sparse_matrix(adj_mx)
        # TODO: should I pow the matrix here?
        edge_idx = edge_idx.to(device)

    if args.problem_type == 'pair':
        # train_dataset = PairData(neg_set, is_training=True, context=args.context)
        train_dataset = PairData(train_set, sampler=sampler, adj_mx=adj_mx, is_training=True, context=args.context)
    else:
        train_dataset = PointData(neg_set, is_training=True, context=args.context)

    user_num = dims[0]
    max_dim = dims[2] if args.context else dims[1]
    if args.problem_type == 'point':
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
                dropout=args.dropout,
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
                gpuid=args.gpu,
                dropout=args.dropout
            )
        elif args.algo_name == 'neumf':
            from daisy.model.point.NeuMFRecommender import PointNeuMF
            model = PointNeuMF(
                user_num, 
                max_dim,
                factors=args.factors,
                num_layers=args.num_layers,
                dropout=args.dropout,
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
                gpuid=args.gpu,
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
                dropout=args.dropout,
                epochs=args.epochs,
                lr=args.lr,
                reg_1=args.reg_1,
                reg_2=args.reg_2,
                loss_type=args.loss_type,
                GCE_flag=args.gce,
                reindex=args.reindex,
                X=X if args.gce else None,
                A=edge_idx if args.gce else None,
                gpuid=args.gpu,
                mf=args.mf,
            )
        elif args.algo_name == 'ncf':
            layers = [len(dims[:-2])*64, 64, 32, 8] if not args.context else [len(dims[:-2])*64, 64, 32, 8]
            from daisy.model.point.NCF import NCF
            model = NCF(
                user_num,
                max_dim,
                factors=args.factors,
                layers=layers,
                GCE_flag=args.gce,
                reindex=args.reindex,
                X=X if args.gce else None,
                A=edge_idx if args.gce else None,
                gpuid=args.gpu,
                mf=args.mf,
                dropout=args.dropout
            )
        elif args.algo_name == 'ngcf':
            from daisy.model.point.NGCF import NGCF
            model = NGCF(user_num, max_dim, df)
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
                dropout=args.dropout,
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
        else:
            raise ValueError('Invalid algorithm name')
    elif args.problem_type == 'pair':
        if args.algo_name == 'mf':
            from daisy.model.pair.MFRecommender import PairMF
            model = PairMF(
                user_num, 
                max_dim,
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
                gpuid=args.gpu,
                dropout=args.dropout
            )
        elif args.algo_name == 'fm':
            from daisy.model.pair.FMRecommender import PairFM
            model = PairFM(
                user_num, 
                max_dim,
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
                gpuid=args.gpu,
                dropout=args.dropout
            )
        elif args.algo_name == 'neumf':
            from daisy.model.pair.NeuMFRecommender import PairNeuMF
            model = PairNeuMF(
                user_num, 
                max_dim,
                factors=args.factors,
                num_layers=args.num_layers,
                dropout=args.dropout,
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
                max_dim,
                factors=args.factors,
                act_function=args.act_func,
                num_layers=args.num_layers,
                batch_norm=args.no_batch_norm,
                dropout=args.dropout,
                epochs=args.epochs,
                lr=args.lr,
                reg_1=args.reg_1,
                reg_2=args.reg_2,
                loss_type=args.loss_type,
                GCE_flag=args.gce,
                reindex=args.reindex,
                X=X if args.gce else None,
                A=edge_idx if args.gce else None,
                gpuid=args.gpu,
                mf=args.mf
            )
        elif args.algo_name == 'ncf':
            layers = [len(dims[:-2])*32, 32, 16, 8] if not args.context else [len(dims[:-2])*32, 32, 16, 8]
            from daisy.model.pair.NCFRecommender import PairNCF
            model = PairNCF(
                user_num,
                max_dim,
                factors=args.factors,
                layers=layers,
                GCE_flag=args.gce,
                reindex=args.reindex,
                X=X if args.gce else None,
                A=edge_idx if args.gce else None,
                gpuid=args.gpu,
                mf=args.mf,
                dropout=args.dropout
            )
        elif args.algo_name == 'deepfm':
            from daisy.model.pair.DeepFMRecommender import PairDeepFM
            model = PairDeepFM(
                user_num,
                max_dim,
                factors=args.factors,
                act_activation=args.act_func,
                num_layers=args.num_layers,
                batch_norm=args.no_batch_norm,
                dropout=args.dropout,
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
        else:
            raise ValueError('Invalid algorithm name')
    else:
        raise ValueError('Invalid problem type')

    ''' BUILD RECOMMENDER PIPELINE '''

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    loaders, candidates = build_evaluation_set(val_ur, total_train_ur, item_pool, candidates_num,
                                               context_flag=args.context)
    
    s_time = time.time()
    # TODO: refactor train
    if args.problem_type == 'pair':
        # model.fit(train_loader)
        from daisy.model.pair.train import train
        train(args, model, train_loader, device, args.context, loaders, candidates, val_ur, writer=writer)
    elif args.problem_type == 'point':
        from daisy.model.point.train import train
        train(args, model, train_loader, device, args.context, loaders, candidates, val_ur, writer=writer)
    else:
        raise ValueError()
    # model.fit(train_loader)
    elapsed_time = time.time() - s_time

    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    time_log.write(f'{args.dataset}_{args.prepro}_{args.test_method}_{args.problem_type}{args.algo_name}'
                   f'_{args.loss_type}_{args.sample_method}_GCE={args.gce},  {minutes:.2f} min, {seconds:.4f}seconds' + '\n')
    time_log.close()

    print('+'*80)
    ''' TEST METRICS '''
    print('TEST_SET: Start Calculating Metrics......')
    loaders_test, candidates_test = build_evaluation_set(test_ur, total_train_ur, item_pool, candidates_num,
                                                         sampler, context_flag=args.context)
    perform_evaluation(loaders_test, candidates_test, model, args, device, test_ur, s_time, minutes_train=minutes,
                       writer=None, seconds_train=seconds)
