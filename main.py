import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import ast
from daisy.utils.sampler import Sampler
from daisy.utils.parser import parse_args
from daisy.utils.splitter import split_test, split_validation, perform_evaluation
from daisy.utils.data import PointData, PairData, incorporate_in_ml100k, sparse_mx_to_torch_sparse_tensor,\
    incorporate_sinfo_by_dim
from daisy.utils.loader import load_rate, get_ur, convert_npy_mat, build_candidates_set, add_last_clicked_item_context

from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import identity, csr_matrix

from IPython import embed


def build_evaluation_set(test_ur, total_train_ur, item_pool, candidates_num, sampler, context_flag=False, tune=False):
    test_ucands = build_candidates_set(test_ur, total_train_ur, item_pool, candidates_num, context_flag=context_flag)

    #embed()
    # get predict result
    print('')
    print('Generate recommend list...')
    print('')
    loaders = {}
    for u in tqdm(test_ucands.keys(), disable=tune):
        # build a test MF dataset for certain user u to accelerate
        if context_flag:
            # tmp = pd.DataFrame({
            #     'user': u[0],
            #     'item': test_ucands[u],
            #     'context': [*u[1:]],
            #     'rating': [0. for _ in test_ucands[u]],  # fake label, make nonsense
            # })
            tmp = pd.DataFrame({
                'user': [u[0] for _ in test_ucands[u]],
                'item': test_ucands[u],
                'context': [[*u[1:]] for _ in test_ucands[u]],
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


def main(args=None):
    ''' all parameter part '''
    if args is None:
        args = parse_args()

    # if args.algo_name == 'ngcf':
    #     args.gce = True

    # for visualization
    date = datetime.now().strftime('%y%m%d%H%M%S')
    if args.logs:
        if len(args.logsname) == 0:
            string1 = "SINFO" if args.side_information else ""
            random_context = "random_context" if args.random_context else ""
            context_type = args.context_type if args.dataset == 'frappe' else ""
            rankall = 'RANK_ALL' if args.rankall else ""
            INIT = "INIT" if args.load_init_weights else ""
            string2 = "reindexed" if args.reindex and not args.gce else f"graph_{args.gcetype}"
            string3 = "_UII_" if args.uii and args.context else "_UIC_"
            string = string1 + string2 + string3
            context_folder = "context" if args.context else "no_context"
            loss = 'BPR' if args.loss_type == "BPR" else "CL"
            sampling = 'neg_sampling_each_epoch' if args.neg_sampling_each_epoch else ""
            stopping = 'not_early_stopping' if args.not_early_stopping else ""

            total_info = f'{args.algo_name}_{rankall}_lr={args.lr}_DO={args.dropout}_bs={args.batch_size}_{string}' \
                f'_{args.epochs}epochs_{sampling}_{stopping}'
            writer = SummaryWriter(log_dir=f'logs/{args.dataset}/{context_folder}/logs_{total_info}_{date}/')

        else:
            writer = SummaryWriter(log_dir=f'logs/{args.dataset}/logs_{args.logsname}_{date}/')
    else:
        writer = SummaryWriter(log_dir=f'logs/nologs/logs/')

    # p = multiprocessing.Pool(args.num_workers)
    # FIX SEED AND SELECT DEVICE
    seed = args.seed
    args.lr = float(args.lr)
    args.batch_size = int(args.batch_size)
    args.dropout = float(args.dropout)
    
    if torch.cuda.is_available() and args.remove_top_users == 0:
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
    df, users, items, unique_original_items, popularity_dic = load_rate(args.dataset, args.prepro, binary=True,
                                                                        context=args.context,
                                                                        gce_flag=args.gce,
                                                                        cut_down_data=args.cut_down_data,
                                                                        remove_top_users=args.remove_top_users,
                                                                        remove_on=args.remove_on,
                                                                        side_info=args.side_information,
                                                                        context_type=args.context_type,
                                                                        context_as_userfeat=args.context_as_userfeat,
                                                                        flag_run_statistics=args.statistics)
    if args.side_information and not args.dataset == 'ml-100k':
        if args.dataset in ['lastfm', 'drugs']:
            if args.context_as_userfeat:
                aux_si = df.iloc[:, :-2].copy()  # take all columns unless user, rating and timestamp
                args.context = False
            else:
                if 'context' in df.columns:
                    aux_si = df.iloc[:, :-3].copy()  # take all columns unless user, rating and timestamp
                    aux_si.drop(columns=['context'], inplace=True)
                else:
                    aux_si = df.iloc[:, :-2].copy()  # take all columns unless user, rating and timestamp

        elif (np.unique(df['timestamp']) == [1])[0]:
            # BIPARTED GRAPH
            args.context = False
            print('BI-PARTED GRAPH WITH X')
            aux_si = df.iloc[:, :-2].copy() # take all columns unless user, rating and timestamp
        else:
            aux_si = df[['item', 'side_info']].copy()
            aux_si = aux_si.drop_duplicates('item')
    else:
        if 'user-feat' in df.columns:
            df.drop(columns=['user-feat'], inplace=True)
        print('NO SIDE EFFECT')
    if args.reindex:
        if 'array_context_flag' in df.columns:  # isinstance(row['context'], list)
            if args.context:
                assert (np.unique(df['timestamp']) == 1)[0] == True
                df['user'] = df['user'].astype(np.int64)
                df['item'] = df['item'].astype(np.int64)
                df['item'] = df['item'] + users
                df['context'] = df['context'].apply(lambda x: ast.literal_eval(x))
                df['context'] = df['context'].apply(lambda x: [protein + (users+items) for protein in x])
                #and type(ast.literal_eval(df['context'][0])) is list:
                # timestamp is forced
        else:
            df = df.astype(np.int64)
            df['item'] = df['item'] + users
            if args.context:
                df = add_last_clicked_item_context(df, args.dataset, args.random_context)
                # add context as independent nodes
                if not args.uii:
                    df['context'] = df['context'] + items
                # TODO: else: df['context'] = df['context'].apply(lambda x: [c] for c in x)
                # check last number is positive
                # np.max(df.to_numpy(), axis=0)
                assert df['item'].tail().values[-1] > 0
    ''' SPLIT DATA '''
    train_set, test_set = split_test(df, args.test_method, args.test_size)
    train_set, val_set, _ = split_validation(train_set, val_method=args.test_method, list_output=False)

    # temporary used for tuning test result
    # train_set = pd.read_csv(f'./experiment_data/train_{args.dataset}_{args.prepro}_{args.test_method}.dat')
    # test_set = pd.read_csv(f'./experiment_data/test_{args.dataset}_{args.prepro}_{args.test_method}.dat')
    df = pd.concat([train_set, test_set], ignore_index=True)

    if not args.context:
        dims = np.max(df.to_numpy().astype(int), axis=0) + 1
    else:
        if 'array_context_flag' in df.columns:
            # type(df.to_numpy()[0][2]) == list
            import itertools
            prot_list = list(itertools.chain(df['context'].values))
            # lenghts = [len(np.unique(seq)) for seq in prot_list]
            # context_num = int(np.max(lenghts))
            context_num = 1

            flattened_proteins = [val for sublist in prot_list for val in sublist]

            dims = np.max(df[['user', 'item']].to_numpy().astype(int), axis=0) + 1
            dims = np.append(dims, [np.max(flattened_proteins)+1])
        else:
            dims = np.max(df.to_numpy().astype(int), axis=0) + 1
            context_num = int(len(dims) - 4)

    ''' GET GROUND-TRUTH AND CANDIDATES '''
    # get ground truth
    test_ur = get_ur(test_set, context=args.context, eval=False)
    val_ur = get_ur(val_set, context=args.context, eval=False)
    
    total_train_ur = get_ur(train_set, context=args.context, eval=True)
    # initial candidate item pool
    item_pool = set(range(dims[0], dims[1])) if args.reindex else set(range(dims[1]))
    candidates_num = items if args.rankall else args.cand_num

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
        # if args.mh > 1:
        #     print(f'[ MULTI HOP {args.mh} ACTIVATED ]')
        #     adj_mx = adj_mx.__pow__(int(args.mh))
        X = sparse_mx_to_torch_sparse_tensor(identity(adj_mx.shape[0])).to(device)
        # X, _ = from_scipy_sparse_matrix(identity(adj_mx.shape[0]))
        # X = X.to(device)
        if args.side_information:
            if args.dataset == 'ml-100k':
                # X_gender = sparse_mx_to_torch_sparse_tensor(X_gender_mx).to(device)
                # X = torch.cat((X, X_gender), -1)  # torch.Size([2096, 2114])  2096 + 18 = 2114
                si = pd.read_csv(f'./data/{args.dataset}/side-information.csv', index_col=0)
                si.rename(columns={'id': 'item', 'genres': 'side_info'}, inplace=True)
                # si = si[['item', 'side_info']]
                if df['item'].min() > 0:    # Reindex items
                    # TODO: INCORPORATE si_extension to X
                    si_extension = incorporate_in_ml100k(si[['item', 'side_info']], X.shape[1], unique_original_items,
                                                         users)
                    X_gender = sparse_mx_to_torch_sparse_tensor(csr_matrix(si_extension.values)).to(device)
                    if args.actors:
                        si.drop(columns=['side_info'], inplace=True)
                        si.rename(columns={'actors': 'side_info'}, inplace=True)
                        si_ext = incorporate_in_ml100k(si[['item', 'side_info']], X.shape[1], unique_original_items, users)
                        X_sinfo = sparse_mx_to_torch_sparse_tensor(csr_matrix(si_ext.values)).to(device)
                        X = torch.cat((X, X_gender, X_sinfo), -1)
                    else:
                        X = torch.cat((X, X_gender), -1)  # torch.Size([2096, 2114])  2096 + 18 = 2114

            elif args.dataset == 'music':  # MORE GENERIC CASE
                si_extension = incorporate_sinfo_by_dim(aux_si, X.shape[1], users)
                X_sinfo = sparse_mx_to_torch_sparse_tensor(csr_matrix(si_extension.values)).to(device)
                X = torch.cat((X, X_sinfo), -1)
            else:  #lastfm  # drugs
                cat_mx = []
                for col in aux_si.columns[2:]:
                    # context_as_userfeat
                    si_extension = incorporate_sinfo_by_dim(aux_si, X.shape[1], dims, col=col,
                                                            contextasfeature=args.context_as_userfeat)
                    X_sinfo = sparse_mx_to_torch_sparse_tensor(csr_matrix(si_extension.astype(str).astype(int).values)).to(device)
                    cat_mx.append(X_sinfo)
                X_sinfo = torch.cat(cat_mx, -1)
                X = torch.cat((X, X_sinfo), -1)
        # embed()
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
    # X = X.to_dense()
    
    if args.gce and args.side_information:
        # TODO: I THINK ITS LIKE THIS! UNCOMMENT
        print('GCE GOOD WAY')
        max_dim = X.shape[1]
        # TODO: I THINK ITS LIKE THIS! COMMENT
        # X = torch.transpose(X, 0, 1)
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
                dropout=args.dropout,
                args=args
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
                dropout=args.dropout,
                args=args
            )
        elif args.algo_name == 'ngcf':
            from daisy.model.pair.NGCF import PairNGCF
            # model = PairNGCF(
            #     n_users=user_num,
            #     n_items=max_dim,
            #     embed_size=args.factors,
            #     adj_matrix=adj_mx,
            #     device=device,
            #     reindex=args.reindex,
            #     n_layers=2
            #
            # )
            model = PairNGCF(
                n_users=user_num,
                max_dim=max_dim,
                emb_dim=args.factors,
                adj_mtx=adj_mx,
                device=device,
                reindex=args.reindex,
                layers=[64, 64],
                node_dropout=args.dropout,
                mess_dropout=0
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
            # layers = [len(dims[:-2])*32, 32, 16, 8] if not args.context else [len(dims[:-2])*32, 32, 16, 8]
            # args.factors = layers[1]
            from daisy.model.pair.NCFRecommender import PairNCF
            model = PairNCF(
                user_num,
                max_dim,
                args.factors,
                num_layers=3,
                GCE_flag=args.gce,
                reindex=args.reindex,
                X=X if args.gce else None,
                A=edge_idx if args.gce else None,
                gpuid=args.gpu,
                mf=args.mf,
                dropout=args.dropout,
                num_context=context_num
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

    def collate_fn(batch):
        return list(zip(*batch))

    ''' BUILD RECOMMENDER PIPELINE '''
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=args.num_workers,
        # collate_fn=lambda x: x
        collate_fn=collate_fn
    )

    # TODO: refactor train
    if args.problem_type == 'pair':
        if args.remove_top_users > 0:
            # do inference
            print('+'*80)
            print('NO TRAINING -- JUST INFERENCE')
            ''' INFERENCE '''
            print('TEST_SET: Start Calculating Metrics......')
            loaders_test, candidates_test = build_evaluation_set(test_ur, total_train_ur, item_pool, candidates_num,
                                                                 sampler, context_flag=args.context)
            s_time = time.time()
            elapsed_time = time.time() - s_time

            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            _, _, _ = perform_evaluation(loaders_test, candidates_test, model, args, device, test_ur, s_time,
                                         minutes_train=minutes, writer=None, seconds_train=seconds, desc=total_info,
                                         populary_dict=popularity_dic)
            exit()

        else:
            loaders, candidates = build_evaluation_set(val_ur, total_train_ur, item_pool, candidates_num, sampler,
                                                       context_flag=args.context)
            s_time = time.time()
            from daisy.model.pair.train import train
            train(args, model, train_loader, device, args.context, loaders, candidates, val_ur, writer=writer,
                  desc=total_info)
    elif args.problem_type == 'point':
        loaders, candidates = build_evaluation_set(val_ur, total_train_ur, item_pool, candidates_num, sampler,
                                                   context_flag=args.context)
        from daisy.model.point.train import train
        train(args, model, train_loader, device, args.context, loaders, candidates, val_ur, writer=writer)
    else:
        raise ValueError()

    elapsed_time = time.time() - s_time

    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'TOTAL ELAPSED TIME: {hours:.2f} hours, {minutes:.2f} min, {seconds:.4f}seconds')

    time_log.write(f'{args.dataset}_{args.prepro}_{args.test_method}_{args.problem_type}{args.algo_name}'
                   f'_{args.loss_type}_{args.sample_method}_GCE={args.gce},  {minutes:.2f} min, {seconds:.4f}seconds' + '\n')
    time_log.close()

    # print('+'*80)
    # ''' TEST METRICS '''
    # print('TEST_SET: Start Calculating Metrics......')
    # loaders_test, candidates_test = build_evaluation_set(test_ur, total_train_ur, item_pool, candidates_num,
    #                                                      sampler, context_flag=args.context)
    # perform_evaluation(loaders_test, candidates_test, model, args, device, test_ur, s_time, minutes_train=minutes,
    #                    writer=None, seconds_train=seconds)


if __name__ == '__main__':
    main()
