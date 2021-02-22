import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from hyperopt import hp, tpe, fmin, Trials, space_eval
import ast

import torch
import torch.utils.data as data
from torch_geometric.utils import from_scipy_sparse_matrix
from IPython import embed


from daisy.utils.sampler import Sampler
from daisy.utils.parser import parse_args
from daisy.model.pair.train import train
from daisy.utils.data import PairData, sparse_mx_to_torch_sparse_tensor, incorporate_in_ml100k, incorporate_sinfo_by_dim
from daisy.utils.splitter import split_test, split_validation
from daisy.utils.loader import load_rate, get_ur, build_candidates_set, add_last_clicked_item_context
from scipy.sparse import identity, csr_matrix


from main import build_evaluation_set


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def tune_main_function():
    def opt_func(space):

        ''' FORMAT DATA AND CHOOSE MODEL '''
        f = space['f']
        args = Struct(**space)

        user_num = dims[0]
        max_dim = dims[2] if args.context else dims[1]
        if args.gce and args.side_information:
            # TODO: I THINK ITS LIKE THIS! UNCOMMENT
            print('GCE GOOD WAY')
            max_dim = X.shape[1]
            # TODO: I THINK ITS LIKE THIS! COMMENT
            # X = torch.transpose(X, 0, 1)

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
            model = PairNGCF(
                n_users=user_num,
                n_items=max_dim,
                embed_size=args.factors,
                adj_matrix=adj_mx,
                device=device,
                reindex=args.reindex,
                n_layers=2
            )
        elif args.algo_name == 'ncf':
            # layers = [len(dims[:-2]) * 32, 32, 16, 8] if not args.context else [len(dims[:-2]) * 32, 32, 16, 8]
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
        else:
            raise ValueError('Invalid algorithm name')

        ''' BUILD RECOMMENDER PIPELINE '''

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        loaders, candidates = build_evaluation_set(val_ur, total_train_ur, item_pool, candidates_num, sampler,
                                                   context_flag=args.context, tune=args.tune)
        try:
            score = train(args, model, train_loader, device, args.context, loaders, candidates, val_ur, tune=args.tune,
                          f=f)
        except:
            score = -1
        return score

    ''' all parameter part '''
    ####################################
    # TODO: convert in function to stop copying code from main.py
    ####################################
    args = parse_args()
    seed = args.seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = "cuda"
    else:
        device = "cpu"

    ''' LOAD DATA AND ADD CONTEXT IF NECESSARY '''
    df, users, items, unique_original_items, _ = load_rate(args.dataset, args.prepro, binary=True,
                                                           context=args.context,
                                                           gce_flag=args.gce, cut_down_data=args.cut_down_data,
                                                           side_info=args.side_information,
                                                           context_type=args.context_type,
                                                           context_as_userfeat=args.context_as_userfeat)
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
            aux_si = df.iloc[:, :-2].copy()  # take all columns unless user, rating and timestamp
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
                df = add_last_clicked_item_context(df, args.dataset)
                if not args.uii:
                    df['context'] = df['context'] + items
                # check last number is positive
                assert df['item'].tail().values[-1] > 0

    ''' SPLIT DATA '''
    train_set, test_set = split_test(df, args.test_method, args.test_size)
    train_set, val_set, _ = split_validation(train_set, val_method=args.test_method, list_output=False)

    # temporary used for tuning test result
    # train_set = pd.read_csv(f'./experiment_data/train_{args.dataset}_{args.prepro}_{args.test_method}.dat')
    # test_set = pd.read_csv(f'./experiment_data/test_{args.dataset}_{args.prepro}_{args.test_method}.dat')
    df = pd.concat([train_set, test_set], ignore_index=True)
    if 'array_context_flag' in df.columns:
        # type(df.to_numpy()[0][2]) == list
        import itertools
        prot_list = list(itertools.chain(df['context'].values))
        context_num = 1

        flattened_proteins = [val for sublist in prot_list for val in sublist]

        dims = np.max(df[['user', 'item']].to_numpy().astype(int), axis=0) + 1
        dims = np.append(dims, [np.max(flattened_proteins) + 1])
    else:
        dims = np.max(df.to_numpy().astype(int), axis=0) + 1
        context_num = int(len(dims) - 4)
        
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

    print('=' * 50, '\n')

    sampler = Sampler(
        dims,
        num_ng=args.num_ng,
        sample_method=args.sample_method,
        sample_ratio=args.sample_ratio,
        reindex=args.reindex
    )

    neg_set, adj_mx = sampler.transform(train_set, is_training=True, context=args.context, pair_pos=None)
    if args.gce:
        if args.mh > 1:
            print(f'[ MULTI HOP {args.mh} ACTIVATED ]')
            adj_mx = adj_mx.__pow__(int(args.mh))
        X = sparse_mx_to_torch_sparse_tensor(identity(adj_mx.shape[0])).to(device)
        if args.side_information:
            if args.dataset == 'ml-100k':
                # X_gender = sparse_mx_to_torch_sparse_tensor(X_gender_mx).to(device)
                # X = torch.cat((X, X_gender), -1)  # torch.Size([2096, 2114])  2096 + 18 = 2114
                si = pd.read_csv(f'./data/{args.dataset}/side-information.csv', index_col=0)
                si.rename(columns={'id': 'item', 'genres': 'side_info'}, inplace=True)
                # si = si[['item', 'side_info']]
                if df['item'].min() > 0:  # Reindex items
                    # TODO: INCORPORATE si_extension to X
                    si_extension = incorporate_in_ml100k(si[['item', 'side_info']], X.shape[1], unique_original_items,
                                                         users)
                    X_gender = sparse_mx_to_torch_sparse_tensor(csr_matrix(si_extension.values)).to(device)
                    if args.actors:
                        si.drop(columns=['side_info'], inplace=True)
                        si.rename(columns={'actors': 'side_info'}, inplace=True)
                        si_ext = incorporate_in_ml100k(si[['item', 'side_info']], X.shape[1], unique_original_items,
                                                       users)
                        X_sinfo = sparse_mx_to_torch_sparse_tensor(csr_matrix(si_ext.values)).to(device)
                        X = torch.cat((X, X_gender, X_sinfo), -1)
                    else:
                        X = torch.cat((X, X_gender), -1)  # torch.Size([2096, 2114])  2096 + 18 = 2114
            elif args.dataset == 'music':  # MORE GENERIC CASE
                si_extension = incorporate_sinfo_by_dim(aux_si, X.shape[0], users)
                X_sinfo = sparse_mx_to_torch_sparse_tensor(csr_matrix(si_extension.values)).to(device)
                X = torch.cat((X, X_sinfo), -1)
            else:  #lastfm  # frappe
                cat_mx = []
                for col in aux_si.columns[2:]:
                    # context_as_userfeat
                    si_extension = incorporate_sinfo_by_dim(aux_si, X.shape[1], dims, col=col,
                                                            contextasfeature=args.context_as_userfeat)
                    X_sinfo = sparse_mx_to_torch_sparse_tensor(csr_matrix(si_extension.astype(str).astype(int).values)).to(device)
                    cat_mx.append(X_sinfo)
                X_sinfo = torch.cat(cat_mx, -1)
                X = torch.cat((X, X_sinfo), -1)

        # We retrieve the graph's edges and send both them and graph to device in the next two lines
        edge_idx, edge_attr = from_scipy_sparse_matrix(adj_mx)
        edge_idx = edge_idx.to(device)
    train_dataset = PairData(train_set, sampler=sampler, adj_mx=adj_mx, is_training=True, context=args.context)
    ####################################
    ####################################
    print('='*50, '\n')
    # begin tuning here
    tune_log_path = 'tune_logs'
    os.makedirs(tune_log_path, exist_ok=True)
    # max_evals = 10 if args.dataset == 'ml-1m' else 50
    max_evals = args.max_evals
    string = 'NOCONTEXT' if not args.context else ''
    mh = f'MH={args.mh}' if args.mh >1 else ''
    string2 = 'UII' if args.uii else 'UIC'
    si_str = 'SINFO' if args.side_information else ''
    context_type = args.context_type if args.dataset == 'frappe' else ""
    f = open(tune_log_path + "/" + f'{args.loss_type}_{args.algo_name}_{context_type}_{string}_{string2}_GCE={args.gce}_{args.gcetype}_{si_str}_'
    f'{args.dataset}_{args.prepro}_{args.val_method}_context_as_userfeat={args.context_as_userfeat}_max_evals={max_evals}.csv',
             'w', encoding='utf-8')
    f.write('HR, NDCG, best_epoch, num_ng, factors, num_heads, dropout, lr, batch_size, reg_1, reg_2' + '\n')
    f.flush()

    # param_limit = param_extract(args)
    # param_dict = confirm_space(param_limit)

    args_dict = vars(args)

    lr_range = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    batch_size_range = [256, 512, 1024, 2048]
    do_range = [0, 0.15, 0.5, 0.6]
    if args_dict['gcetype'] == 'gat':
        num_heads_range = [1, 2, 3, 4, 5, 6]
    else:
        num_heads_range = [1]

    args_dict['lr'] = hp.choice('lr', lr_range)
    args_dict['batch_size'] = hp.choice('batch_size', batch_size_range)
    args_dict['dropout'] = hp.choice('dropout', do_range)
    args_dict['num_heads'] = hp.choice('num_heads', num_heads_range)

    args_dict['epochs'] = args_dict['tune_epochs']
    args_dict['f'] = f
    args_dict['tune'] = True

    trials = Trials()
    # trials = pickle.load(open("myfile.p", "rb"))  # then max_evals needs to be set to 200

    space = defaultdict(None, args_dict)
    best = fmin(fn=opt_func,
                space=space, algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    # pickle.dump(trials, open("myfile.p", "wb"))
    print(""*20 +'BEST HYPER_PARAMS:' + ""*20)
    print("lr = " + str(lr_range[best['lr']]))
    print("batch_size = " + str(batch_size_range[best['batch_size']]))
    print("dropout = " + str(do_range[best['dropout']]))
    print("num_heads = " + str(num_heads_range[best['num_heads']]))
    # lr_range[best['lr']]
    best_options = space_eval(space, trials.argmin)

    f.write('BEST ITERATION PARAMS' + '\n')
    f.write(f"-, -, -, {best_options['num_ng']}, {best_options['factors']}, {best_options['num_heads']}, {best_options['dropout']},"
            f"+ {best_options['lr']},{best_options['batch_size']}, {best_options['reg_1']}, {best_options['reg_2']}" + '\n')
    f.flush()
    f.close()
    return best_options


if __name__ == '__main__':

    best_options = tune_main_function()
    from main import *
    args = parse_args()
    args.lr = best_options['lr']
    args.batch_size = best_options['batch_size']
    args.dropout = best_options['dropout']
    args.num_heads = best_options['num_heads']

    main(args)
