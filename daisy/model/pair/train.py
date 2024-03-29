import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from IPython import embed
import numpy as np
import os
import torch.backends.cudnn as cudnn
from daisy.utils.splitter import perform_evaluation
from hyperopt import STATUS_OK
from hyperopt.progress import tqdm_progress_callback, no_progress_callback


def train(args, model, train_loader, device, context_flag, loaders, candidates, val_ur,  writer=None, tune=False,
          f=None, desc=None):
    cudnn.benchmark = True
    model.to(device)
    if args.optimizer == 'adagrad': #args.optimizer == '':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, initial_accumulator_value=1e-8)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    init_weight_path = f'weights_{args.algo_name}_GCE={args.gce}-{"UII" if args.uii else "UIC"}-epoch_1.pkl'
    if args.load_init_weights:
        if os.path.exists(f'weights/{args.dataset}/{init_weight_path}'):
            checkpoint = torch.load(f"weights/{args.dataset}/{init_weight_path}")
            # checkpoint = torch.load("weights/??/checkpoint_??.pt", map_location=('cpu' if device != 'cuda' else None))
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint['optimizer'])

            assert (model.embeddings.weight == checkpoint["state_dict"]['embeddings.weight'])[0].sum().item() == args.factors
        else:
            raise ValueError('Trying to load INIT weights that do not exist!')

    print(f'RUN FOR {args.epochs} EPOCHS')
    # IDEA: RANDOM EVALUATION
    # populary_dict = train_loader.dataset.set['item'].value_counts(normalize=True).to_dict()

    res, writer, _ = perform_evaluation(loaders, candidates, model, args, device, val_ur, writer=writer, epoch=0,
                                        tune=tune, populary_dict=None)
    best_hr = res[10][0]
    best_ndcg = res[10][1]
    best_hr_20 = res[20][1]
    best_ndcg_20 = res[20][1]
    early_stopping_counter = 0
    stop = False
    best_epoch = 0
    if args.not_early_stopping:
        print("IT WILL NEVER DO EARLY STOPPING!")
    fnl_metric = []
    for epoch in range(1, args.epochs + 1):
        if stop:
            print(f'PRINT BEST VALIDATION RESULTS (ndcg optimization) on epoch {best_epoch}:')
            # print(best_res)
            break
        if args.neg_sampling_each_epoch:
            train_loader.dataset._neg_sampling()

        if tune:
            pbar = train_loader
        else:
            # set process bar display
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')

        model.train()

        for i, (user, item_i, context, item_j, label) in enumerate(pbar):
            user = torch.LongTensor(user).to(device)
            item_i = torch.LongTensor(item_i).to(device)
            item_j = torch.LongTensor(item_j).to(device)
            # if isinstance(context, list) and context_flag:
            if context_flag:
                context = list(context)
                if isinstance(context[0], list):
                    context = [torch.LongTensor(c).to(device) for c in context]
                else:
                    context = torch.LongTensor(context).to(device)
            else:
                context = None
            label = torch.LongTensor(label).to(device)
            # context = context.to(device) if context_flag else None

            model.zero_grad()
            pred_i, pred_j = model(user, item_i, item_j, context)

            if args.loss_type == 'BPR':
                loss = -(pred_i - pred_j).sigmoid().log().sum()
            elif args.loss_type == 'HL':
                loss = torch.clamp(1 - (pred_i - pred_j) * label, min=0).sum()
            elif args.loss_type == 'TL':  # TOP1-loss
                loss = (pred_j - pred_i).sigmoid().mean() + pred_j.pow(2).sigmoid().mean()
            else:
                raise ValueError(f'Invalid loss type: {args.loss_type}')

            if args.reindex:
                # [torch.norm(l) for l in model.parameters()]
                pass
                # if args.gce:
                #     loss += model.reg_2 * model.embeddings.GCN_module.weight.norm()   # 3.6643
                # else:
                #     loss += model.reg_2 * model.embeddings.weight.norm()   # 3.6643
            else:
                loss += model.reg_1 * (model.embed_item.weight.norm(p=1) + model.embed_user.weight.norm(p=1))
                loss += model.reg_2 * (model.embed_item.weight.norm() + model.embed_user.weight.norm())

            if torch.isnan(loss):
                break
                # raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the recommender')

            loss.backward()
            optimizer.step()

            if not tune:
                pbar.set_postfix(loss=loss.item())
            if writer is not None:
                writer.add_scalar('loss/train', loss.item(), epoch * len(train_loader) + i)

        res, writer, tmp_pred_10 = perform_evaluation(loaders, candidates, model, args, device, val_ur, writer=writer,
                                                      epoch=epoch, tune=tune, populary_dict=None)

        if args.save_initial_weights and epoch == 1:
            dirname = f'weights/{args.dataset}'
            os.makedirs(dirname, exist_ok=True)
            # model.embeddings.weight
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(dirname, init_weight_path))

        if res[10][1] > best_ndcg:
            best_ndcg = res[10][1]
            best_hr = res[10][0]
            best_ndcg_20 = res[20][1]
            best_hr_20 = res[20][0]
            # best_disc_hr = res[10][2]
            best_res = res
            best_epoch = epoch
            early_stopping_counter = 0
            stop = False
            filename_weights = f'weights/{args.dataset}/best_weights/{args.algo_name}'
            os.makedirs(filename_weights, exist_ok=True)
            # model.embeddings.weight
            best_checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
        else:
            early_stopping_counter += 1
            if early_stopping_counter == 10 and not args.not_early_stopping:
                print('Satisfy early stop mechanism')
                stop = True
        fnl_metric.append(tmp_pred_10)
        if tune:
            print(f'[Epoch {epoch:03d} DONE]')

    if tune:
        fnl_metric = np.array(fnl_metric).mean(axis=0)   # hr , ndcg
        score = fnl_metric[1]

        # get final validation metrics result by average operation
        print('=' * 20, 'Metrics for All Validation', '=' * 20)
        print(f'HR@10: {fnl_metric[0]:.4f}')
        print(f'NDCG@10: {fnl_metric[1]:.4f}')
        # print(f'Discounted_HR@10: {fnl_metric[2]:.4f}')

        # record all tuning result and settings
        fnl_metric = [f'{mt:.4f}' for mt in fnl_metric]
        line = ','.join(fnl_metric) + f',{best_epoch},{args.num_ng},{args.factors},{args.num_heads},{args.dropout},{args.lr},{args.batch_size},' \
            f'{args.reg_1},{args.reg_2}' + '\n'
        f.write(line)
        f.flush()
        return -score
    else:
        print('')
        print(f'++++++++++ BEST METRICS (epoch {best_epoch}) +++++++++')
        print(f'HR@10: {best_hr:.4f}')
        print(f'NDCG@10: {best_ndcg:.4f}')
        print(f'HR@20: {best_hr_20:.4f}')
        print(f'NDCG@20: {best_ndcg_20:.4f}')
        # print(f'Discounted_HR@10: {best_disc_hr:.4f}')
        name = f'best_epoch={best_epoch}_{desc}.pkl'
        # torch.save(best_checkpoint, os.path.join(filename_weights, name))


