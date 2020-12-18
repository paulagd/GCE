import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from IPython import embed
import torch.backends.cudnn as cudnn
from daisy.utils.splitter import perform_evaluation


def train(args, model, train_loader, device, context_flag, writer, loaders, candidates, val_ur):
    cudnn.benchmark = True

    model.to(device)
    if args.optimizer == 'adagrad': #args.optimizer == '':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, initial_accumulator_value=1e-8)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    print(f'RUN FOR {args.epochs} EPOCHS')
    # IDEA: RANDOM EVALUATION
    res = perform_evaluation(loaders, candidates, model, args, device, val_ur, writer=writer, epoch=0)
    # best_hr = res[10][0]
    best_ndcg = res[10][1]
    early_stopping_counter = 0
    stop = False
    best_epoch = 0
    if not args.not_early_stopping:
        print("IT WILL NEVER DO EARLY STOPPING!")
    for epoch in range(1, args.epochs + 1):
        if stop and not args.not_early_stopping:
            print(f'PRINT BEST VALIDATION RESULTS (ndcg optimization) on epoch {best_epoch}:')
            print(best_res)
            break
        if args.neg_sampling_each_epoch:
            train_loader.dataset._neg_sampling()
        # set process bar display
        pbar = tqdm(train_loader)
        pbar.set_description(f'[Epoch {epoch:03d}]')
        for i, (user, item_i, context, item_j, label) in enumerate(pbar):
            user = user.to(device)
            item_i = item_i.to(device)
            item_j = item_j.to(device)
            context = context.to(device) if context_flag else None
            label = label.to(device)
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
                raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the recommender')

            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
            writer.add_scalar('loss/train', loss.item(), epoch * len(train_loader) + i)

        res = perform_evaluation(loaders, candidates, model, args, device, val_ur, writer=writer, epoch=epoch)

        if res[10][1] > best_ndcg:
        # if res[10][0] > best_hr:
            best_ndcg = res[10][1]
            # best_hr = res[10][0]
            best_res = res
            best_epoch = epoch
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter == 10:
                print('Satisfy early stop mechanism')
                stop = True
