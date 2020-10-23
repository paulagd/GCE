import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from IPython import embed


def train(args, model, train_loader, device):

    model.to(device)
    if args.algo_name == 'nfm': #args.optimizer == '':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, initial_accumulator_value=1e-8)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    last_loss = 0.
    early_stopping_counter = 0
    stop = False
    model.train()

    for epoch in range(1, args.epochs + 1):
        if stop:
            break
        current_loss = 0.
        # set process bar display
        pbar = tqdm(train_loader)
        pbar.set_description(f'[Epoch {epoch:03d}]')
        for user, item_i, item_j, label in pbar:

            user = user.to(device)
            item_i = item_i.to(device)
            item_j = item_j.to(device)
            label = label.to(device)

            model.zero_grad()
            pred_i, pred_j = model(user, item_i, item_j)

            if args.loss_type == 'BPR':
                loss = -(pred_i - pred_j).sigmoid().log().sum()
            elif args.loss_type == 'HL':
                loss = torch.clamp(1 - (pred_i - pred_j) * label, min=0).sum()
            elif args.loss_type == 'TL':  # TOP1-loss
                loss = (pred_j - pred_i).sigmoid().mean() + pred_j.pow(2).sigmoid().mean()
            else:
                raise ValueError(f'Invalid loss type: {args.loss_type}')

            loss += model.reg_1 * (model.embed_item.weight.norm(p=1) + model.embed_user.weight.norm(p=1))
            loss += model.reg_2 * (model.embed_item.weight.norm() + model.embed_user.weight.norm())

            if torch.isnan(loss):
                raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the recommender')

            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
            current_loss += loss.item()

        if (last_loss < current_loss):
            early_stopping_counter += 1
            if early_stopping_counter == 10:
                print('Satisfy early stop mechanism')
                stop = True
        else:
            early_stopping_counter = 0
        last_loss = current_loss

        # TODO: EVALUATION OF VALIDATION AND RETURN METRICS
        # TODO: TENSORBOARD HR and NDCG
        # TODO: TENSORBOARD LOSS last_loss
