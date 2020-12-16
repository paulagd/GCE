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

    last_loss = 0.
    early_stopping_counter = 0
    stop = False
    model.train()
    print(f'RUN FOR {args.epochs} EPOCHS')
    for epoch in range(1, args.epochs + 1):
        if stop:
            break
        current_loss = 0.
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

        # perform_evaluation(loaders, candidates, model, writer=None, epoch=None)
        perform_evaluation(loaders, candidates, model, args, device, val_ur, writer=writer, epoch=epoch)
