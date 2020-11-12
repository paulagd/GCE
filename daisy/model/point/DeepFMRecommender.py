import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn


class PointDeepFM(nn.Module):
    def __init__(self,
                 user_num, 
                 max_dim,
                 factors, 
                 act_activation,
                 num_layers,  # [32, 32] for example
                 batch_norm,
                 q, 
                 epochs, 
                 lr, 
                 reg_1=0.,
                 reg_2=0.,
                 loss_type='CL',
                 optimizer='adam',
                 gpuid='0', 
                 early_stop=True):
        """
        Point-wise DeepFM Recommender Class
        Parameters
        ----------
        user_num : int, the number of users
        max_dim : int, the number of items
        factors : int, the number of latent factor
        act_activation : str, activation function for hidden layer
        num_layers : int, number of hidden layers
        batch_norm : bool, whether to normalize a batch of data
        q : float, dropout rate
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(PointDeepFM, self).__init__()
        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.factors = factors
        self.act_function = act_activation
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.dropout = q
        self.epochs = epochs
        self.lr = lr
        self.reg_1 = reg_1
        self.reg_2 = reg_2
        self.loss_type = loss_type
        self.early_stop = early_stop
        self.optimizer = optimizer

        self.embed_user = nn.Embedding(user_num, factors)
        self.embed_item = nn.Embedding(max_dim, factors)

        self.u_bias = nn.Embedding(user_num, 1)
        self.i_bias = nn.Embedding(max_dim, 1)

        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        fm_modules = []
        if self.batch_norm:
            fm_modules.append(nn.BatchNorm1d(factors))
        fm_modules.append(nn.Dropout(self.dropout))
        self.fm_layers = nn.Sequential(*fm_modules)

        deep_modules = []
        in_dim = factors * 2   # user & item
        for _ in range(self.num_layers):  # _ is dim if layers is list
            out_dim = in_dim
            deep_modules.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            if self.batch_norm:
                deep_modules.append(nn.BatchNorm1d(out_dim))
            if self.act_function == 'relu':
                deep_modules.append(nn.ReLU())
            elif self.act_function == 'sigmoid':
                deep_modules.append(nn.Sigmoid())
            elif self.act_function == 'tanh':
                deep_modules.append(nn.Tanh())
            deep_modules.append(nn.Dropout(self.dropout))

        self.deep_layers = nn.Sequential(*deep_modules)
        self.deep_out = nn.Linear(in_dim, 1, bias=False)

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.embed_item.weight, std=0.01)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.constant_(self.u_bias.weight, 0.0)
        nn.init.constant_(self.i_bias.weight, 0.0)

        # for deep layers
        for m in self.deep_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        nn.init.xavier_normal_(self.deep_out.weight)

    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)

        fm = embed_user * embed_item
        fm = self.fm_layers(fm)
        y_fm = fm.sum(dim=-1)

        y_fm = y_fm + self.u_bias(user) + self.i_bias(item) + self.bias_

        if self.num_layers:
            fm = self.deep_layers(fm)

        y_deep = torch.cat((embed_user, embed_item), dim=-1)
        y_deep = self.deep_layers(y_deep)

        # since BCELoss will automatically transfer pred with sigmoid
        # there is no need to use extra nn.Sigmoid(pred)
        pred = y_fm + y_deep

        return pred.view(-1)

    def fit(self, train_loader):
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr)

        elif self.optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f'Invalid OPTIMIZER : {self.loss_type}')

        if self.loss_type == 'CL':
            criterion = nn.BCEWithLogitsLoss(reduction='sum')
        elif self.loss_type == 'SL':
            criterion = nn.MSELoss(reduction='sum')
        else:
            raise ValueError(f'Invalid loss type: {self.loss_type}')

        last_loss = 0.
        early_stopping_counter = 0
        for epoch in range(1, self.epochs + 1):
            self.train()

            current_loss = 0.
            # set process bar display
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')
            for user, item, label in pbar:
                if torch.cuda.is_available():
                    user = user.cuda()
                    item = item.cuda()
                    label = label.cuda()
                else:
                    user = user.cpu()
                    item = item.cpu()
                    label = label.cpu()

                self.zero_grad()
                prediction = self.forward(user, item)

                loss = criterion(prediction, label)
                loss += self.reg_1 * (self.embed_item.weight.norm(p=1) +self.embed_user.weight.norm(p=1))
                loss += self.reg_2 * (self.embed_item.weight.norm() +self.embed_user.weight.norm())

                if torch.isnan(loss):
                    raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the recommender')

                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
                current_loss += loss.item()

            self.eval()
            if (last_loss < current_loss) and self.early_stop:
                early_stopping_counter += 1
                if early_stopping_counter == 4:
                    print('Satisfy early stop mechanism')
                    break
            else:
                early_stopping_counter = 0
            last_loss = current_loss

    def predict(self, u, i):
        pred = self.forward(u, i).cpu()
        
        return pred


