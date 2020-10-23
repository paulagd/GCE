import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn


class PairFM(nn.Module):
    def __init__(self,
                 user_num, 
                 item_num, 
                 factors=84, 
                 epochs=20, 
                 lr=0.001, 
                 reg_1=0.,
                 reg_2=0.,
                 loss_type='BPR',
                 gpuid='0', 
                 early_stop=True):
        """
        Pair-wise FM Recommender Class
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        factors : int, the number of latent factor
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(PairFM, self).__init__()

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.epochs = epochs
        self.lr = lr
        self.reg_1 = reg_1
        self.reg_2 = reg_2

        self.embed_user = nn.Embedding(user_num, factors)
        self.embed_item = nn.Embedding(item_num, factors)

        self.u_bias = nn.Embedding(user_num, 1)
        self.i_bias = nn.Embedding(item_num, 1)

        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        # init weight
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        nn.init.constant_(self.u_bias.weight, 0.0)
        nn.init.constant_(self.i_bias.weight, 0.0)

        self.loss_type = loss_type
        self.early_stop = early_stop

    def forward(self, u, i, j):
        user = self.embed_user(u)
        item_i = self.embed_item(i)
        item_j = self.embed_item(j)

        # inner product part
        pred_i = (user * item_i).sum(dim=-1, keepdim=True)
        pred_j = (user * item_j).sum(dim=-1, keepdim=True)

        # add bias
        pred_i += self.u_bias(u) + self.i_bias(i) + self.bias_
        pred_j += self.u_bias(u) + self.i_bias(j) + self.bias_

        return pred_i.view(-1), pred_j.view(-1)

    def predict(self, u, i):
        pred_i, _ = self.forward(u, i, i)

        return pred_i.cpu()