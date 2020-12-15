import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from daisy.model.GCE.gce import GCE, FactorizationMachine, MultiLayerPerceptron
import torch.backends.cudnn as cudnn
from IPython import embed


class PairDeepFM(nn.Module):
    def __init__(self,
                 user_num, 
                 max_dim,
                 factors, 
                 act_activation,
                 num_layers,  # [32, 32] for example
                 batch_norm,
                 dropout,
                 epochs, 
                 lr, 
                 reg_1=0.,
                 reg_2=0.,
                 loss_type='CL', 
                 gpuid='0',
                 reindex=False,
                 X=None,
                 A=None,
                 GCE_flag=False,
                 early_stop=True,
                 context_flag=False,
                 mlp_dims=(16, 16),
                 mf=False,
                 ):
        """
        Pair-wise DeepFM Recommender Class
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
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
        super(PairDeepFM, self).__init__()
        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.factors = factors
        self.act_function = act_activation
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.reg_1 = reg_1
        self.reg_2 = reg_2
        self.loss_type = loss_type
        self.early_stop = early_stop
        self.mlp_dims = mlp_dims
        self.reindex = reindex
        self.GCE_flag = GCE_flag

        if reindex:
            if GCE_flag:
                print('GCE EMBEDDINGS DEFINED')
                self.embeddings = GCE(max_dim, factors, X, A)
            else:
                self.embeddings = nn.Embedding(max_dim, factors)
                self.bias = nn.Embedding(max_dim, 1)
                self.bias_ = nn.Parameter(torch.tensor([0.0]))

                nn.init.normal_(self.embeddings.weight, std=0.01)
                nn.init.constant_(self.bias.weight, 0.0)
        else:
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

        self.input_mlp = 3 * factors if context_flag else 2 * factors
        self.deep_layers = MultiLayerPerceptron(self.input_mlp, mlp_dims, self.dropout)
        self._init_weight()

    def _init_weight(self):
        if self.reindex and not self.GCE_flag:
            nn.init.normal_(self.embeddings.weight, std=0.01)
            nn.init.constant_(self.bias.weight, 0.0)
        elif not self.reindex:
            nn.init.normal_(self.embed_item.weight, std=0.01)
            nn.init.normal_(self.embed_user.weight, std=0.01)
            nn.init.constant_(self.u_bias.weight, 0.0)
            nn.init.constant_(self.i_bias.weight, 0.0)

        # # for deep layers
        # for m in self.deep_layers:
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)
        # nn.init.xavier_normal_(self.deep_out.weight)

    def forward(self, u, i, j, c, inference=False):
        pred_i = self._out(u, i, c)
        if not inference:
            pred_j = self._out(u, j, c)
        else:
            pred_j = pred_i

        return pred_i, pred_j

    def _out(self, user, item, context):
        if self.reindex:
            if context is None:
                embeddings = self.embeddings(torch.stack((user, item), dim=1))
            else:
                embeddings = self.embeddings(torch.stack((user, item, context), dim=1))
            fm = embeddings.prod(dim=1)  # shape [256, 32]
        else:
            embed_user = self.embed_user(user)
            embed_item = self.embed_item(item)
            fm = embed_user * embed_item

        fm = self.fm_layers(fm)
        y_fm = fm.sum(dim=-1)

        if self.reindex and not self.GCE_flag:
            y_fm += y_fm + self.bias_

        elif not self.GCE_flag:
            y_fm += y_fm + self.u_bias(user) + self.i_bias(item) + self.bias_

        # if self.num_layers:
        #     fm = self.deep_layers(fm)
        if self.reindex:
            y_deep = embeddings.view(-1, self.input_mlp)  # torch.Size([256, 192])
            y_deep = self.deep_layers(y_deep).squeeze()
        else:
            y_deep = torch.cat((embed_user, embed_item), dim=-1)
            y_deep = self.deep_layers(y_deep)

        # since BCELoss will automatically transfer pred with sigmoid
        # there is no need to use extra nn.Sigmoid(pred)
        pred = y_fm + y_deep

        return pred.view(-1)

    def predict(self, u, i, c):
        pred_i, _ = self.forward(u, i, i, c, inference=True)

        return pred_i.cpu()
