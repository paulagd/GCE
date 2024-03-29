import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from IPython import embed
from daisy.model.GCE.gce import GCE, FactorizationMachine


class PointFM(nn.Module):
    def __init__(self, 
                 user_num, 
                 max_dim,
                 factors=84, 
                 epochs=20,
                 optimizer='adam',
                 lr=0.001,
                 reg_1=0.001,
                 reg_2=0.001,
                 loss_type='SL',
                 gpuid='0',
                 reindex=False,
                 X=None,
                 A=None,
                 GCE_flag=False,
                 dropout=0,
                 early_stop=True):
        """
        Point-wise FM Recommender Class
        Parameters
        ----------
        user_num : int, the number of users
        max_dim : int, the number of items
        factors : int, the number of latent factor
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(PointFM, self).__init__()

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.epochs = epochs
        self.lr = lr
        self.optimizer = optimizer
        self.reg_1 = reg_1
        self.reg_2 = reg_2
        self.dropout = dropout

        self.reindex = reindex
        self.GCE_flag = GCE_flag
        self.fm = FactorizationMachine(reduce_sum=True)

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

            # init weight
            nn.init.normal_(self.embed_user.weight, std=0.01)
            nn.init.normal_(self.embed_item.weight, std=0.01)
            nn.init.constant_(self.u_bias.weight, 0.0)
            nn.init.constant_(self.i_bias.weight, 0.0)

        self.loss_type = loss_type

    def forward(self, user, item, context):

        if self.reindex:
            if context is None:
                embeddings = self.embeddings(torch.stack((user, item), dim=1))
            else:
                embeddings = self.embeddings(torch.stack((user, item, context), dim=1))
                
            nn.functional.dropout(embeddings, p=self.dropout, training=self.training, inplace=True)
            # pred = embeddings.prod(dim=1).sum(dim=1, keepdim=True)
            pred = self.fm(embeddings)

            if not self.GCE_flag:
                pred += self.bias(torch.stack((user, item), dim=1)).sum() + self.bias_
            # return torch.squeeze(ix)
            return pred.view(-1)
        else:
            embed_user = self.embed_user(user)
            embed_item = self.embed_item(item)

            pred = (embed_user * embed_item).sum(dim=-1, keepdim=True)
            pred += self.u_bias(user) + self.i_bias(item) + self.bias_

            return pred.view(-1)

    def predict(self, u, i, c):
        pred = self.forward(u, i, c).cpu()
        
        return pred
