import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from daisy.model.GCE.gce import GCE, FactorizationMachine
from IPython import embed


# class PairNCF(nn.Module):
#
#     def __init__(self, user_num, max_dim, factors, layers=[64,32,16,8], reindex=False, GCE_flag=False, X=None,
#                  A=None, mf=False, gpuid='0', dropout=0):
#         super(PairNCF, self).__init__()
# 

class PairNCF(nn.Module):
    
    def __init__(self, user_num, max_dim, factor_num, num_layers=3, dropout=0, model='NeuMF-end', gpuid='0',
                 GMF_model=None, MLP_model=None, reindex=False, GCE_flag=False, X=None, A=None, mf=False, num_context=0):
        super(PairNCF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors;
        num_layers: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
        GMF_model: pre-trained GMF weights;
        MLP_model: pre-trained MLP weights.
        """
        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True
        self.reindex = reindex
        self.GCE_flag = GCE_flag
        self.mf_flag = mf
        self.dropout = dropout
        self.num_context = num_context    # TODO: EXTEND FOR MORE THAN 1 CONTEXT

        if reindex:
            self.fm = FactorizationMachine(reduce_sum=False)

            if self.GCE_flag:
                print('GCE EMBEDDINGS DEFINED')
                self.embeddings = GCE(max_dim, factor_num, X, A)
                self.embeddings_MLP = GCE(max_dim, factor_num * (2 ** (num_layers - 1)), X, A)
            else:
                self.embeddings = nn.Embedding(max_dim, factor_num)
                self.embeddings_MLP = nn.Embedding(max_dim, factor_num * (2 ** (num_layers - 1)))
        else:
            self.embed_user_GMF = nn.Embedding(user_num, factor_num)
            self.embed_item_GMF = nn.Embedding(max_dim, factor_num)
            self.embed_user_MLP = nn.Embedding(
                    user_num, factor_num * (2 ** (num_layers - 1)))
            self.embed_item_MLP = nn.Embedding(
                max_dim, factor_num * (2 ** (num_layers - 1)))

        MLP_modules = []

        for i in range(num_layers):

            # input_size = factor_num * ((2 + self.num_context) ** (num_layers - i))
            input_size = int(factor_num + (factor_num/2)*self.num_context) * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            if i == num_layers-1:
                MLP_modules.append(nn.Linear(input_size, factor_num))
            else:
                MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)
        # self.predict_layer = nn.Linear(factor_num * (2 + self.num_context), 1)

        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        if not self.model == 'NeuMF-pre':
            if self.reindex:
                if not self.GCE_flag:
                    nn.init.normal_(self.embeddings.weight, std=0.01)
                    nn.init.normal_(self.embeddings_MLP.weight, std=0.01)
                else:
                    pass
            else:
                nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
                nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
                nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
                nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, user, item_i, item_j, context, inference=False):
        pred_i = self._out(user, item_i, context)
        if not inference:
            pred_j = self._out(user, item_j, context)
        else:
            pred_j = pred_i

        return pred_i, pred_j

    def _out(self, u, i, c):

        if self.reindex:
            if c is None:
                embeddings = self.embeddings(torch.stack((u, i), dim=1))
                embeddings_MLP = self.embeddings_MLP(torch.stack((u, i), dim=1))
            else:
                if isinstance(c, list) and len(c) > 0:
                    context_mean_emb = torch.mean(self.embeddings(torch.stack(c, dim=1)), dim=1).unsqueeze(dim=1)
                    context_mean_emb_MLP = torch.mean(self.embeddings_MLP(torch.stack(c, dim=1)), dim=1).unsqueeze(dim=1)
                    # embeddings = self.embeddings(torch.cat((torch.stack((u, i), dim=1), context), dim=1))
                    embeddings = torch.cat((self.embeddings(torch.stack((u, i), dim=1)), context_mean_emb), dim=1)
                    embeddings_MLP = torch.cat((self.embeddings_MLP(torch.stack((u, i), dim=1)), context_mean_emb_MLP),
                                               dim=1)
                    # embeddings = self.embeddings(torch.cat((torch.stack((u, i), dim=1), context), dim=1))
                    # embeddings_MLP = self.embeddings_MLP(torch.cat((torch.stack((u, i), dim=1), context), dim=1))
                else:
                    embeddings = self.embeddings(torch.stack((u, i, c), dim=1))
                    embeddings_MLP = self.embeddings_MLP(torch.stack((u, i, c), dim=1))
            output_GMF = self.fm(embeddings)
            # output_MLP = embeddings_MLP.view(embeddings_MLP.shape[0], -1)
            # embed()
            output_MLP = self.MLP_layers(embeddings_MLP.view(output_GMF.shape[0], -1))

        else:
            if not self.model == 'MLP':
                embed_user_GMF = self.embed_user_GMF(u)
                embed_item_GMF = self.embed_item_GMF(i)
                output_GMF = embed_user_GMF * embed_item_GMF
            if not self.model == 'GMF':
                embed_user_MLP = self.embed_user_MLP(u)
                embed_item_MLP = self.embed_item_MLP(i)
                interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
                output_MLP = self.MLP_layers(interaction)

        concat = torch.cat((output_GMF, output_MLP), -1)
        prediction = self.predict_layer(concat)
        return prediction.view(-1)

    def predict(self, u, i, c):
        pred_i, _ = self.forward(u, i, i, c, inference=True)

        return pred_i.cpu()
