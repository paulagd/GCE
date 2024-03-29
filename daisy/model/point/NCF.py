import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from daisy.model.GCE.gce import GCE, FactorizationMachine
from IPython import embed


class NCF(nn.Module):

    def __init__(self, user_num, max_dim, factors, layers=[128, 64, 32, 8], reindex=False, GCE_flag=False, X=None,
                 A=None, mf=False, gpuid='0', dropout=0):
        super(NCF, self).__init__()
        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.fc_layers = torch.nn.ModuleList()
        self.finalLayer = torch.nn.Linear(layers[-1], 1)
        
        self.reindex = reindex
        self.GCE_flag = GCE_flag
        self.mf_flag = mf
        self.dropout = dropout

        if reindex:
            self.fm = FactorizationMachine(reduce_sum=False)
            if self.GCE_flag:
                print('GCE EMBEDDINGS DEFINED')
                self.embeddings = GCE(max_dim, factors, X, A)
            else:
                self.embeddings = nn.Embedding(max_dim, factors)
        else:
            self.uEmbd = nn.Embedding(user_num, factors)
            self.iEmbd = nn.Embedding(max_dim, factors)

        for From, To in zip(layers[:-1], layers[1:]):
            self.fc_layers.append(nn.Linear(From, To))

    def forward(self, u, i, c):

        if self.reindex:
            if c is None:
                embeddings = self.embeddings(torch.stack((u, i), dim=1))
                #torch.Size([256, 128])
            else:
                embeddings = self.embeddings(torch.stack((u, i, c), dim=1))
            x = embeddings.view(embeddings.shape[0], -1)

        else:
            uembd = self.uEmbd(u)
            iembd = self.iEmbd(i)
            x = torch.cat([uembd, iembd], dim=1)
            
        for l in self.fc_layers:
            x = l(x)
            x = nn.ReLU()(x)

        prediction = self.finalLayer(x)
        return prediction.flatten()

    def predict(self, u, i, c):
        pred = self.forward(u, i, c).cpu()

        return pred