import os
from tqdm import tqdm

import torch
import torch.nn as nn
from scipy.sparse import coo_matrix
from scipy.sparse import vstack
from scipy import sparse
import numpy as np
from daisy.model.GCE.gce import GCE
from IPython import embed


class GNNLayer(nn.Module):

    def __init__(self,inF,outF):

        super(GNNLayer,self).__init__()
        self.inF = inF
        self.outF = outF
        self.linear = torch.nn.Linear(in_features=inF, out_features=outF)
        self.interActTransform = torch.nn.Linear(in_features=inF, out_features=outF)

    def forward(self, laplacianMat,selfLoop,features):
        # for GCF ajdMat is a (N+M) by (N+M) mat
        # laplacianMat L = D^-1(A)D^-1 # 拉普拉斯矩阵
        L1 = laplacianMat + selfLoop
        L2 = laplacianMat.cuda()
        L1 = L1.cuda()
        inter_feature = torch.sparse.mm(L2,features)
        inter_feature = torch.mul(inter_feature,features)

        inter_part1 = self.linear(torch.sparse.mm(L1,features))
        inter_part2 = self.interActTransform(torch.sparse.mm(L2,inter_feature))

        return inter_part1+inter_part2


class NGCF(nn.Module):

    def __init__(self, userNum, itemNum, df, embedSize=100, layers=[100, 80, 50], useCuda=True):

        super(NGCF, self).__init__()
        self.useCuda = useCuda
        self.userNum = userNum
        self.itemNum = itemNum
        self.uEmbd = nn.Embedding(userNum, embedSize)
        self.iEmbd = nn.Embedding(itemNum, embedSize)
        self.GNNlayers = torch.nn.ModuleList()
        self.LaplacianMat = self.buildLaplacianMat(df) # sparse format
        self.leakyRelu = nn.LeakyReLU()
        self.selfLoop = self.getSparseEye(self.userNum+self.itemNum)

        # self.transForm1 = nn.Linear(in_features=layers[-1]*(len(layers))*2, out_features=64)
        self.transForm1 = nn.Linear(in_features=460, out_features=64)
        self.transForm2 = nn.Linear(in_features=64, out_features=32)
        self.transForm3 = nn.Linear(in_features=32, out_features=1)

        for From, To in zip(layers[:-1], layers[1:]):
            self.GNNlayers.append(GNNLayer(From, To))

    def getSparseEye(self, num):
        i = torch.LongTensor([[k for k in range(0,num)],[j for j in range(0,num)]])
        val = torch.FloatTensor([1]*num)
        return torch.sparse.FloatTensor(i,val)

    def buildLaplacianMat(self, rt):

        rt_item = rt['item'] + self.userNum
        uiMat = coo_matrix((rt['rating'], (rt['user'], rt['item'])))

        uiMat_upperPart = coo_matrix((rt['rating'], (rt['user'], rt_item)))
        uiMat = uiMat.transpose()
        uiMat.resize((self.itemNum, self.userNum + self.itemNum))

        A = sparse.vstack([uiMat_upperPart,uiMat])
        selfLoop = sparse.eye(self.userNum+self.itemNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag, -0.5)
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i,data)
        return SparseL

    def getFeatureMat(self):
        uidx = torch.LongTensor([i for i in range(self.userNum)])
        iidx = torch.LongTensor([i for i in range(self.itemNum)])
        if self.useCuda == True:
            uidx = uidx.cuda()
            iidx = iidx.cuda()

        userEmbd = self.uEmbd(uidx)
        itemEmbd = self.iEmbd(iidx)
        features = torch.cat([userEmbd, itemEmbd], dim=0)
        return features

    def forward(self, userIdx, itemIdx, c):

        userIdx = list(userIdx.cpu().data)
        itemIdx = list(itemIdx.cpu().data)
        # gcf data propagation
        features = self.getFeatureMat()
        finalEmbd = features.clone()
        for gnn in self.GNNlayers:
            features = gnn(self.LaplacianMat, self.selfLoop, features)
            features = nn.ReLU()(features)
            finalEmbd = torch.cat([finalEmbd, features.clone()], dim=1)

        userEmbd = finalEmbd[userIdx]
        itemEmbd = finalEmbd[itemIdx]
        embd = torch.cat([userEmbd, itemEmbd], dim=1)
        embd = nn.ReLU()(self.transForm1(embd))
        embd = self.transForm2(embd)
        embd = self.transForm3(embd)
        prediction = embd.flatten()

        return prediction

    def predict(self, u, i, c):
        pred = self.forward(u, i, c).cpu()

        return pred