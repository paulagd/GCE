import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from IPython import embed
from daisy.model.GCE.gce import GCE


class PointNeuMF(nn.Module):
    def __init__(self,
                 user_num, 
                 max_dim,
                 factors, 
                 num_layers, 
                 dropout,
                 lr, 
                 epochs,
                 optimizer='adam',
                 reg_1=0.001,
                 reg_2=0.001, 
                 loss_type='CL', 
                 model_name='NeuMF-end',
                 GMF_model=None,  # generalized matrix factorization
                 MLP_model=None,  # models the interactions from two pathways instead of simple inner products
                 gpuid='0',
                 reindex=False,
                 GCE_flag=False,
                 early_stop=True):
        """
        Point-wise NeuMF Recommender Class
        Parameters
        ----------
        user_num : int, number of users;
        item_num : int, number of items;
        factors : int, the number of latent factor
        num_layers : int, number of hidden layers
        q : float, dropout rate
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        model_name : str, model name
        GMF_model : Object, pre-trained GMF weights;
        MLP_model : Object, pre-trained MLP weights.
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(PointNeuMF, self).__init__()
        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.lr = lr
        self.epochs = epochs
        self.reg_1 = reg_1
        self.reg_2 = reg_2
        self.optimizer = optimizer

        self.dropout = dropout
        self.model = model_name
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model
        self.reindex = reindex
        self.GCE_flag = GCE_flag

        if reindex:
            self.embed_GMF = nn.Embedding(max_dim, factors)
            self.embed_MLP = nn.Embedding(max_dim, factors * (2 ** (num_layers - 1)))

        else:
            self.embed_user_GMF = nn.Embedding(user_num, factors)
            self.embed_item_GMF = nn.Embedding(max_dim, factors)

            self.embed_user_MLP = nn.Embedding(user_num, factors * (2 ** (num_layers - 1)))
            self.embed_item_MLP = nn.Embedding(max_dim, factors * (2 ** (num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factors * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factors
        else:
            predict_size = factors * 2

        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

        self.loss_type = loss_type
        self.early_stop = early_stop

    def _init_weight_(self):
        '''weights initialization'''
        if not self.model == 'NeuMF-pre':
            print(f'MODEL NEUMF == {self.model}')
            if self.reindex:
                nn.init.normal_(self.embed_GMF.weight, std=0.01)
                nn.init.normal_(self.embed_MLP.weight, std=0.01)
            else:
                nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
                nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
                nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
                nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight, 
                                     a=1, nonlinearity='sigmoid')
            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()

        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item_MLP.weight)

            # mlp layers
            for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat([self.GMF_model.predict_layer.weight, 
                                        self.MLP_model.predict_layer.weight], dim=1)
            predict_bias = self.GMF_model.predict_layer.bias + self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.weight.data.copy_(0.5 * predict_bias)

    def forward(self, user, item):
        
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF

        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), dim=-1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            # embed()
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)

    def fit(self, train_loader):
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

        if self.loss_type == 'CL':
            criterion = nn.BCEWithLogitsLoss(reduction='sum')
        elif self.loss_type == 'SL':
            criterion = nn.MSELoss(reduction='sum')
        else:
            raise ValueError(f'Invalid loss type: {self.loss_type}')

        if self.model == 'NeuMF-pre':
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.lr)

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
                if self.reindex:
                    # TODO: implement regularization for reindexed items
                    pass
                else:
                    loss += self.reg_1 * (self.embed_item_GMF.weight.norm(p=1) + self.embed_user_GMF.weight.norm(p=1))
                    loss += self.reg_1 * (self.embed_item_MLP.weight.norm(p=1) + self.embed_user_MLP.weight.norm(p=1))

                    loss += self.reg_2 * (self.embed_item_GMF.weight.norm() + self.embed_user_GMF.weight.norm())
                    loss += self.reg_2 * (self.embed_item_MLP.weight.norm() + self.embed_user_MLP.weight.norm())

                if torch.isnan(loss):
                    raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the recommender')

                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
                current_loss += loss.item()
            
            self.eval()
            # delta_loss = float(current_loss - last_loss)
            # if (abs(delta_loss) < 1e-5) and self.early_stop:
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
