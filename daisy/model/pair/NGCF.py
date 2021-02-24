import os
from tqdm import tqdm

import torch
import torch.optim as optim
from daisy.model.GCE.gce import GCE, FactorizationMachine
from IPython import embed
import numpy as np
import torch.nn as nn
#from torch.nn import init, LeakyReLU, Linear, Module, ModuleList, Embedding, Parameter
import torch.nn.functional as F
import scipy.sparse as sp


def to_sparse_tensor(X):
  """
  Convert a sparse numpy object to a sparse pytorch tensor.
  Note that the tensor does not yet live on the GPU
  """
  coo = X.tocoo().astype(np.float32)
  i = torch.LongTensor(np.mat((coo.row, coo.col)))
  v = torch.FloatTensor(coo.data)
  return torch.sparse.FloatTensor(i, v, coo.shape)


def split_mtx(X, n_folds=200):
  """
  Split a matrix/Tensor into n parts.
  Useful for processing large matrices in batches
  """
  X_folds = []
  fold_len = X.shape[0]//n_folds
  for i in range(n_folds):
    start = i * fold_len
    if i == n_folds -1:
      end = X.shape[0]
    else:
      end = (i + 1) * fold_len
    X_folds.append(X[start:end])
  return X_folds


class PairNGCF(nn.Module):
    def __init__(self, n_users, max_dim, emb_dim, adj_mtx, device, reindex=True, reg=1e-5, layers=[64,64],
                 node_dropout=0, mess_dropout=0.1):
        # def __init__(self, n_users, n_items, embed_size, adj_matrix, device, reindex, n_layers=1):

        super().__init__()

        # initialize Class attributes
        self.n_users = n_users
        self.n_items = max_dim - n_users
        self.emb_dim = emb_dim
        self.adj_mtx = adj_mtx + sp.eye(adj_mtx.shape[0])  # + sp.dok_matrix(np.eye(adj_mtx.shape[0]), dtype=np.float32)
        self.laplacian = adj_mtx
        self.reg = reg
        self.layers = layers
        self.n_layers = len(self.layers)
        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout
        self.device = device
        self.reindex = reindex

        # self.u_g_embeddings = nn.Parameter(torch.empty(n_users, emb_dim+np.sum(self.layers)))
        # self.i_g_embeddings = nn.Parameter(torch.empty(n_items, emb_dim+np.sum(self.layers)))

        # Initialize weights
        self.weight_dict = self._init_weights()
        print("Weights initialized.")

        # Create Matrix 'A', PyTorch sparse tensor of SP adjacency_mtx
        self.A = self._convert_sp_mat_to_sp_tensor(self.adj_mtx)
        self.L = self._convert_sp_mat_to_sp_tensor(self.laplacian)

    # initialize weights
    def _init_weights(self):
        print("Initializing weights...")
        weight_dict = nn.ParameterDict()

        initializer = torch.nn.init.xavier_uniform_

        weight_dict['user_embedding'] = nn.Parameter(initializer(torch.empty(self.n_users, self.emb_dim).to(self.device)))
        weight_dict['item_embedding'] = nn.Parameter(initializer(torch.empty(self.n_items, self.emb_dim).to(self.device)))

        weight_size_list = [self.emb_dim] + self.layers

        for k in range(self.n_layers):
            weight_dict['W_gc_%d' % k] = nn.Parameter(
                initializer(torch.empty(weight_size_list[k], weight_size_list[k + 1]).to(self.device)))
            weight_dict['b_gc_%d' % k] = nn.Parameter(initializer(torch.empty(1, weight_size_list[k + 1]).to(self.device)))

            weight_dict['W_bi_%d' % k] = nn.Parameter(
                initializer(torch.empty(weight_size_list[k], weight_size_list[k + 1]).to(self.device)))
            weight_dict['b_bi_%d' % k] = nn.Parameter(initializer(torch.empty(1, weight_size_list[k + 1]).to(self.device)))

        return weight_dict

    # convert sparse matrix into sparse PyTorch tensor
    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        Convert scipy sparse matrix to PyTorch sparse matrix
        Arguments:
        ----------
        X = Adjacency matrix, scipy sparse matrix
        """
        coo = X.tocoo().astype(np.float32)
        i = torch.LongTensor(np.mat([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)
        res = torch.sparse.FloatTensor(i, v, coo.shape).to(self.device)
        return res

    # apply node_dropout
    def _droupout_sparse(self, X):
        """
        Drop individual locations in X

        Arguments:
        ---------
        X = adjacency matrix (PyTorch sparse tensor)
        dropout = fraction of nodes to drop
        noise_shape = number of non non-zero entries of X
        """

        node_dropout_mask = ((self.node_dropout) + torch.rand(X._nnz())).floor().bool().to(self.device)
        i = X.coalesce().indices()
        v = X.coalesce()._values()
        i[:, node_dropout_mask] = 0
        v[node_dropout_mask] = 0
        X_dropout = torch.sparse.FloatTensor(i, v, X.shape).to(X.device)

        return X_dropout.mul(1 / (1 - self.node_dropout))

    def forward(self, u, i, j, c=None, inference=False):
        pred_i, pred_j = self._out(u, i, j, c, inference)
        # if not inference:
        #     pred_i, pred_j = self._out(u, j, c)
        # else:
        #     pred_j = pred_i

        return pred_i, pred_j

    def _out(self, u, i, j, c=None, inference=False):
        """
        Computes the forward pass

        Arguments:
        ---------
        u = user
        i = positive item (user interacted with item)
        j = negative item (user did not interact with item)
        """
        # apply drop-out mask
        A_hat = self._droupout_sparse(self.A) if self.node_dropout > 0 else self.A
        L_hat = self._droupout_sparse(self.L) if self.node_dropout > 0 else self.L

        ego_embeddings = torch.cat([self.weight_dict['user_embedding'], self.weight_dict['item_embedding']], 0)

        all_embeddings = [ego_embeddings]

        # forward pass for 'n' propagation layers
        for k in range(self.n_layers):
            # weighted sum messages of neighbours
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            side_L_embeddings = torch.sparse.mm(L_hat, ego_embeddings)

            # transformed sum weighted sum messages of neighbours
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) + self.weight_dict[
                'b_gc_%d' % k]

            # bi messages of neighbours
            bi_embeddings = torch.mul(ego_embeddings, side_L_embeddings)
            # transformed bi messages of neighbours
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) + self.weight_dict[
                'b_bi_%d' % k]

            # non-linear activation
            ego_embeddings = F.leaky_relu(sum_embeddings + bi_embeddings)
            # + message dropout
            mess_dropout_mask = nn.Dropout(self.mess_dropout)
            ego_embeddings = mess_dropout_mask(ego_embeddings)

            # normalize activation
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings.append(norm_embeddings)

        all_embeddings = torch.cat(all_embeddings, 1)

        # back to user/item dimension
        u_g_embeddings, i_g_embeddings = all_embeddings.split([self.n_users, self.n_items], 0)

        self.u_g_embeddings = nn.Parameter(u_g_embeddings)
        self.i_g_embeddings = nn.Parameter(i_g_embeddings)
        if not inference:
            u_emb = u_g_embeddings[u]  # user embeddings
            p_emb = i_g_embeddings[i - self.n_users]  # positive item embeddings
            n_emb = i_g_embeddings[j - self.n_users]  # negative item embeddings

            y_ui = torch.mul(u_emb, p_emb).sum(dim=1)
            y_uj = torch.mul(u_emb, n_emb).sum(dim=1)
        else:
            u_emb = self.u_g_embeddings[u]  # user embeddings
            p_emb = self.i_g_embeddings[i - self.n_users]  # positive item embeddings

            y_ui = torch.mul(u_emb, p_emb).sum(dim=1)
            y_uj = y_ui

        return y_ui, y_uj

    def predict(self, u, i, c=None):
        pred_i, _ = self.forward(u, i, i, c=None, inference=True)
        return pred_i.cpu()


# class NGCF(Module):
#     def __init__(self, n_users, n_items, embed_size, adj_matrix, device, reindex, n_layers=1):
#         super().__init__()
#         self.n_users = n_users
#         self.max_dim = n_items
#         self.embed_size = embed_size
#         self.n_layers = n_layers
#         self.adj_matrix = adj_matrix
#         self.device = device
#         self.reindex = reindex
#         self.fm = FactorizationMachine(reduce_sum=True)
#
#         if self.reindex:
#             self.embeddings = Parameter(torch.rand(self.max_dim, self.embed_size))
#             self.embeddings_final = Parameter(torch.zeros((self.max_dim, self.embed_size * (n_layers + 1))))
#         else:
#             pass
#             # # The (user/item)_embeddings are the initial embedding matrix E
#             # self.user_embeddings = Embedding(n_users, self.embed_size)
#             # self.item_embeddings = Embedding(self.max_dim, self.embed_size)
#             # # The (user/item)_embeddings_final are the final concatenated embeddings [E_1..E_L]
#             # # Stored for easy tracking of final embeddings throughout optimization and eval
#             # self.user_embeddings_final = Embedding(n_users, self.embed_size * (n_layers + 1))
#             # self.item_embeddings_final = Embedding(self.max_dim, self.embed_size * (n_layers + 1))
#
#             # # The (user/item)_embeddings are the initial embedding matrix E
#             # self.user_embeddings = Parameter(torch.rand(n_users, self.embed_size))
#             # self.item_embeddings = Parameter(torch.rand(self.max_dim, self.embed_size))
#             # # The (user/item)_embeddings_final are the final concatenated embeddings [E_1..E_L]
#             # # Stored for easy tracking of final embeddings throughout optimization and eval
#             # self.user_embeddings_final = Parameter(torch.zeros((n_users, self.embed_size * (n_layers + 1))))
#             # self.item_embeddings_final = Parameter(torch.zeros((self.max_dim, self.embed_size * (n_layers + 1))))
#
#         # The linear transformations for each layer
#         self.W1 = ModuleList([Linear(self.embed_size, self.embed_size) for _ in range(0, self.n_layers)])
#         self.W2 = ModuleList([Linear(self.embed_size, self.embed_size) for _ in range(0, self.n_layers)])
#         self.act = LeakyReLU()
#
#         # Initialize each of the trainable weights with the Xavier initializer
#         self.init_weights()
#
#     def init_weights(self):
#         for name, parameter in self.named_parameters():
#             if ('bias' not in name):
#                 init.xavier_uniform_(parameter)
#
#     def forward(self, u, i, j, c, inference=False):
#         pred_i = self._out(u, i, c)
#         if not inference:
#             pred_j = self._out(u, j, c)
#         else:
#             pred_j = pred_i
#
#         return pred_i, pred_j
#
#     def _out(self, u, i, c=None):
#         # adj_splits = split_mtx(self.adj_matrix, n_folds=1)
#         if self.reindex:
#             embeddings = self.embeddings
#             # embeddings = embeddings.view((-1, self.embed_size))
#         else:
#             embeddings = torch.cat((self.user_embeddings, self.item_embeddings))
#
#         final_embeddings = [embeddings]
#
#         for l in range(self.n_layers):
#             # embedding_parts = []
#             # for part in adj_splits:
#             #     embedding_parts.append(torch.sparse.mm(to_sparse_tensor(part).to(self.device), embeddings))
#
#             # Message construction
#             # t1_embeddings = torch.cat(embedding_parts, 0)
#             t1_embeddings = torch.sparse.mm(to_sparse_tensor(self.adj_matrix).to(self.device), embeddings)
#             t1 = self.W1[l](t1_embeddings)
#             t2_embeddings = embeddings.mul(t1_embeddings)
#             t2 = self.W2[l](t2_embeddings)
#
#             # Message aggregation
#             embeddings = self.act(t1 + t2)
#             normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
#             final_embeddings.append(normalized_embeddings)
#
#         # Make sure to update the (user/item)_embeddings(_final)
#         final_embeddings = torch.cat(final_embeddings, 1)
#         if not self.reindex:
#             final_u_embeddings, final_i_embeddings = final_embeddings.split((self.n_users, self.n_items), 0)
#             self.user_embeddings_final = Parameter(final_u_embeddings)
#             self.item_embeddings_final = Parameter(final_i_embeddings)
#             return torch.mul(final_u_embeddings[u], final_i_embeddings[i]).sum(dim=1)
#         else:
#             self.embeddings_final = Parameter(final_embeddings)
#             if c is not None:
#                 if isinstance(c, list) and len(c) > 0:
#                     # context = torch.stack(c, dim=1)
#                     context_embedding = []
#                     for context in c:
#                         context_embedding.append(torch.mean(self.embeddings_final[context], dim=0))
#                     context_embedding = torch.stack(context_embedding)
#                     # fm_input = torch.cat((torch.stack((self.embeddings_final[u], self.embeddings_final[i]), dim=1),
#                     # context_embedding), dim=1)
#                     fm_input = torch.stack((self.embeddings_final[u], self.embeddings_final[i], context_embedding), dim=1)
#                     return self.fm(fm_input).squeeze()
#                 else:
#                     fm_input = torch.stack((self.embeddings_final[u], self.embeddings_final[i], self.embeddings_final[c]), dim=1)
#                     return self.fm(fm_input).squeeze()
#             else:
#                 return torch.mul(self.embeddings_final[u], self.embeddings_final[i]).sum(dim=1)
#
#     def predict(self, u, i, c):
#         pred_i, _ = self.forward(u, i, i, c, inference=True)
#
#         return pred_i.cpu()
