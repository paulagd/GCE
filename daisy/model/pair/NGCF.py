import os
from tqdm import tqdm

import torch
import torch.optim as optim
from daisy.model.GCE.gce import GCE, FactorizationMachine
from IPython import embed
import numpy as np
from torch.nn import init, LeakyReLU, Linear, Module, ModuleList, Embedding, Parameter
import torch.nn.functional as F


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


class PairNGCF(Module):
    def __init__(self, n_users, n_items, embed_size, adj_matrix, device, reindex, n_layers=1):
        super().__init__()
        self.n_users = n_users
        self.max_dim = n_items
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.adj_matrix = adj_matrix
        self.device = device
        self.reindex = reindex
        self.fm = FactorizationMachine(reduce_sum=True)

        if self.reindex:
            self.embeddings = Parameter(torch.rand(self.max_dim, self.embed_size))
            self.embeddings_final = Parameter(torch.zeros((self.max_dim, self.embed_size * (n_layers + 1))))
        else:
            pass
            # # The (user/item)_embeddings are the initial embedding matrix E
            # self.user_embeddings = Embedding(n_users, self.embed_size)
            # self.item_embeddings = Embedding(self.max_dim, self.embed_size)
            # # The (user/item)_embeddings_final are the final concatenated embeddings [E_1..E_L]
            # # Stored for easy tracking of final embeddings throughout optimization and eval
            # self.user_embeddings_final = Embedding(n_users, self.embed_size * (n_layers + 1))
            # self.item_embeddings_final = Embedding(self.max_dim, self.embed_size * (n_layers + 1))

            # # The (user/item)_embeddings are the initial embedding matrix E
            # self.user_embeddings = Parameter(torch.rand(n_users, self.embed_size))
            # self.item_embeddings = Parameter(torch.rand(self.max_dim, self.embed_size))
            # # The (user/item)_embeddings_final are the final concatenated embeddings [E_1..E_L]
            # # Stored for easy tracking of final embeddings throughout optimization and eval
            # self.user_embeddings_final = Parameter(torch.zeros((n_users, self.embed_size * (n_layers + 1))))
            # self.item_embeddings_final = Parameter(torch.zeros((self.max_dim, self.embed_size * (n_layers + 1))))

        # The linear transformations for each layer
        self.W1 = ModuleList([Linear(self.embed_size, self.embed_size) for _ in range(0, self.n_layers)])
        self.W2 = ModuleList([Linear(self.embed_size, self.embed_size) for _ in range(0, self.n_layers)])
        self.act = LeakyReLU()

        # Initialize each of the trainable weights with the Xavier initializer
        self.init_weights()

    def init_weights(self):
        for name, parameter in self.named_parameters():
            if ('bias' not in name):
                init.xavier_uniform_(parameter)

    def forward(self, u, i, j, c, inference=False):
        pred_i = self._out(u, i, c)
        if not inference:
            pred_j = self._out(u, j, c)
        else:
            pred_j = pred_i

        return pred_i, pred_j

    def _out(self, u, i, c=None):
        # adj_splits = split_mtx(self.adj_matrix, n_folds=1)
        if self.reindex:
            embeddings = self.embeddings
            # embeddings = embeddings.view((-1, self.embed_size))
        else:
            embeddings = torch.cat((self.user_embeddings, self.item_embeddings))

        final_embeddings = [embeddings]

        for l in range(self.n_layers):
            # embedding_parts = []
            # for part in adj_splits:
            #     embedding_parts.append(torch.sparse.mm(to_sparse_tensor(part).to(self.device), embeddings))

            # Message construction
            # t1_embeddings = torch.cat(embedding_parts, 0)
            t1_embeddings = torch.sparse.mm(to_sparse_tensor(self.adj_matrix).to(self.device), embeddings)
            t1 = self.W1[l](t1_embeddings)
            t2_embeddings = embeddings.mul(t1_embeddings)
            t2 = self.W2[l](t2_embeddings)

            # Message aggregation
            embeddings = self.act(t1 + t2)
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            final_embeddings.append(normalized_embeddings)

        # Make sure to update the (user/item)_embeddings(_final)
        final_embeddings = torch.cat(final_embeddings, 1)
        if not self.reindex:
            final_u_embeddings, final_i_embeddings = final_embeddings.split((self.n_users, self.n_items), 0)
            self.user_embeddings_final = Parameter(final_u_embeddings)
            self.item_embeddings_final = Parameter(final_i_embeddings)
            return torch.mul(final_u_embeddings[u], final_i_embeddings[i]).sum(dim=1)
        else:
            self.embeddings_final = Parameter(final_embeddings)
            if c is not None:
                #self.embeddings_final[c]
                # TODO: do FM of context as well and return prediction
                if isinstance(c, list) and len(c) > 0:
                    context = torch.stack(c, dim=1)
                    context_embedding = torch.mean(self.embeddings_final[context], dim=1)
                    fm_input = torch.stack((self.embeddings_final[u], self.embeddings_final[i], context_embedding), dim=1)
                    return self.fm(fm_input).squeeze()
                else:
                    fm_input = torch.stack((self.embeddings_final[u], self.embeddings_final[i], self.embeddings_final[c]), dim=1)
                    return self.fm(fm_input).squeeze()
            else:
                return torch.mul(self.embeddings_final[u], self.embeddings_final[i]).sum(dim=1)

    def predict(self, u, i, c):
        pred_i, _ = self.forward(u, i, i, c, inference=True)

        return pred_i.cpu()
