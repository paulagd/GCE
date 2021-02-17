import torch
from torch_geometric.nn import GCNConv, GATConv, SGConv, SAGEConv
from IPython import embed


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        return self.mlp(x)


class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

    
class GCE(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, features, train_mat_edges, dropout=0, args=None):

        super(GCE, self).__init__()

        self.A = train_mat_edges
        self.features = features  # so far, Identity matrix
        # self.field_dims = field_dims  # so far, Identity matrix
        # self.embed_dim = embed_dim  # so far, Identity matrix
        # GCNConv applies the convolution over the graph

        if args.gcetype == 'gat':
            print('GAT!!')
            self.GCN_module = GATConv(int(field_dims), embed_dim, concat=True, heads=args.num_heads, dropout=dropout)
        elif args.gcetype == 'sage':
            print('SAGE!!')
            self.features = self.features.to_dense()  # so far, Identity matrix
            self.GCN_module = SAGEConv(int(field_dims), embed_dim)
        # elif args.gcetype == 'sgc':
        #     print(f'SGC hop {args.mh}!!')
        #     self.GCN_module = SGConv(field_dims, embed_dim, cached=True, K=args.mh)
        else:
            print('GCN!!')
            self.GCN_module = GCNConv(field_dims, embed_dim, cached=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """

        return self.GCN_module(self.features, self.A)[x]
