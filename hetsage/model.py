import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import NNConv


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation='LeakyReLU'):
        super().__init__()

        if not isinstance(hidden_sizes, list):
            hidden_sizes = [hidden_sizes]

        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        act = getattr(nn, activation)
        layers.append(act())

        for i, s in enumerate(hidden_sizes):
            if i < len(hidden_sizes) - 1:
                output_feats = hidden_sizes[i + 1]
            else:
                output_feats = output_size
            layers.append(nn.Linear(hidden_sizes[i], output_feats))
            if i < len(hidden_sizes) - 1:
                layers.append(act())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y

    def is_cuda(self):
        return next(self.parameters()).is_cuda


class Model(nn.Module):
    def __init__(self,
                 graph_info,
                 neighbor_steps,
                 embed_size=256,
                 emb_hidden=[256, 256],
                 hidden_size=256,
                 activation='LeakyReLU'):
        super().__init__()

        self.embed_size = embed_size
        embedders = {}
        for node_type, node_props in graph_info['in_nodes'].items():
            embedders[node_type] = MLP(input_size=node_props['in_size'],
                                       hidden_sizes=emb_hidden,
                                       output_size=embed_size,
                                       activation=activation)
        self.embedders = nn.ModuleDict(embedders)

        self.embedders_out = MLP(input_size=graph_info['target_node']['in_size'],
                                 hidden_sizes=emb_hidden,
                                 output_size=embed_size,
                                 activation=activation)

        root_weight = True
        module_list = [
            NNConv(embed_size,
                   hidden_size,
                   MLP(input_size=graph_info['edges']['in_size'],
                       hidden_sizes=[hidden_size, hidden_size],
                       output_size=hidden_size * embed_size,
                       activation=activation),
                   aggr='mean',
                   root_weight=root_weight,
                   bias=True)
        ]
        bns = [torch.nn.BatchNorm1d(hidden_size)]
        for s in range(1, neighbor_steps):
            module_list.append(
                NNConv(hidden_size,
                       hidden_size,
                       MLP(input_size=graph_info['edges']['in_size'],
                           hidden_sizes=[hidden_size, hidden_size],
                           output_size=hidden_size * hidden_size,
                           activation=activation),
                       aggr='mean',
                       root_weight=root_weight,
                       bias=True))
            bns.append(torch.nn.BatchNorm1d(hidden_size))

        self.convs = nn.ModuleList(module_list)
        self.bns = nn.ModuleList(bns)

        self.act = getattr(torch.nn, activation)()

        output_size = graph_info['target_node']['out_size']
        self.lin1 = torch.nn.Linear(hidden_size, output_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_nodes, adjs):
        num_nodes = sum([node_info.x.shape[0] for node_type, node_info in input_nodes.items()])
        # embed
        h = torch.zeros((num_nodes, self.embed_size)).to(self.device)
        h += np.nan
        for node_type, node_info in input_nodes.items():
            if node_type == 'target':
                h[node_info.h_id] = self.embedders_out(node_info.x)
            else:
                h[node_info.h_id] = self.embedders[node_type](node_info.x)

        # message passing
        for i, (edge_index, e_feat, size) in enumerate(adjs):
            h_target = h[:size[1]]
            h = self.convs[i]((h, h_target), edge_index, e_feat)
            h = self.act(h)
            # h = self.bns[i](h)
            # out, h = self.gru(m.unsqueeze(0), h)
            # out = out.squeeze(0)

        h = self.lin1(h)
        h = self.act(h)
        # return F.log_softmax(h, dim=-1)
        return h
