from typing import Callable, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, OptTensor, Size

from .data import Adj


class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size,
                 activation='LeakyReLU',
                 final_act=False):
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
            if i < len(hidden_sizes) - 1 or final_act:
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
                 emb_hidden=[256, 256],
                 embed_size=256,
                 hidden_size=256,
                 activation='LeakyReLU'):
        super().__init__()

        self.embed_size = embed_size
        embedders = {}
        for node_type, node_props in graph_info['in_nodes'].items():
            embedders[node_type] = MLP(input_size=node_props['in_size'],
                                       hidden_sizes=emb_hidden,
                                       output_size=embed_size,
                                       activation=activation,
                                       final_act=True)
        self.embedders = nn.ModuleDict(embedders)

        self.embedders_out = MLP(input_size=graph_info['target_node']['in_size'],
                                 hidden_sizes=emb_hidden,
                                 output_size=embed_size,
                                 activation=activation,
                                 final_act=True)

        root_layer = True
        module_list = [
            NNConv(
                embed_size,
                hidden_size,
                edge_nn=MLP(input_size=graph_info['edges']['in_size'],
                            hidden_sizes=[hidden_size, hidden_size],
                            output_size=hidden_size * embed_size,
                            activation=activation,
                            final_act=True),
                node_nn=MLP(input_size=hidden_size + embed_size,
                            hidden_sizes=[hidden_size, hidden_size],
                            output_size=hidden_size,
                            activation=activation,
                            final_act=False),
                aggr='mean',
                root_layer=root_layer,
            )
        ]
        bns = [torch.nn.BatchNorm1d(hidden_size)]
        for s in range(1, neighbor_steps):
            module_list.append(
                NNConv(
                    hidden_size,
                    hidden_size,
                    edge_nn=MLP(input_size=graph_info['edges']['in_size'],
                                hidden_sizes=[hidden_size, hidden_size],
                                output_size=hidden_size * hidden_size,
                                activation=activation,
                                final_act=True),
                    node_nn=MLP(input_size=hidden_size * 2,
                                hidden_sizes=[hidden_size, hidden_size],
                                output_size=hidden_size,
                                activation=activation,
                                final_act=False),
                    aggr='mean',
                    root_layer=root_layer,
                ))
            bns.append(torch.nn.BatchNorm1d(hidden_size))

        self.convs = nn.ModuleList(module_list)
        self.bns = nn.ModuleList(bns)

        self.act = getattr(torch.nn, activation)()

        output_size = graph_info['target_node']['out_size']
        self.lin1 = torch.nn.Linear(hidden_size, hidden_size)
        self.lin2 = torch.nn.Linear(hidden_size, output_size)

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
            # h = self.act(h)
            # print('before', h)
            # h = self.bns[i](h)
            # print('after', h)
            # out, h = self.gru(m.unsqueeze(0), h)
            # out = out.squeeze(0)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)
        return h


class NNConv(MessagePassing):
    def __init__(self,
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 node_nn: Callable,
                 edge_nn: Callable,
                 aggr: str = 'add',
                 root_layer: bool = True,
                 **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_nn = edge_nn
        self.node_nn = node_nn
        self.aggr = aggr

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.in_channels_l = in_channels[0]

        if root_layer:
            self.root = nn.Linear(in_channels[1], out_channels)
        else:
            self.root = None

        self.bn = torch.nn.BatchNorm1d(out_channels + in_channels[1])
        # self.reset_parameters()

    # def reset_parameters(self):
    #     reset(self.nn)
    #     if self.root is not None:
    #         uniform(self.root.size(0), self.root)
    #     zeros(self.bias)

    def forward(self,
                x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                edge_attr: OptTensor = None,
                size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None and self.root is not None:
            pass
            # out = self.root(x_r)
            # out += self.root(x_r)
            # root_out = F.leaky_relu(self.root(x_r))
            # print('out', out)
            # print('root_out', root_out)
            # out += root_out
            # print('out', out)

        out = torch.cat([out, x_r], dim=-1)
        out = F.leaky_relu(out)
        out = self.bn(out)
        out = self.node_nn(out)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        weight = self.edge_nn(edge_attr)
        weight = weight.view(-1, self.in_channels_l, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
