import ipdb
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# from torch_geometric.data import NeighborSampler
from torch_geometric.nn import NNConv

from .data import NeighborSampler, featurize


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation='ReLU'):
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
                 nx_graph,
                 target,
                 embed_size=256,
                 emb_hidden=[256, 256],
                 hidden_size=256,
                 device='cpu'):
        super().__init__()

        self.device = device
        self.embed_size = embed_size
        # featurize
        target_node, target_prop = target.split(':')
        self.target_node = target_node
        self.g, self.target_nodes, self.targets, self.node_features, self.edge_feats = featurize(
            nx_graph, target_node, target_prop)
        # self.target_nodes = self.target_nodes[:1000]
        # k = 1500
        k = self.target_nodes.size(0) + 1
        perm = torch.randperm(self.target_nodes.size(0))
        subset_idx = perm[:k]
        last_tng_id = int(0.8 * subset_idx.size(0))
        tng_idx, _ = torch.sort(subset_idx[:last_tng_id])
        val_idx, _ = torch.sort(subset_idx[last_tng_id:])

        # import ipdb; ipdb.set_trace()
        unique_targets, target_counts = torch.unique(self.targets, return_counts=True)
        print('Data stats', len(self.targets), 100 * target_counts / float(len(self.targets)))
        print('Tng len', len(tng_idx))
        print('Val len', len(val_idx))
        self.tng_target_nodes = self.target_nodes[tng_idx]
        self.tng_targets = self.targets[tng_idx]
        # FIXME: this is wrong, it includes edges to val nodes in the val set
        self.val_target_nodes = self.target_nodes[val_idx]
        self.val_targets = self.targets[val_idx]
        edge_idx = torch.tensor(list(self.g.edges)).t().contiguous()
        tng_edge_idx = self.filter_edge_index(edge_idx, self.val_target_nodes)

        # import ipdb
        # ipdb.set_trace()
        # neigbor_sizes = [3, 2]
        # neigbor_sizes = [10, 10, 10, 10]
        # neigbor_sizes = [-1, -1]
        neigbor_sizes = [10, 10, 10]
        # neigbor_sizes = [25]
        batch_size = min(50, len(self.tng_targets) // 4)
        # batch_size = 5
        workers = 1
        self.tng_loader = NeighborSampler(
            edge_idx,
            node_idx=self.tng_target_nodes,
            sizes=neigbor_sizes,
            batch_size=batch_size,
            # batch_size=1024,
            # shuffle=False,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=True)

        self.val_loader = NeighborSampler(
            # tng_edge_idx,
            edge_idx,
            node_idx=self.val_target_nodes,
            # node_idx=self.tng_target_nodes,
            sizes=neigbor_sizes,
            batch_size=batch_size,
            # batch_size=1024,
            shuffle=False,
            # shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=True)

        self.edge_feats = self.edge_feats.to(device)
        self.targets = self.targets.to(device)
        self.target_nodes = self.target_nodes.to(device)
        embedders = {}
        for node_type, node_props in self.node_features.items():
            self.node_features[node_type]['x_in'] = node_props['x_in'].to(device)
            self.node_features[node_type]['n_ids'] = node_props['n_ids'].to(device)
            embedders[node_type] = MLP(input_size=node_props['x_in'].shape[1],
                                       hidden_sizes=emb_hidden,
                                       output_size=embed_size)
            if node_type == target_node:
                self.node_features[node_type]['x_out'] = node_props['x_out'].to(device)
                self.embedders_out = MLP(input_size=node_props['x_out'].shape[1],
                                         hidden_sizes=emb_hidden,
                                         output_size=embed_size)
        self.embedders = nn.ModuleDict(embedders)

        output_size = self.node_features[target_node]['y'].shape[1]

        # import ipdb
        # ipdb.set_trace()
        root_weight = True
        self.convs = nn.ModuleList([
            NNConv(embed_size,
                   hidden_size,
                   MLP(input_size=self.edge_feats.shape[1],
                       hidden_sizes=[hidden_size, hidden_size],
                       output_size=hidden_size * embed_size),
                   aggr='mean',
                   root_weight=root_weight,
                   bias=True),
            NNConv(hidden_size,
                   hidden_size,
                   MLP(input_size=self.edge_feats.shape[1],
                       hidden_sizes=[hidden_size, hidden_size],
                       output_size=hidden_size * hidden_size),
                   aggr='mean',
                   root_weight=root_weight,
                   bias=True),
            NNConv(hidden_size,
                   hidden_size,
                   MLP(input_size=self.edge_feats.shape[1],
                       hidden_sizes=[hidden_size, hidden_size],
                       output_size=hidden_size * hidden_size),
                   aggr='mean',
                   root_weight=root_weight,
                   bias=True),
            NNConv(hidden_size,
                   hidden_size,
                   MLP(input_size=self.edge_feats.shape[1],
                       hidden_sizes=[hidden_size, hidden_size],
                       output_size=hidden_size * hidden_size),
                   aggr='mean',
                   root_weight=root_weight,
                   bias=True),
        ])
        self.bns = nn.ModuleList([
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
        ])
        self.lin1 = torch.nn.Linear(hidden_size, output_size)

    def filter_edge_index(self, edge_idx, node_idx):
        mask = [
            i for i, edge in enumerate(edge_idx.t())
            if not (edge[0] in node_idx or edge[1] in node_idx)
        ]
        mask = torch.LongTensor(mask)
        return edge_idx[:, mask]

    def get_targets(self, n_id):
        ind_map = self.get_ind_map(n_id, self.target_nodes)
        return self.targets[ind_map[:, 2]]

    def get_ind_map(self, n_id1, n_id2, ignore1=[]):
        # ind_map = [
        #     torch.tensor([i, id, torch.nonzero(n_id2 == id, as_tuple=False).squeeze()])
        #     for i, id in enumerate(n_id1) if id in n_id2
        # ]
        ind_map = []
        for i, id in enumerate(n_id1):
            if id in n_id2 and i not in ignore1:
                t = torch.tensor([i, id, torch.nonzero(n_id2 == id, as_tuple=False).squeeze()])
                ind_map.append(t)
        if len(ind_map) == 0:
            return None
        return torch.stack(ind_map)

    def forward(self, n_id, adjs):
        # embed
        h = torch.zeros((n_id.shape[0], self.embed_size)).to(self.device)
        h += np.nan
        for node_type, node_props in self.node_features.items():
            np_n_ids = node_props['n_ids']
            ind_map = self.get_ind_map(n_id[:, 0], np_n_ids,
                                       torch.nonzero(n_id[:, 1] == 1, as_tuple=False).squeeze())
            if ind_map is None:
                continue

            h[ind_map[:, 0]] = self.embedders[node_type](node_props['x_in'][ind_map[:, 2]])

        node_props = self.node_features[self.target_node]
        np_n_ids = node_props['n_ids']
        ind_map = self.get_ind_map(n_id[:, 0], np_n_ids,
                                   torch.nonzero(n_id[:, 1] == 0, as_tuple=False).squeeze())
        if ind_map is None:
            print('WTF???')

        h[ind_map[:, 0]] = self.embedders_out(node_props['x_out'][ind_map[:, 2]])
        # import ipdb
        # ipdb.set_trace()

        # import ipdb; ipdb.set_trace()
        # message passing
        for i, (edge_index, e_id, size) in enumerate(adjs):
            h_target = h[:size[1]]
            # h_target = h[n_id[:, 2] > i]
            h = self.convs[i]((h, h_target), edge_index, self.edge_feats[e_id])
            h = F.relu(h)
            # h = self.bns[i](h)
            # h = self.convs[i](h, edge_index, self.edge_feats[e_id])
            # target_mask = n_id[:, 2] > i
            # h = h[target_mask]
            # out, h = self.gru(m.unsqueeze(0), h)
            # out = out.squeeze(0)

        # h = F.relu(h)
        # h = self.lin1(h)
        # h = F.relu(h)
        # return F.log_softmax(h, dim=-1)
        return h
