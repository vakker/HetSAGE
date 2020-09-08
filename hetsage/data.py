import collections
import copy
import logging
import re
from typing import List, NamedTuple, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from torch_sparse import SparseTensor
from tqdm import tqdm

import torch_geometric
from torch_geometric.data.sampler import Adj
from torch_geometric.utils import (contains_isolated_nodes,
                                   contains_self_loops, is_undirected)

tensor = torch.FloatTensor


def get_props(g):
    return {
        data['nodetype']: list(data.get('properties', {}).keys())
        for n, data in g.nodes(data=True)
    }


def featurize(g, target_node, target_prop):
    g = nx.convert_node_labels_to_integers(g)
    edge_cats = set()
    for n1, n2, data in tqdm(g.edges(data=True)):
        edge_cats = edge_cats.union({data['label']})
    edge_cats = sorted(list(edge_cats))
    edge_feats = []
    for n1, n2, k, data in tqdm(g.edges(data=True, keys=True)):
        # g.edges[n1, n2, k]['x'] = (data['label'] == np.array(edge_cats)).astype(np.int)
        edge_feats.append((data['label'] == np.array(edge_cats)).astype(np.int))
    edge_feats = tensor(np.stack(edge_feats))

    features = {}
    for n, data in tqdm(g.nodes(data=True)):
        nt = data['nodetype']
        if nt in features:
            if data.get('prop', {}).keys() != features[nt]['prop'].keys():
                raise ValueError('Inconsistent prop keys')
            if data.get('single_cat', {}).keys() != features[nt]['single_cat'].keys():
                raise ValueError('Inconsistent prop keys')
            if data.get('multi_cat', {}).keys() != features[nt]['multi_cat'].keys():
                raise ValueError('Inconsistent prop keys')

            for p, v in data.get('prop', {}).items():
                if v < features[nt]['prop'][p]['min']:
                    features[nt]['prop'][p]['min'] = v
                if v > features[nt]['prop'][p]['max']:
                    features[nt]['prop'][p]['max'] = v

            for p, v in data.get('single_cat', {}).items():
                features[nt]['single_cat'][p].add(v)
            for p, v in data.get('multi_cat', {}).items():
                features[nt]['multi_cat'][p].update(v)
        else:
            prop = {p: {'min': v, 'max': v} for p, v in data.get('prop', {}).items()}
            single_cat = {p: {v} for p, v in data.get('single_cat', {}).items()}
            multi_cat = {p: set(v) for p, v in data.get('multi_cat', {}).items()}

            features[nt] = {
                'prop': prop,
                'single_cat': single_cat,
                'multi_cat': multi_cat,
            }

    # feature_mats = {}
    for nt in features:
        for pc, cats in features[nt]['single_cat'].items():
            features[nt]['single_cat'][pc] = sorted(list(cats))
        for pc, cats in features[nt]['multi_cat'].items():
            features[nt]['multi_cat'][pc] = sorted(list(cats))
        features[nt].update({'x_in': [], 'x_out': [], 'y': [], 'n_ids': []})

    target_nodes = []
    targets = []
    for n, data in tqdm(g.nodes(data=True)):
        nt = data['nodetype']
        prop_keys = set(features[nt]['prop'].keys())
        single_cat_keys = set(features[nt]['single_cat'].keys())
        multi_cat_keys = set(features[nt]['multi_cat'].keys())
        if nt == target_node:
            prop_keys = prop_keys - {target_prop}
            single_cat_keys = single_cat_keys - {target_prop}
            multi_cat_keys = multi_cat_keys - {target_prop}

        nd = data['nodetype']
        prop = data.get('prop', {})
        single_cats = data.get('single_cat', {})
        multi_cats = data.get('multi_cat', {})
        x_p = get_prop(prop, prop_keys, features[nt]['prop'])
        x_sc = get_cat(single_cats, single_cat_keys, features[nt]['single_cat'])
        x_mc = get_cat(multi_cats, multi_cat_keys, features[nt]['multi_cat'])
        if nd == target_node:
            if target_prop in features[nt]['prop']:
                y = get_prop(prop, [target_prop], features[nt]['prop'])
            elif target_prop in features[nt]['single_cat']:
                y = get_cat(single_cats, [target_prop], features[nt]['single_cat'])
            elif target_prop in features[nt]['multi_cat']:
                y = get_cat(multi_cats, [target_prop], features[nt]['multi_cat'])
            else:
                raise ValueError(f'{target_prop} is not a property')
            target_nodes.append(n)
            # targets.append(torch.tensor(y))
            targets.append(torch.nonzero(torch.tensor(y) == 1, as_tuple=False).squeeze())
        else:
            y = []
        # g.nodes[n]['x_in'] = torch.tensor(np.concatenate([x_p, x_c, y]))
        # g.nodes[n]['x_out'] = torch.tensor(np.concatenate([x_p, x_c]))
        # g.nodes[n]['x_in'] = get_tensor(np.concatenate([x_p, x_c]))
        g.nodes[n]['x_in'] = get_tensor(np.concatenate([x_p, x_sc, x_mc, y]))
        # g.nodes[n]['x_in'] = get_tensor(np.concatenate([x_p, x_sc, x_mc]))
        # g.nodes[n]['x_in'] = get_tensor(np.concatenate([y]))
        g.nodes[n]['x_out'] = get_tensor(np.concatenate([x_p, x_sc, x_mc]))
        g.nodes[n]['y'] = torch.tensor(y)
        # features[nt]['x_in'].append(torch.tensor([n]).float())
        features[nt]['x_in'].append(g.nodes[n]['x_in'])
        # features[nt]['x_in'].append(get_tensor(np.concatenate([x_p, x_c, y])))
        features[nt]['x_out'].append(g.nodes[n]['x_out'])
        features[nt]['y'].append(g.nodes[n]['y'])
        features[nt]['n_ids'].append(torch.tensor(n))
        # features[nt]['x_in_size'] = g.nodes[n]['x_in'].shape[0]
        # features[nt]['x_out_size'] = g.nodes[n]['x_out'].shape[0]
        # features[nt]['y_size'] = g.nodes[n]['y'].shape[0]

    for nt in features:
        features[nt]['x_in'] = torch.stack(features[nt]['x_in'])
        features[nt]['x_out'] = torch.stack(features[nt]['x_out'])
        features[nt]['y'] = torch.stack(features[nt]['y'])
        features[nt]['n_ids'] = torch.stack(features[nt]['n_ids'])

    target_nodes = torch.LongTensor(target_nodes)
    targets = torch.stack(targets)
    return g, target_nodes, targets, features, edge_feats


def get_tensor(np_arr):
    if np_arr.shape[0] > 0:
        return tensor(np_arr)
    return tensor([0])


def get_cat(cats, cat_keys, all_cats):
    x_c = []
    # x_c = [(cats[k] == np.array(all_cats[k])).astype(np.int) for k in cat_keys]
    for k in cat_keys:
        if not isinstance(cats[k], list):
            cs = [cats[k]]
        else:
            cs = cats[k]
        c_ = []
        for i, c in enumerate(all_cats[k]):
            if c in cs:
                c_.append(1)
            else:
                c_.append(0)
        # (cats[k] == np.array(all_cats[k])).astype(np.int)
        x_c.append(c_)
    if x_c:
        return np.concatenate(x_c)
    return np.array(x_c)


def get_prop(props, prop_keys, all_props, normalize=True):
    x_p = np.array([
        norm(props[k], all_props[k]['min'], all_props[k]['max']) if normalize else props[k]
        for k in prop_keys
    ])
    return x_p


def norm(v, v_min, v_max):
    return (v - v_min) / (v_max - v_min)


class NeighborSampler(torch.utils.data.DataLoader):
    def __init__(self,
                 edge_index: torch.Tensor,
                 sizes: List[int],
                 node_idx: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 flow: str = "source_to_target",
                 **kwargs):

        N = int(edge_index.max() + 1) if num_nodes is None else num_nodes
        edge_attr = torch.arange(edge_index.size(1))
        adj = SparseTensor(row=edge_index[0],
                           col=edge_index[1],
                           value=edge_attr,
                           sparse_sizes=(N, N),
                           is_sorted=False)
        adj = adj.t() if flow == 'source_to_target' else adj
        self.adj = adj.to('cpu')

        if node_idx is None:
            node_idx = torch.arange(N)
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        self.sizes = sizes
        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        super(NeighborSampler, self).__init__(node_idx.tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        n_id_offset = 0
        n_id_map = []
        edge_indeces = [[] for _ in self.sizes]
        e_ids = [[] for _ in self.sizes]

        for target_id in batch:
            n_id = target_id.unsqueeze(dim=0)
            n_id_targets = []
            for i, size in enumerate(self.sizes):
                n_id_targets.append(n_id)
                adj, n_id = self.adj.sample_adj(n_id, size, replace=False)
                if self.flow == 'source_to_target':
                    adj = adj.t()
                row, col, e_id = adj.coo()
                row += n_id_offset
                col += n_id_offset
                size = adj.sparse_sizes()
                edge_index = torch.stack([row, col], dim=0)
                edge_indeces[i].append(edge_index)
                e_ids[i].append(e_id)

            is_target = n_id == target_id
            n_id_layers = torch.zeros_like(n_id)  # * len(n_id_targets)
            for i, targets in enumerate(reversed(n_id_targets)):
                id_in_layer = [idx for idx, n in enumerate(n_id) if n in targets]
                n_id_layers[id_in_layer] = i + 1
            n_id_map.append(torch.stack([n_id, is_target, n_id_layers], dim=1))
            n_id_offset += len(n_id)

        n_id_map = torch.cat(n_id_map)
        _, sorted_idx = torch.sort(n_id_map[:, 2], descending=True)
        n_id_map = n_id_map[sorted_idx]
        adjs = []
        for i, size in enumerate(self.sizes):
            edge_index = torch.cat(edge_indeces[i], dim=-1)
            edge_index = reindex(sorted_idx, edge_index)
            e_id = torch.cat(e_ids[i], dim=-1)
            M = edge_index[0].max().item() + 1
            N = edge_index[1].max().item() + 1
            # M = (n_id_map[:, 2] >= i).sum().item()
            # N = (n_id_map[:, 2] > i).sum().item()
            size = (M, N)
            adjs.append(Adj(edge_index, e_id, size))

        # import ipdb
        # ipdb.set_trace()
        if adjs[0].size[-1] != len(batch):
            import ipdb
            ipdb.set_trace()
        return batch_size, n_id_map, adjs[::-1]

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)


def reindex(idx_map, edge_index):
    edge_reindex = torch.ones_like(edge_index) * -1
    for new_idx, old_idx in enumerate(idx_map):
        edge_reindex[edge_index == old_idx] = new_idx

    return edge_reindex
