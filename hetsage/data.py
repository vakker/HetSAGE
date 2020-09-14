from typing import List, NamedTuple, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from torch_sparse import SparseTensor
from tqdm import tqdm

tensor = torch.FloatTensor


def get_props(g):
    return {
        data['nodetype']: list(data.get('properties', {}).keys())
        for n, data in g.nodes(data=True)
    }


def featurize(g, target_node, target_prop, include_target_label=True):
    g = nx.convert_node_labels_to_integers(g)
    edge_cats = set()
    for n1, n2, data in tqdm(g.edges(data=True)):
        edge_cats = edge_cats.union({data['label']})
    edge_cats = sorted(list(edge_cats))
    edge_feats = []
    for n1, n2, k, data in tqdm(g.edges(data=True, keys=True)):
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
            targets.append(torch.nonzero(torch.tensor(y) == 1, as_tuple=False).squeeze())
        else:
            y = []
        if include_target_label:
            g.nodes[n]['x_in'] = get_tensor(np.concatenate([x_p, x_sc, x_mc, y]))
        else:
            g.nodes[n]['x_in'] = get_tensor(np.concatenate([x_p, x_sc, x_mc]))
        g.nodes[n]['x_out'] = get_tensor(np.concatenate([x_p, x_sc, x_mc]))
        g.nodes[n]['y'] = torch.tensor(y)
        features[nt]['x_in'].append(g.nodes[n]['x_in'])
        features[nt]['x_out'].append(g.nodes[n]['x_out'])
        features[nt]['y'].append(g.nodes[n]['y'])
        features[nt]['n_ids'].append(torch.tensor(n))

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
                 edge_features: torch.Tensor,
                 sizes: List[int],
                 node_idx: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 flow: str = "source_to_target",
                 **kwargs):

        N = int(edge_index.max() + 1) if num_nodes is None else num_nodes
        adj = SparseTensor(row=edge_index[0],
                           col=edge_index[1],
                           value=edge_features,
                           sparse_sizes=(N, N),
                           is_sorted=False)
        adj = adj.t() if flow == 'source_to_target' else adj
        self.adj = adj

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
        e_feats = [[] for _ in self.sizes]

        for target_id in batch:
            n_id = target_id.unsqueeze(dim=0)
            n_id_targets = []
            for i, size in enumerate(self.sizes):
                n_id_targets.append(n_id)
                adj, n_id = self.adj.sample_adj(n_id, size, replace=False)
                if self.flow == 'source_to_target':
                    adj = adj.t()
                row, col, e_feat = adj.coo()
                row += n_id_offset
                col += n_id_offset
                size = adj.sparse_sizes()
                edge_index = torch.stack([row, col], dim=0)
                edge_indeces[i].append(edge_index)
                e_feats[i].append(e_feat)

            is_target = n_id == target_id
            n_id_layers = torch.zeros_like(n_id)  # * len(n_id_targets)
            for i, targets in enumerate(reversed(n_id_targets)):
                id_in_layer = [idx for idx, n in enumerate(n_id) if n in targets]
                n_id_layers[id_in_layer] = i + 1
            n_id_map.append(torch.stack([n_id, is_target, n_id_layers], dim=1))
            n_id_offset += len(n_id)

        n_id_map = torch.cat(n_id_map)
        _, sorted_idx = torch.sort(n_id_map[:, 2], descending=True)
        _, sorted_idx_inv = torch.sort(sorted_idx)
        n_id_map = n_id_map[sorted_idx]
        adjs = []
        for i, size in enumerate(self.sizes):
            edge_index = torch.cat(edge_indeces[i], dim=-1)
            edge_index = reindex(sorted_idx, sorted_idx_inv, edge_index)
            e_feat = torch.cat(e_feats[i], dim=0)
            M = edge_index[0].max().item() + 1
            N = edge_index[1].max().item() + 1
            size = (M, N)
            adjs.append(Adj(edge_index, e_feat, size))

        return batch_size, n_id_map, adjs[::-1]

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)


def reindex(idx_map, idx_map_inv, edge_index):
    edge_reindex = torch.ones_like(edge_index) * -1
    for i in range(edge_reindex.shape[1]):
        edge_reindex[0, i] = idx_map_inv[edge_index[0, i]]
        edge_reindex[1, i] = idx_map_inv[edge_index[1, i]]

    return edge_reindex


class DataManager:
    def __init__(self,
                 graph_file,
                 target,
                 include_target_label=True,
                 neighbor_sizes=[20, 20],
                 batch_size=200,
                 workers=1,
                 target_node_lim=None):
        # load graph
        g = nx.nx.read_gpickle(graph_file)
        self.neighbor_steps = len(neighbor_sizes)

        # featurize
        target_node, target_prop = target.split(':')
        self.target_node = target_node
        self.g, self.target_nodes, self.targets, self.node_features, self.edge_feats = featurize(
            g, target_node, target_prop, include_target_label)

        edge_idx = torch.tensor(list(self.g.edges)).t().contiguous()

        self.targets_sparse = SparseTensor(row=self.target_nodes,
                                           col=torch.zeros_like(self.target_nodes),
                                           value=self.targets)

        self.node_map = torch.zeros((len(self.g.nodes), 2), dtype=torch.long)
        for i, _ in enumerate(self.node_map):
            for node_type_id, (node_type, node_props) in enumerate(self.node_features.items()):
                idx = torch.nonzero(i == node_props['n_ids'], as_tuple=False)
                if idx.shape[0] == 1:
                    idx = idx.squeeze(dim=-1)
                    self.node_map[i, 0] = node_type_id
                    self.node_map[i, 1] = idx

        if target_node_lim:
            k = target_node_lim
        else:
            k = self.target_nodes.size(0) + 1
        perm = torch.randperm(self.target_nodes.size(0))
        subset_idx = perm[:k]
        last_tng_id = int(0.8 * subset_idx.size(0))
        tng_idx, _ = torch.sort(subset_idx[:last_tng_id])
        val_idx, _ = torch.sort(subset_idx[last_tng_id:])

        unique_targets, target_counts = torch.unique(self.targets, return_counts=True)
        self.target_weights = 1 / target_counts.float()
        self.target_weights /= self.target_weights.sum()
        print('Data stats', len(self.targets), 100 * target_counts / float(len(self.targets)))
        print('Tng len', len(tng_idx))
        print('Val len', len(val_idx))
        self.tng_target_nodes = self.target_nodes[tng_idx]
        self.tng_targets = self.targets[tng_idx]
        # FIXME: is this wrong? It includes edges to val nodes in the val set
        self.val_target_nodes = self.target_nodes[val_idx]
        self.val_targets = self.targets[val_idx]
        # tng_edge_idx = self.filter_edge_index(edge_idx, self.val_target_nodes)

        self.tng_loader = NeighborSampler(
            # tng_edge_idx,
            edge_idx,
            self.edge_feats,
            node_idx=self.tng_target_nodes,
            sizes=neighbor_sizes,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )

        self.val_loader = NeighborSampler(
            edge_idx,
            self.edge_feats,
            node_idx=self.val_target_nodes,
            sizes=neighbor_sizes,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )

        self.graph_info = {
            'in_nodes': {},
        }
        for node_type, node_props in self.node_features.items():
            self.graph_info['in_nodes'][node_type] = {'in_size': node_props['x_in'].shape[1]}

        self.graph_info['target_node'] = {
            'in_size': self.node_features[target_node]['x_out'].shape[1],
            'out_size': self.node_features[target_node]['y'].shape[1]
        }
        self.graph_info['edges'] = {'in_size': self.edge_feats.shape[1]}

    def filter_edge_index(self, edge_idx, node_idx):
        mask = [
            i for i, edge in enumerate(edge_idx.t())
            if not (edge[0] in node_idx or edge[1] in node_idx)
        ]
        mask = torch.LongTensor(mask)
        return edge_idx[:, mask]

    def get_targets(self, n_id):
        _, _, value = self.targets_sparse[n_id].coo()
        return value

    def get_id_map(self, node_id):
        node_map = {
            node_type: {
                'x': [],
                'h_id': []
            }
            for node_type, node_props in self.node_features.items()
        }
        node_map.update({'target': {'x': [], 'h_id': []}})

        nodes = [
            torch.tensor(range(len(node_id))).unsqueeze(-1),
            node_id[:, 1].unsqueeze(-1),
            self.node_map[node_id[:, 0]],
        ]
        nodes = torch.cat(nodes, dim=1)
        for node_type_id, node_type in enumerate(list(self.node_features.keys())):
            idx = (nodes[:, 1] == 1) & (nodes[:, 2] == node_type_id)
            if torch.any(idx):
                node_map['target']['x'] = self.node_features[node_type]['x_out'][nodes[idx, 3]]
                node_map['target']['h_id'] = nodes[idx, 0]

            idx = (nodes[:, 1] != 1) & (nodes[:, 2] == node_type_id)
            if torch.any(idx):
                node_map[node_type]['x'] = self.node_features[node_type]['x_in'][nodes[idx, 3]]
                node_map[node_type]['h_id'] = nodes[idx, 0]

        node_map_out = {}
        for node_type, n_map in node_map.items():
            if len(node_map[node_type]['x']) == 0:
                continue
            node_map_out[node_type] = NodeMap(
                node_map[node_type]['x'],
                node_map[node_type]['h_id'].squeeze(),
            )

        return node_map_out


class NodeMap(NamedTuple):
    x: torch.Tensor
    h_id: torch.Tensor

    def to(self, *args, **kwargs):
        return NodeMap(
            self.x.to(*args, **kwargs),
            self.h_id.to(*args, **kwargs),
        )


class Adj(NamedTuple):
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(
            self.edge_index.to(*args, **kwargs),
            self.edge_features.to(*args, **kwargs),
            self.size,
        )
