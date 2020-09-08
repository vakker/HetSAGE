import argparse
import json
import random
from os import path as osp

import ipdb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

import torch_geometric
from hetsage.model import Model
from hetsage.utils import init_random


def zero_grad(model):
    for p in model.parameters():
        p.grad = None


def run_iter(model, optimizer, device):
    metrics = {}
    loss, acc = _run_iter(model, model.tng_loader, optimizer, device=device)
    metrics['tng-loss'] = loss
    metrics['tng-acc'] = acc

    with torch.no_grad():
        # loss, acc = _run_iter(model, model.tng_loader, device=device)
        # metrics['tng2-loss'] = loss
        # metrics['tng2-acc'] = acc
        loss, acc = _run_iter(model, model.val_loader, device=device)
        metrics['val-loss'] = loss
        metrics['val-acc'] = acc

    return metrics


def _run_iter(model, data_loader, optimizer=None, device='cpu'):
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_correct = 0
    total_nodes = 0
    i = 0
    for batch_size, n_id, adjs in tqdm(data_loader):
        # import ipdb; ipdb.set_trace()
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        if isinstance(adjs, torch_geometric.data.sampler.Adj):
            adjs = [adjs]
        adjs = [adj.to(device) for adj in adjs]
        n_id = n_id.to(device)
        targets = model.get_targets(n_id[:batch_size, 0])
        # targets = model.get_targets(n_id[torch.nonzero(n_id[:, 1] == 1,
        #                                                as_tuple=False).squeeze()][:, 0])
        if optimizer is not None:
            # zero_grad(model)
            optimizer.zero_grad()
        out = model(n_id, adjs)
        loss = F.nll_loss(F.log_softmax(out, dim=-1), targets)
        if optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach()) * batch_size

        # import ipdb; ipdb.set_trace()
        # out_np = out.cpu().to_numpy()
        y_pred = torch.argmax(out.detach(), dim=-1)
        # if i >= 0:
        #     print(i)
        #     print(n_id[:batch_size, 0])
        #     print(adjs[0].edge_index.t())
        #     print(out)
        #     print(targets)
        #     print(y_pred)
        # i += 1
        total_correct += float((y_pred == targets).sum())
        total_nodes += batch_size

    loss = total_loss / total_nodes
    approx_acc = total_correct / total_nodes

    return loss, approx_acc


# @torch.no_grad()
# def test():
#     model.eval()

#     out = model.inference(x)

#     y_true = y.cpu().unsqueeze(-1)
#     y_pred = out.argmax(dim=-1, keepdim=True)

#     results = []
#     for mask in [data.train_mask, data.val_mask, data.test_mask]:
#         results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

#     return results


def main(args):
    init_random()
    # load graph
    g = nx.readwrite.gml.read_gml(args.gml)

    if args.use_gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model = Model(g, args.target, device=device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for epoch in range(1, 1 + args.max_epochs):
        metrics = run_iter(model, optimizer, device=device)
        tng_loss = metrics['tng-loss']
        tng_acc = metrics['tng-acc']
        # tng2_loss = metrics['tng2-loss']
        # tng2_acc = metrics['tng2-acc']
        val_loss = metrics['val-loss']
        val_acc = metrics['val-acc']
        msg = ''
        msg += f'Epoch {epoch:02d}, '
        msg += f'Tng loss: {tng_loss:.4f}, '
        msg += f'Tng acc: {100*tng_acc:.2f}, '
        # msg += f'Tng2 loss: {tng2_loss:.4f}, '
        # msg += f'Tng2 acc: {100*tng2_acc:.2f}, '
        msg += f'Val loss: {val_loss:.4f}, '
        msg += f'Val acc: {100*val_acc:.2f}'
        print(msg)
        # train_acc, val_acc, test_acc = test()
        # print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, ' f'Test: {test_acc:.4f}')


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--gml')
    PARSER.add_argument('--target')
    PARSER.add_argument('--use-gpu', action='store_true')
    PARSER.add_argument('--max-epochs', type=int)

    ARGS = PARSER.parse_args()
    main(ARGS)
