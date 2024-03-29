import argparse
import time
from os import path as osp

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
import yaml
from tqdm import tqdm, trange

from hetsage.data import DataManager
from hetsage.model import Model
from hetsage.utils import (TB, flatten, init_random, print_metrics,
                           write_metrics)


def run_training(epochs, data_manager, model, optimizer, writer, device='cpu', weight_loss=False):
    metrics = {}
    with torch.no_grad():
        loss, acc = run_iter(model,
                             data_manager,
                             data_manager.tng_loader,
                             device=device,
                             weight_loss=weight_loss)
        metrics['acc/tng'] = acc
        metrics['loss/tng'] = loss

        loss, acc = run_iter(model,
                             data_manager,
                             data_manager.val_loader,
                             device=device,
                             weight_loss=weight_loss)
        metrics['acc/val'] = acc
        metrics['loss/val'] = loss
        print_metrics(0, metrics)
        write_metrics(0, metrics, writer)
        writer.write_csv(metrics, 0)

    final_metrics = {
        'max-acc/val': 0,
        'max-acc/tng': 0,
        'min-loss/val': np.inf,
        'min-loss/tng': np.inf,
    }
    for epoch in trange(1, 1 + epochs):
        loss, acc = run_iter(model,
                             data_manager,
                             data_manager.tng_loader,
                             optimizer,
                             device=device,
                             weight_loss=weight_loss)
        final_metrics['max-acc/tng'] = max(final_metrics['max-acc/tng'], acc)
        final_metrics['min-loss/tng'] = min(final_metrics['min-loss/tng'], loss)
        metrics['acc/tng'] = acc
        metrics['loss/tng'] = loss

        with torch.no_grad():
            loss, acc = run_iter(model,
                                 data_manager,
                                 data_manager.val_loader,
                                 device=device,
                                 weight_loss=weight_loss)
        final_metrics['max-acc/val'] = max(final_metrics['max-acc/val'], acc)
        final_metrics['min-loss/val'] = min(final_metrics['min-loss/val'], loss)
        metrics['acc/val'] = acc
        metrics['loss/val'] = loss
        print_metrics(epoch, metrics)
        write_metrics(epoch, metrics, writer)
        writer.write_csv(metrics, epoch)
        write_metrics(epoch, final_metrics, writer)
    return final_metrics


def run_iter(model, data_manager, data_loader, optimizer=None, device='cpu', weight_loss=False):
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_correct = 0
    total_nodes = 0
    timing = {
        'forward': 0,
        'backward': 0,
        'data': 0,
        'transfer': 0,
    }

    data_time = time.time()
    for batch_size, n_id, adjs in tqdm(data_loader, leave=False):
        timing['data'] += time.time() - data_time

        transfer_time = time.time()
        if isinstance(adjs, torch_geometric.data.sampler.Adj):
            adjs = [adjs]
        adjs = [adj.to(device, non_blocking=True) for adj in adjs]
        node_map = data_manager.get_id_map(n_id)
        node_map = {k: v.to(device, non_blocking=True) for k, v in node_map.items()}
        targets = data_manager.get_targets(n_id[:batch_size, 0])
        targets = targets.to(device, non_blocking=True)
        if optimizer is not None:
            # print_grad(model)
            # zero_grad(model)
            optimizer.zero_grad()
        timing['transfer'] += time.time() - transfer_time
        f_time = time.time()
        out = model(node_map, adjs)
        timing['forward'] += time.time() - f_time
        loss = F.cross_entropy(out,
                               targets,
                               weight=data_manager.target_weights.to(device, non_blocking=True)
                               if weight_loss else None)
        if optimizer is not None:
            b_time = time.time()
            loss.backward()
            optimizer.step()
            timing['backward'] += time.time() - b_time

        total_loss += float(loss.detach()) * batch_size

        y_pred = torch.argmax(out.detach(), dim=-1)
        # print(out)
        # print(y_pred)
        # print(targets)
        total_correct += float((y_pred == targets).sum())
        total_nodes += batch_size

        data_time = time.time()

    loss = total_loss / total_nodes
    acc = 100 * total_correct / total_nodes

    # print('Timing stats:', timing)
    return loss, acc


def main(args):
    init_random(args.seed)
    configs = yaml.safe_load(open(osp.join(args.logdir, 'config.yaml')))
    device = torch.device(args.device)

    data_params = configs['data_params']
    data_manager = DataManager(args.graph, **data_params, workers=args.workers, seed=args.seed)
    model_params = configs['model_params']
    model = Model(data_manager.graph_info, data_manager.neighbor_steps, **model_params)
    model = model.to(device)
    opt_class = getattr(torch.optim, configs['optim'])
    optimizer = opt_class(model.parameters(), **configs['optim_params'])

    writer = TB(log_dir=osp.join(args.logdir, 'seed-' + str(args.seed)), purge_step=0)

    metrics = run_training(args.max_epochs,
                           data_manager,
                           model,
                           optimizer,
                           writer,
                           device=device,
                           weight_loss=configs['weight_loss'])

    writer.add_hparams(flatten(configs), metrics)
    writer.close()


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--logdir')
    PARSER.add_argument('--graph')
    PARSER.add_argument('--seed', type=int, default=0)
    PARSER.add_argument('--workers', type=int, default=2)
    PARSER.add_argument('--device', default='cuda:0')
    PARSER.add_argument('--max-epochs', type=int, default=50)

    ARGS = PARSER.parse_args()
    main(ARGS)
