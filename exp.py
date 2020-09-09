import argparse
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

import torch_geometric
from hetsage.data import DataManager
from hetsage.model import Model
from hetsage.utils import init_random


def zero_grad(model):
    for p in model.parameters():
        p.grad = None


def run_iter(data_manager, model, optimizer, device):
    metrics = {}
    loss, acc = _run_iter(model, data_manager, data_manager.tng_loader, optimizer, device=device)
    metrics['tng-loss'] = loss
    metrics['tng-acc'] = acc

    with torch.no_grad():
        # loss, acc = _run_iter(model, model.tng_loader, device=device)
        # metrics['tng2-loss'] = loss
        # metrics['tng2-acc'] = acc
        loss, acc = _run_iter(model, data_manager, data_manager.val_loader, device=device)
        metrics['val-loss'] = loss
        metrics['val-acc'] = acc

    return metrics


def _run_iter(model, data_manager, data_loader, optimizer=None, device='cpu'):
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_correct = 0
    total_nodes = 0
    timing = {'forward': 0, 'backward': 0, 'data': 0}

    data_time = time.time()
    for batch_size, n_id, adjs in tqdm(data_loader):
        timing['data'] += time.time() - data_time

        if isinstance(adjs, torch_geometric.data.sampler.Adj):
            adjs = [adjs]
        adjs = [adj.to(device) for adj in adjs]
        node_map = data_manager.get_id_map(n_id)
        node_map = {k: v.to(device) for k, v in node_map.items()}
        targets = data_manager.get_targets(n_id[:batch_size, 0])
        targets = targets.to(device)
        if optimizer is not None:
            # zero_grad(model)
            optimizer.zero_grad()
        f_time = time.time()
        out = model(node_map, adjs)
        timing['forward'] += time.time() - f_time
        loss = F.nll_loss(F.log_softmax(out, dim=-1), targets)
        if optimizer is not None:
            b_time = time.time()
            loss.backward()
            optimizer.step()
            timing['backward'] += time.time() - b_time

        total_loss += float(loss.detach()) * batch_size

        y_pred = torch.argmax(out.detach(), dim=-1)
        total_correct += float((y_pred == targets).sum())
        total_nodes += batch_size

        data_time = time.time()

    loss = total_loss / total_nodes
    approx_acc = total_correct / total_nodes

    # print('Timing stats:', timing)
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

    if args.use_gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    data_manager = DataManager(args.gml,
                               args.target,
                               include_target_label=not args.no_label,
                               # neighbor_sizes=[10, 10, 10, 10],
                               neighbor_sizes=[10, 10],
                               workers=args.workers)
    model = Model(
        data_manager.graph_info,
        data_manager.neighbor_steps,
        embed_size=16,
        emb_hidden=[256, 64],
        hidden_size=32,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(1, 1 + args.max_epochs):
        metrics = run_iter(data_manager, model, optimizer, device=device)
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
    PARSER.add_argument('--workers', type=int, default=2)
    PARSER.add_argument('--use-gpu', action='store_true')
    PARSER.add_argument('--no-label', action='store_true')
    PARSER.add_argument('--max-epochs', type=int)

    ARGS = PARSER.parse_args()
    main(ARGS)
