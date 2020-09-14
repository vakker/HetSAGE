import argparse
import time

import torch
import torch.nn.functional as F
import torch_geometric
from tqdm import tqdm

from hetsage.data import DataManager
from hetsage.model import Model
from hetsage.utils import init_random


def print_grad(model):
    for p in model.parameters():
        if p is not None and p.grad is not None:
            print(torch.norm(p.grad))


def zero_grad(model):
    for p in model.parameters():
        p.grad = None


def run_iter(data_manager, model, optimizer, device, initial=False, weight_loss=False):
    loss_all = {}
    acc_all = {}

    if initial:
        with torch.no_grad():
            loss, acc = _run_iter(model,
                                  data_manager,
                                  data_manager.val_loader,
                                  device=device,
                                  weight_loss=weight_loss)
            loss_all['val0'] = loss
            acc_all['val0'] = acc

    loss, acc = _run_iter(model,
                          data_manager,
                          data_manager.tng_loader,
                          optimizer,
                          device=device,
                          weight_loss=weight_loss)
    loss_all['tng'] = loss
    acc_all['tng'] = acc

    with torch.no_grad():
        loss, acc = _run_iter(model,
                              data_manager,
                              data_manager.val_loader,
                              device=device,
                              weight_loss=weight_loss)
        loss_all['val'] = loss
        acc_all['val'] = acc

    metrics = {'loss': loss_all, 'acc': acc_all}
    return metrics


def _run_iter(model, data_manager, data_loader, optimizer=None, device='cpu', weight_loss=False):
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
    for batch_size, n_id, adjs in tqdm(data_loader):
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
        loss = F.cross_entropy(
            out, targets, weight=data_manager.target_weights.to(device, non_blocking=True) if weight_loss else None)
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
    init_random(args.seed)

    device = torch.device(args.device)

    data_manager = DataManager(
        args.graph,
        args.target,
        batch_size=args.batch_size,
        include_target_label=not args.no_label,
        # neighbor_sizes=[-1] * 5,
        neighbor_sizes=[50, 50],
        workers=args.workers,
    )
    # model = Model(
    #     data_manager.graph_info,
    #     data_manager.neighbor_steps,
    #     emb_hidden=[16, 16],
    #     embed_size=16,
    #     hidden_size=16,
    # )
    # CORA, MovieLens, Muta
    model = Model(
        data_manager.graph_info,
        data_manager.neighbor_steps,
        emb_hidden=[256, 64],
        embed_size=32,
        hidden_size=32,
    )
    model = model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1, 1 + args.max_epochs):
        metrics = run_iter(data_manager,
                           model,
                           optimizer,
                           device=device,
                           initial=epoch == 1,
                           weight_loss=args.weight_loss)
        msg = ''
        msg += f'Epoch {epoch:02d}, '
        for s, v in metrics['acc'].items():
            msg += f'{s} acc: {100*v:.2f}, '
        for s, v in metrics['loss'].items():
            msg += f'{s} loss: {v:.5f}, '
        print(msg)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--graph')
    PARSER.add_argument('--target')
    PARSER.add_argument('--seed', type=int, default=0)
    PARSER.add_argument('--workers', type=int, default=2)
    PARSER.add_argument('--activation', default='LeakyReLU')
    PARSER.add_argument('--batch-size', type=int, default=200)
    PARSER.add_argument('--device', default='cuda:0')
    PARSER.add_argument('--no-label', action='store_true')
    PARSER.add_argument('--weight-loss', action='store_true')
    PARSER.add_argument('--max-epochs', type=int)

    ARGS = PARSER.parse_args()
    main(ARGS)
