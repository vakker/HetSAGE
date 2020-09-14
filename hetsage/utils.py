import collections
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from tqdm import tqdm


def init_random(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def print_grad(model):
    for p in model.parameters():
        if p is not None and p.grad is not None:
            print(torch.norm(p.grad))


def zero_grad(model):
    for p in model.parameters():
        p.grad = None


def write_metrics(epoch, metrics, writer):
    for m, v in metrics.items():
        writer.add_scalar(m, v, epoch)


def print_metrics(epoch, metrics):
    msg = ''
    msg += f'Epoch {epoch:02d}, '
    for m, v in metrics.items():
        msg += f'{m}: {v:.5f}, '
    tqdm.write(msg)


def flatten(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, ', '.join([str(e) for e in v])))
        else:
            items.append((new_key, v))
    return dict(items)


class TB(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            # for k, v in metric_dict.items():
            #     w_hp.add_scalar(k, v)
