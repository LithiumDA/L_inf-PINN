import os
import torch.distributed as dist

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from .equation import HJBEquation

from .derivative_wrapper import build_wrapper

from ..model import MLP
from .data import HJBDataset
from src.utils.glob import setup_logging, config
from src.utils import build_lr
import torch.multiprocessing as mp

def train(rank, world_size, config):
    cfg = config.hjb
    setup(rank, world_size)
    logger = setup_logging()
    logger.info(f"starting {rank}")

    layers = [cfg.equation.x_dim+1] + [cfg.model.width]*(cfg.model.depth-1) + [1]
    g = MLP(layers).to(rank)

    if world_size == 1:
        ddp_g = g
    else:
        ddp_g = DDP(g, device_ids=[rank])

    f = build_wrapper(cfg, ddp_g)

    dataset = HJBDataset(domain_bsz=cfg.train.batch.domain_size, \
        bound_bsz=cfg.train.batch.boundary_size, \
        xdim=cfg.equation.x_dim, T=cfg.equation.T, rank=rank \
        )
    hjb = HJBEquation(cfg.equation.x_dim, cfg.equation.T, cfg.equation.mu)

    if rank == 0:
        with open(cfg.test.data_path, 'rb') as fp:
            test_data = torch.load(fp)
            test_X = test_data['X'].type(torch.FloatTensor)
            test_Y = test_data['Y'].type(torch.FloatTensor)
            test_grad_x = test_data['grad_x'].type(torch.FloatTensor)

    optimizer, scheduler = build_lr(ddp_g, cfg.train, cfg.train.iteration)

    for i in range(cfg.train.iteration):
        f.train()
        optimizer.zero_grad()

        domain_X, boundary_X = dataset.get_online_data()
 
        # adversarial training
        if cfg.train.adversarial.domain_is_adv:
            domain_X = pgd(domain_X, f, hjb.domain_loss, step_cnt=cfg.train.adversarial.grad_step_cnt, \
                step_size=cfg.train.adversarial.grad_step_size, t_lower_bound=0, t_upper_bound=cfg.equation.T)
        if cfg.train.adversarial.boundary_is_adv:
            boundary_X = pgd(boundary_X, f, hjb.boundary_loss, step_cnt=cfg.train.adversarial.grad_step_cnt, \
                step_size=cfg.train.adversarial.grad_step_size, t_lower_bound=cfg.equation.T, t_upper_bound=cfg.equation.T)
        
        dloss = hjb.domain_loss(domain_X, f)
        bloss = hjb.boundary_loss(boundary_X, f)
        loss = cfg.train.loss.domain*dloss + cfg.train.loss.boundary*bloss

        if rank == 0:
            logger.info(f'iteration {i}| loss {loss.detach().cpu().item():.5f}\t| '
                f'domain {dloss.detach().cpu().item():.5f}\t| boundary {bloss.detach().cpu().item():.5f}\t'
            )

        if cfg.model.derivative != 'gt':
            loss.backward()
            optimizer.step()
            scheduler.step()

        if rank==0 and (i+1)%cfg.test.step==0:
            # test the model, only test in one thread.
            i_avg_err, i_rel_err = test(cfg, f, test_X, test_Y, rank, norm_type='l1')
            logger.info(f'L1 test error: average {i_avg_err}, relative {i_rel_err}')
            i_avg_err, i_rel_err = test(cfg, f, test_X, test_Y, rank, norm_type='l2')
            logger.info(f'L2 test error: average {i_avg_err}, relative {i_rel_err}')
            i_avg_err, i_rel_err = test(cfg, f, test_X, torch.concat([test_Y[:, None], test_grad_x], dim=1), rank, norm_type='w11')
            logger.info(f'W11 test error: average {i_avg_err}, relative {i_rel_err}')

    cleanup(rank, world_size)

def pgd(x, f, loss_func, step_cnt=5, step_size=0.2, t_lower_bound=0.0, t_upper_bound=1.0):
    for _ in range(step_cnt):
        x.requires_grad_()
        loss = loss_func(x, f)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + step_size * torch.sign(grad.detach())
        x[:,-1] = torch.clamp(x[:,-1], t_lower_bound, t_upper_bound)
    return x

def test(cfg, f, X, Y, rank, norm_type='l1'):
    if norm_type == 'l1':
        return test_l1(cfg, f, X, Y, rank)
    elif norm_type == 'l2':
        return test_l2(cfg, f, X, Y, rank)
    elif norm_type == 'w11':
        return test_w11(cfg, f, X, Y, rank)
    else:
        raise NotImplementedError
    
def test_l1(cfg, f, X, Y, rank):
    with torch.no_grad():
        f.eval()
        dataloader = DataLoader(TensorDataset(X, Y), batch_size=cfg.test.batch_size)
        tot_err, tot_norm = 0, 0
        for x, y in dataloader:
            x, y = x.to(rank), y.to(rank)
            pred_y = f(x).squeeze()
            err = (pred_y - y).abs().sum()
            y_norm = y.abs().sum()
            tot_err += err.cpu().item()
            tot_norm += y_norm
        avg_err = tot_err/X.shape[0]
        rel_err = tot_err/tot_norm
    return avg_err, rel_err

def test_l2(cfg, f, X, Y, rank):
    with torch.no_grad():
        f.eval()
        dataloader = DataLoader(TensorDataset(X, Y), batch_size=cfg.test.batch_size)
        tot_err, tot_norm = 0, 0
        for x, y in dataloader:
            x, y = x.to(rank), y.to(rank)
            pred_y = f(x).squeeze()
            err = ((pred_y - y)**2).sum()
            y_norm = (y**2).sum()
            tot_err += err.cpu().item()
            tot_norm += y_norm
        tot_err, tot_norm = tot_err**0.5, tot_norm**0.5
        avg_err = tot_err/(X.shape[0]**0.5)
        rel_err = tot_err/tot_norm
    return avg_err, rel_err

def test_w11(cfg, f, X, Y, rank):
    f.eval()
    dataloader = DataLoader(TensorDataset(X, Y), batch_size=cfg.test.batch_size)
    tot_err, tot_norm = 0, 0
    for x, y in dataloader:
        x, y = x.to(rank), y.to(rank)
        pred_y = f(x)
        y_x = f.dx(x)
        pred_y = torch.cat([pred_y, y_x], dim=1)
        err = (pred_y - y).abs().sum()
        y_norm = y.abs().sum()
        tot_err += err.cpu().item()
        tot_norm += y_norm
    avg_err = tot_err/X.shape[0]
    rel_err = tot_err/tot_norm
    return avg_err, rel_err

def setup(rank, world_size):
    if world_size<=1:
        return
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup(rank, world_size):
    if world_size<=1:
        return
    dist.destroy_process_group()

def hjb_training():
    if config.hjb.gpu_cnt == 1:
        train(0, 1, config)
        # supervised_train(0, 1, config)
        # grad_train(0, 1, config)
    else:
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= config.hjb.gpu_cnt, \
            f"Requires at least {config.hjb.gpu_cnt} GPUs to run, but got {n_gpus}"
        world_size = n_gpus

        mp.spawn(train,
                args=(world_size, config),
                nprocs=world_size,
                join=True)
