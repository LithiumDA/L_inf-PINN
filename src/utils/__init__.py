import torch

def build_lr(model, cfg, tot_epoch=-1):
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    if cfg.scheduler.name == 'linear':
        if tot_epoch == -1:
            tot_epoch = cfg.epoch
        warm_epoch = cfg.scheduler.warm_up_steps
        lr_lambda = lambda epoch: 1-(epoch-warm_epoch)/(tot_epoch-warm_epoch) if epoch >= warm_epoch \
                            else epoch/warm_epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR( \
            optimizer_adam, \
            lr_lambda=lr_lambda, \
        )
    elif cfg.scheduler.name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_adam, step_size=cfg.scheduler.step_size, 
                                                    gamma=cfg.scheduler.gamma)
    else:
        # Default is constant learning rate
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_adam, lr_lambda=lambda epoch:1)
    return optimizer_adam, scheduler
