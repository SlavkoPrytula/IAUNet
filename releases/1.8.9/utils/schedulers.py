from torch.optim import lr_scheduler


def fetch_scheduler(cfg, optimizer):
    scheduler = None

    if cfg.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max,
                                                   eta_min=cfg.min_lr)
    elif cfg.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.T_0,
                                                             eta_min=cfg.min_lr)
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=cfg.min_lr, )
    elif cfg.scheduler == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    elif cfg.scheduler is None:
        return None

    return scheduler
