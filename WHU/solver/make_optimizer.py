import re

import torch


def _llrd_scale(key, decay):
    """Layer-wise LR scale for the ViT backbone. decay=0 disables LLRD."""
    if decay <= 0:
        return 1.0
    m = re.match(r'^base\.blocks\.(\d+)\.', key)
    if m:
        # blocks.0 is the earliest/shallowest layer.
        return decay ** (11 - int(m.group(1)))
    if key.startswith('base.patch_embed'):
        return decay ** 12
    if key in ('base.cls_token', 'base.pos_embed'):
        return 0.1
    if key.startswith('base.norm'):
        return 0.5
    return 1.0


def make_optimizer(cfg, model, center_criterion):
    params = []
    decay = getattr(cfg.MODEL, 'LLRD_DECAY', 0.0)
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR * _llrd_scale(key, decay)
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = lr * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center
