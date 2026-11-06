# encoding: utf-8

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .wrt_loss import TripletLoss_WRT
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        triplet = TripletLoss()
        print("using soft triplet loss for training")
    elif 'wrt' in cfg.MODEL.METRIC_LOSS_TYPE:
        triplet = TripletLoss_WRT(
            use_camera=cfg.MODEL.WRT_CAMERA,
            cam_neg_weight=cfg.MODEL.WRT_CAM_NEG_WEIGHT,
        )
        print("using WRT loss for training (camera-aware: {})".format(
            cfg.MODEL.WRT_CAMERA))
    else:
        raise ValueError('expected METRIC_LOSS_TYPE should be triplet or wrt but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler.upper() != 'PKM':
        raise ValueError('expected PKM sampler, but got {}'.format(sampler))

    def loss_func(score, feat, target, camids=None, epoch=0):
        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            ID_LOSS = xent(score, target)
        else:
            ID_LOSS = F.cross_entropy(score, target)

        beta = cfg.MODEL.WRT_NORM_BETA
        if beta > 0 and cfg.MODEL.WRT_NORM_WARMUP > 0:
            beta = beta * min(1.0, float(epoch) / cfg.MODEL.WRT_NORM_WARMUP)
        if 'wrt' in cfg.MODEL.METRIC_LOSS_TYPE:
            TRI_LOSS = triplet(feat, target, camids=camids, norm_beta=beta)[0]
        else:
            # Plain TripletLoss does not accept camids/norm_beta.
            TRI_LOSS = triplet(feat, target)[0]
        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS, ID_LOSS, TRI_LOSS

    return loss_func, center_criterion
