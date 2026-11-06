"""Verify that a UAD checkpoint is fully loaded by the finetune path.

Usage:
  python tools/check_checkpoint_load.py --config_file configs/LRA_UAD_A0.yml \
      --checkpoint logs/UAD_1000_DDP/transformer_120.pth --num_classes 500
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from model import make_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num_classes', type=int, default=500)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(['MODEL.PRETRAIN_CHOICE', 'finetune',
                         'MODEL.PRETRAIN_PATH', args.checkpoint])
    cfg.freeze()

    model = make_model(cfg, num_class=args.num_classes,
                       camera_num=0, view_num=0)
    model.eval()

    try:
        obj = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        obj = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in obj:
        obj = obj['state_dict']
    if 'model' in obj:
        obj = obj['model']
    ckpt = {k.replace('module.', ''): v for k, v in obj.items()}

    own = model.state_dict()
    missing = [k for k in own if k not in ckpt]
    unexpected = [k for k in ckpt if k not in own]
    mismatched = [k for k in own
                  if k in ckpt and own[k].shape != ckpt[k].shape]
    max_diffs = {}
    for k in own:
        if k in ckpt and own[k].shape == ckpt[k].shape \
                and own[k].dtype.is_floating_point:
            max_diffs[k] = float((own[k] - ckpt[k]).abs().max())
    worst = sorted(max_diffs.items(), key=lambda x: -x[1])[:10]
    print('model keys:', len(own), 'checkpoint keys:', len(ckpt))
    print('missing:', len(missing), missing[:10])
    print('unexpected:', len(unexpected), unexpected[:10])
    print('shape mismatch:', len(mismatched), mismatched[:10])
    print('largest load differences:')
    for k, v in worst:
        print('  {:.3e} {}'.format(v, k))
    if not missing and not mismatched and all(v < 1e-7 for v in max_diffs.values()):
        print('PASS: all model parameters match the checkpoint.')
    else:
        print('CHECK: inspect missing/mismatched keys above.')


if __name__ == '__main__':
    main()
