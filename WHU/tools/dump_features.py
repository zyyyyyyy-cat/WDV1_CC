"""Dump UAD query/gallery features for offline fusion and diagnostics.

Usage (single GPU):
  CUDA_VISIBLE_DEVICES=0 python tools/dump_features.py \
      --config_file configs/UAD.yml \
      --weight logs/UAD_1000_DDP/transformer_120.pth \
      --output /path/uad_1000.npz

Saved keys: qf/gf (float32), q_pids/g_pids, q_camids/g_camids,
q_modids/g_modids, q_keys/g_keys.  A key is "<modality>/<basename>".
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from datasets import make_dataloader
from model import make_model


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--weight', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--tta_flip', action='store_true')
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    # Build a random-init model and load exactly the requested checkpoint.
    # This avoids depending on the config's PRETRAIN_PATH (which may be an
    # ImageNet path or a different experiment's weight).
    cfg.merge_from_list(['MODEL.DIST_TRAIN', 'False',
                         'MODEL.PRETRAIN_CHOICE', 'none',
                         'TEST.WEIGHT', args.weight])
    if args.batch_size is not None:
        cfg.merge_from_list(['TEST.IMS_PER_BATCH', str(args.batch_size)])
    cfg.merge_from_list(['TEST.TTA_FLIP', str(args.tta_flip)])
    cfg.freeze()

    train_loader, train_normal_loaders, val_loaders, num_querys, \
        num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes,
                       camera_num=camera_num, view_num=view_num)
    model.load_param_finetune(args.weight)
    device = 'cuda'
    model.to(device).eval()

    qf, gf = [], []
    qp, gp, qc, gc, qm, gm, qk, gk = [], [], [], [], [], [], [], []
    for mode, (loader, num_query) in enumerate(zip(val_loaders, num_querys), 1):
        feats, pids, cams, mods, keys = [], [], [], [], []
        with torch.no_grad():
            for img, vid, camid, camidt, modid, paths in loader:
                img = img.to(device)
                feat = model(img, mode=mode)
                if cfg.TEST.TTA_FLIP:
                    feat_flip = model(torch.flip(img, dims=[3]), mode=mode)
                    feat = F.normalize(feat, dim=1) + F.normalize(feat_flip, dim=1)
                feats.append(feat.float().cpu())
                pids.extend(list(vid))
                cams.extend(list(camid))
                mods.extend(list(modid))
                keys.extend(['{}/{}'.format(mode, os.path.basename(p))
                             for p in paths])
        feats = torch.cat(feats, dim=0)
        qf.append(feats[:num_query])
        gf.append(feats[num_query:])
        qp.extend(pids[:num_query]); gp.extend(pids[num_query:])
        qc.extend(cams[:num_query]); gc.extend(cams[num_query:])
        qm.extend(mods[:num_query]); gm.extend(mods[num_query:])
        qk.extend(keys[:num_query]); gk.extend(keys[num_query:])
        print('mode {}: query {}, gallery {}'.format(
            mode, num_query, len(pids) - num_query))

    np.savez_compressed(
        args.output,
        qf=torch.cat(qf).numpy().astype(np.float32),
        gf=torch.cat(gf).numpy().astype(np.float32),
        q_pids=np.asarray(qp, dtype=np.int64),
        g_pids=np.asarray(gp, dtype=np.int64),
        q_camids=np.asarray(qc, dtype=np.int64),
        g_camids=np.asarray(gc, dtype=np.int64),
        q_modids=np.asarray(qm, dtype=np.int64),
        g_modids=np.asarray(gm, dtype=np.int64),
        q_keys=np.asarray(qk),
        g_keys=np.asarray(gk),
        weight_sha256=np.asarray(_sha256(args.weight)),
        config=np.asarray(args.config_file),
    )
    print('saved', args.output)


if __name__ == '__main__':
    main()
