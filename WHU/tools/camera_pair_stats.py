"""Estimate same-camera positive/negative pair fractions in PKM batches.

This is a metadata-only diagnostic; it does not run the model.  It answers
whether camera-aware WRT is worth trying and whether down-weighting
same-camera positives would remove cross-modal supervision.

Usage:
  python tools/camera_pair_stats.py --config_file configs/UAD.yml \
      --batches 20
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from datasets.whu_mars import WHU_MARS
from datasets.sampler import PKMSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--batches', type=int, default=20)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    modalities = list(cfg.DATASETS.MODALITIES)
    ds = WHU_MARS(root=cfg.DATASETS.ROOT_DIR, modalities=modalities,
                  dataset_dir=cfg.DATASETS.DATASET_DIR, verbose=False)
    sampler = PKMSampler(ds.train, cfg.SOLVER.IMS_PER_BATCH,
                         cfg.DATALOADER.NUM_INSTANCE, modalities)

    per_batch = cfg.SOLVER.IMS_PER_BATCH
    pos_total = neg_total = pos_same = neg_same = 0
    cross_pos_total = cross_pos_same = 0
    n_batches = 0
    batch_tuples = []
    for tup in sampler:
        batch_tuples.append(tup)
        if len(batch_tuples) < per_batch:
            continue

        pids, cams, mods = [], [], []
        for m_index, modality in enumerate(modalities):
            for tuple_index in batch_tuples:
                _, pid, camid, _ = ds.train[modality][tuple_index[m_index]]
                pids.append(pid)
                cams.append(camid)
                mods.append(m_index + 1)
        pids = np.asarray(pids)
        cams = np.asarray(cams)
        mods = np.asarray(mods)
        n = len(pids)
        is_pos = pids[:, None] == pids[None, :]
        is_neg = ~is_pos
        same_cam = cams[:, None] == cams[None, :]
        np.fill_diagonal(is_pos, False)
        np.fill_diagonal(is_neg, False)
        np.fill_diagonal(same_cam, False)
        pos_total += int(is_pos.sum())
        pos_same += int((is_pos & same_cam).sum())
        neg_total += int(is_neg.sum())
        neg_same += int((is_neg & same_cam).sum())
        cross = mods[:, None] != mods[None, :]
        cross_pos_total += int((is_pos & cross).sum())
        cross_pos_same += int((is_pos & cross & same_cam).sum())
        n_batches += 1
        batch_tuples = []
        if n_batches >= args.batches:
            break

    def pct(a, b):
        return 100.0 * a / max(b, 1)

    print('batches analysed:', n_batches)
    print('images per batch:', per_batch * len(modalities))
    print('positive pairs: {}, same-camera: {} ({:.2f}%)'.format(
        pos_total, pos_same, pct(pos_same, pos_total)))
    print('negative pairs: {}, same-camera: {} ({:.2f}%)'.format(
        neg_total, neg_same, pct(neg_same, neg_total)))
    print('cross-modal positive pairs: {}, same-camera: {} ({:.2f}%)'.format(
        cross_pos_total, cross_pos_same, pct(cross_pos_same, cross_pos_total)))
    print('NOTE: same-camera cross-modal positives are the main TIR/NIR '
          'alignment signal; do NOT blindly down-weight same-camera positives.')


if __name__ == '__main__':
    main()
