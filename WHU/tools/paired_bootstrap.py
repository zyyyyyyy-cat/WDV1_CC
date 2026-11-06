"""Paired bootstrap for two per-query result sets in a fuse_features report.

Both result sets must have been evaluated on the same query order (the same
manifest), so first_ranks/APs are paired.  Reports the observed delta and a
95% bootstrap CI for R1/R5/R10/mAP.

Usage:
  python tools/paired_bootstrap.py --report fusion_1000.json \
      --a uad_only --b fused --n 10000
"""
import argparse
import json

import numpy as np


def metrics(first, aps):
    first = np.asarray(first)
    aps = np.asarray(aps)
    return dict(r1=float((first <= 1).mean()),
                r5=float((first <= 5).mean()),
                r10=float((first <= 10).mean()),
                mAP=float(aps.mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--a', default='uad_only')
    parser.add_argument('--b', default='fused')
    parser.add_argument('--n', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    report = json.load(open(args.report, encoding='utf-8'))
    a = report[args.a]['overall']
    b = report[args.b]['overall']
    fa, fb = np.asarray(a['first_ranks']), np.asarray(b['first_ranks'])
    aa, ab = np.asarray(a['APs']), np.asarray(b['APs'])
    if len(fa) != len(fb) or len(aa) != len(ab) or len(fa) != len(aa):
        raise RuntimeError('per-query lists are not paired/length-matched')

    obs_a, obs_b = metrics(fa, aa), metrics(fb, ab)
    print('{}: R1 {:.4f} R5 {:.4f} R10 {:.4f} mAP {:.4f}'.format(
        args.a, obs_a['r1'], obs_a['r5'], obs_a['r10'], obs_a['mAP']))
    print('{}: R1 {:.4f} R5 {:.4f} R10 {:.4f} mAP {:.4f}'.format(
        args.b, obs_b['r1'], obs_b['r5'], obs_b['r10'], obs_b['mAP']))

    rng = np.random.default_rng(args.seed)
    n = len(fa)
    deltas = {k: [] for k in ('r1', 'r5', 'r10', 'mAP')}
    for _ in range(args.n):
        idx = rng.integers(0, n, size=n)
        ma, mb = metrics(fa[idx], aa[idx]), metrics(fb[idx], ab[idx])
        for k in deltas:
            deltas[k].append(mb[k] - ma[k])
    print('observed and 95% CI for ({}-{}):'.format(args.b, args.a))
    for k in ('r1', 'r5', 'r10', 'mAP'):
        obs = obs_b[k] - obs_a[k]
        lo, hi = np.percentile(deltas[k], [2.5, 97.5])
        print('  {:3s}: {:+.4f}  CI [{:+.4f}, {:+.4f}]'.format(k, obs, lo, hi))


if __name__ == '__main__':
    main()
