"""Offline UAD + CLIP-ReID feature/score fusion.

Both inputs must be .npz files produced by tools/dump_features.py (same keys):
  qf, gf, q_pids, g_pids, q_camids, g_camids, q_modids, g_modids,
  q_keys, g_keys

Usage:
  python tools/fuse_features.py --uad uad_1000.npz --clip clip_1000.npz \
      --mode score --alpha 0.5 --output fusion_1000.json
"""
import argparse
import json

import numpy as np

MAX_RANK = 50


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def align_common(a, b, split):
    """Reorder b's arrays to a's key order."""
    keys_a = a[f'{split}_keys'].astype(str)
    keys_b = b[f'{split}_keys'].astype(str)
    if set(keys_a.tolist()) != set(keys_b.tolist()):
        raise RuntimeError(f'{split} key sets differ between the two dumps')
    pos = {k: i for i, k in enumerate(keys_b.tolist())}
    idx = np.asarray([pos[k] for k in keys_a.tolist()], dtype=np.int64)
    out = {f'{split}f': b[f'{split}f'][idx]}
    for name in ('pids', 'camids', 'modids'):
        out[f'{split}_{name}'] = b[f'{split}_{name}'][idx]
    return out


def eval_block(S, qp, qc, gp, gc, max_rank=MAX_RANK):
    nq = S.shape[0]
    cmc_sum = np.zeros(max_rank, dtype=np.float64)
    aps, inps, firsts = [], [], []
    for i in range(nq):
        s = S[i]
        valid = gc != qc[i]
        sv = s[valid]
        pv = gp[valid]
        rel = pv == qp[i]
        n_rel = int(rel.sum())
        if n_rel == 0:
            continue
        sv_sorted = np.sort(sv)[::-1]
        rel_s = sv[rel]
        ranks = np.searchsorted(-sv_sorted, -rel_s, side='left') + 1
        ranks = np.sort(ranks)
        prec = np.arange(1, n_rel + 1, dtype=np.float64) / ranks
        aps.append(float(prec.mean()))
        first = int(ranks[0])
        last = int(ranks[-1])
        firsts.append(first)
        inps.append(n_rel / float(last))
        c = np.zeros(max_rank, dtype=np.float64)
        c[min(first - 1, max_rank - 1):] = 1.0
        cmc_sum += c
    n = len(aps)
    if n == 0:
        return dict(n=0, rank1=float('nan'), rank5=float('nan'),
                    rank10=float('nan'), mAP=float('nan'),
                    mINP=float('nan'), first_ranks=[], APs=[], INPs=[])
    cmc_sum /= n
    return dict(n=n, rank1=float(cmc_sum[0]), rank5=float(cmc_sum[4]),
                rank10=float(cmc_sum[9]), mAP=float(np.mean(aps)),
                mINP=float(np.mean(inps)), first_ranks=firsts,
                APs=aps, INPs=inps)


def eval_all(S, qp, qc, qm, gp, gc, gm):
    out = {'overall': eval_block(S, qp, qc, gp, gc)}
    pairs = {}
    for qmod in (1, 2, 3):
        qi = np.where(qm == qmod)[0]
        for gmod in (1, 2, 3):
            gi = np.where(gm == gmod)[0]
            pairs[f'{qmod}->{gmod}'] = eval_block(
                S[np.ix_(qi, gi)], qp[qi], qc[qi], gp[gi], gc[gi])
    out['pairs'] = pairs
    ag = {}
    for name, qsel, gsel in (('A->G', qc >= 5, gc < 5),
                             ('G->A', qc < 5, gc >= 5)):
        qi, gi = np.where(qsel)[0], np.where(gsel)[0]
        ag[name] = eval_block(S[np.ix_(qi, gi)], qp[qi], qc[qi],
                              gp[gi], gc[gi])
    out['ag'] = ag
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uad', required=True)
    parser.add_argument('--clip', required=True)
    parser.add_argument('--mode', choices=['score', 'rrf', 'concat'],
                        default='score')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='weight of CLIP in score/concat fusion')
    parser.add_argument('--rrf_k', type=int, default=60)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    ua = dict(np.load(args.uad, allow_pickle=True))
    cb = dict(np.load(args.clip, allow_pickle=True))
    qb = align_common(ua, cb, 'q')
    gb = align_common(ua, cb, 'g')
    Qu, Qc = normalize(ua['qf']), normalize(qb['qf'])
    Gu, Gc = normalize(ua['gf']), normalize(gb['gf'])
    qp = ua['q_pids']; qc = ua['q_camids']; qm = ua['q_modids']
    gp = ua['g_pids']; gc = ua['g_camids']; gm = ua['g_modids']

    Su = Qu @ Gu.T
    Sc = Qc @ Gc.T
    if args.mode == 'score':
        S = (1 - args.alpha) * Su + args.alpha * Sc
    elif args.mode == 'concat':
        a = np.sqrt(1 - args.alpha)
        b = np.sqrt(args.alpha)
        Q = normalize(np.concatenate([a * Qu, b * Qc], axis=1))
        G = normalize(np.concatenate([a * Gu, b * Gc], axis=1))
        S = Q @ G.T
    else:
        Ru = np.empty_like(Su, dtype=np.int32)
        Rc = np.empty_like(Sc, dtype=np.int32)
        for i in range(Su.shape[0]):
            Ru[i] = np.argsort(-Su[i])
            Rc[i] = np.argsort(-Sc[i])
        rank_u = np.empty_like(Ru)
        rank_c = np.empty_like(Rc)
        rank_u[np.arange(len(Ru))[:, None], Ru] = np.arange(Ru.shape[1])
        rank_c[np.arange(len(Rc))[:, None], Rc] = np.arange(Rc.shape[1])
        S = 1.0 / (args.rrf_k + rank_u) + 1.0 / (args.rrf_k + rank_c)

    report = {
        'mode': args.mode,
        'alpha': args.alpha,
        'uad_only': eval_all(Su, qp, qc, qm, gp, gc, gm),
        'clip_only': eval_all(Sc, qp, qc, qm, gp, gc, gm),
        'fused': eval_all(S, qp, qc, qm, gp, gc, gm),
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    for k in ('uad_only', 'clip_only', 'fused'):
        o = report[k]['overall']
        print('{:9s} mAP {:.4f} R1 {:.4f} R5 {:.4f} R10 {:.4f} mINP {:.4f}'.format(
            k, o['mAP'] * 100, o['rank1'] * 100, o['rank5'] * 100,
            o['rank10'] * 100, o['mINP'] * 100))
    print('saved', args.output)


if __name__ == '__main__':
    main()
