"""Offline retrieval lab for UAD / WHU-MARS-1000.

Loads the cached UAD pre-BN CLS features and reproduces the official
evaluation protocol (L2 normalization, squared Euclidean, exclude all
gallery images from the query camera) without running the model.

Also evaluates training-free score-calibration variants that only change
how similarities from different gallery modalities are merged for the
overall AS-ReID protocol. Pair protocols are unaffected by monotone
per-modality transforms, so they are not recomputed for each variant.

Usage:
  python offline_eval.py --limit 256       # quick sanity check
  python offline_eval.py                   # full 6405 queries
  python offline_eval.py --variant zscore  # one variant
"""
import argparse
import json
import os
import time
import numpy as np

FEAT_NPZ = os.path.join(os.path.dirname(__file__),
                        "..", "..", "..", "WHU_Diagnosis", "WHU_1000",
                        "features_bb7907c940c6.npz")
FEAT_NPZ = os.path.abspath(FEAT_NPZ)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_RANK = 50


def load_features():
    d = np.load(FEAT_NPZ, allow_pickle=True)
    f = d["features"].astype(np.float32)
    pids = d["pids"].astype(np.int64)
    cams = d["camids"].astype(np.int64)
    mods = d["modids"].astype(np.int64)
    splits = d["splits"].astype(str)
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    q = splits == "query"
    g = splits == "gallery"
    return (f[q], pids[q], cams[q], mods[q],
            f[g], pids[g], cams[g], mods[g])


def _eval_one_query(d, g_pids, g_cams, q_pid, q_cam, max_rank=MAX_RANK):
    """d: distances (lower better) to all gallery images."""
    valid = g_cams != q_cam
    dv = d[valid]
    pv = g_pids[valid]
    rel = pv == q_pid
    n_rel = int(rel.sum())
    if n_rel == 0:
        return None
    # ranks of relevant items among valid gallery (1-based)
    order = np.argsort(dv, kind="stable")
    dv_sorted = dv[order]
    rel_d = dv[rel]
    ranks = np.searchsorted(dv_sorted, rel_d, side="left") + 1
    ranks = np.sort(ranks)
    # AP = mean precision at each relevant rank
    prec = np.arange(1, n_rel + 1, dtype=np.float64) / ranks
    ap = float(prec.mean())
    first = int(ranks[0])
    last = int(ranks[-1])
    cmc = np.zeros(max_rank, dtype=np.float64)
    # CMC is 1 from the first relevant rank onward.
    cmc[min(first - 1, max_rank - 1):] = 1.0
    inp = n_rel / float(last)
    return cmc, ap, inp, first


def evaluate(D, q_pids, q_cams, g_pids, g_cams, max_rank=MAX_RANK):
    """D: (nq, ng) distances. Returns R1,R5,R10,mAP,mINP, and per-query ranks."""
    nq = D.shape[0]
    cmc_sum = np.zeros(max_rank, dtype=np.float64)
    aps = []
    inps = []
    firsts = np.full(nq, -1, dtype=np.int64)
    for i in range(nq):
        out = _eval_one_query(D[i], g_pids, g_cams, q_pids[i], q_cams[i], max_rank)
        if out is None:
            continue
        cmc, ap, inp, first = out
        cmc_sum += cmc
        aps.append(ap)
        inps.append(inp)
        firsts[i] = first
    n = len(aps)
    cmc_sum /= max(n, 1)
    return (float(cmc_sum[0]), float(cmc_sum[4]), float(cmc_sum[9]),
            float(np.mean(aps)), float(np.mean(inps)), firsts, n)


def calibrate(S, mode, topk=1000):
    """S: (chunk, n_gallery_of_one_modality) similarities, higher better.
    Per-query normalization across the gallery of one modality."""
    if mode == "raw":
        return S
    if mode == "mean":
        return S - S.mean(axis=1, keepdims=True)
    if mode == "zscore":
        mu = S.mean(axis=1, keepdims=True)
        sd = S.std(axis=1, keepdims=True) + 1e-6
        return (S - mu) / sd
    if mode == "std":
        sd = S.std(axis=1, keepdims=True) + 1e-6
        return S / sd
    if mode == "robust":
        med = np.median(S, axis=1, keepdims=True)
        mad = np.median(np.abs(S - med), axis=1, keepdims=True) + 1e-6
        return (S - med) / mad
    if mode == "topk_z":
        k = min(topk, S.shape[1])
        part = np.partition(S, -k, axis=1)[:, -k:]
        mu = part.mean(axis=1, keepdims=True)
        sd = part.std(axis=1, keepdims=True) + 1e-6
        return (S - mu) / sd
    if mode == "topk_mean":
        k = min(topk, S.shape[1])
        part = np.partition(S, -k, axis=1)[:, -k:]
        return S - part.mean(axis=1, keepdims=True)
    raise ValueError(mode)


def run(mode, limit=None, chunk=128, topk=1000, verbose=True):
    (qf, qp, qc, qm, gf, gp, gc, gm) = load_features()
    nq = qf.shape[0] if limit is None else min(limit, qf.shape[0])
    qf, qp, qc, qm = qf[:nq], qp[:nq], qc[:nq], qm[:nq]
    # per-modality gallery blocks (all three modalities are present)
    gblocks = []
    for m in (1, 2, 3):
        idx = np.where(gm == m)[0]
        gblocks.append((idx, gf[idx]))
    # D columns are concatenated modality-major, so reorder gallery metadata
    # to the same order before evaluation.
    gorder = np.concatenate([idx for idx, _ in gblocks])
    gp, gc, gm = gp[gorder], gc[gorder], gm[gorder]
    # distances: use -similarity so lower is better
    all_D = []
    t0 = time.time()
    for start in range(0, nq, chunk):
        end = min(start + chunk, nq)
        qc_chunk = qf[start:end]
        parts = []
        for idx, gb in gblocks:
            S = qc_chunk @ gb.T
            S = calibrate(S, mode, topk=topk)
            parts.append(-S)  # lower better
        D = np.concatenate(parts, axis=1)
        all_D.append(D)
        if verbose:
            print(f"[{mode}] {end}/{nq} elapsed {time.time()-t0:.1f}s", flush=True)
    D = np.concatenate(all_D, axis=0)
    r1, r5, r10, mAP, mINP, firsts, n = evaluate(D, qp, qc, gp, gc)
    return dict(mode=mode, n=n, r1=r1, r5=r5, r10=r10, mAP=mAP, mINP=mINP,
                firsts=firsts.tolist(), seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--topk", type=int, default=1000)
    args = ap.parse_args()
    variants = ["raw", "mean", "zscore", "std", "robust", "topk_z", "topk_mean"]
    if args.variant != "all":
        variants = [args.variant]
    results = []
    for v in variants:
        res = run(v, limit=args.limit, chunk=args.chunk, topk=args.topk)
        results.append(res)
        print(json.dumps({k: res[k] for k in
                          ("mode", "n", "r1", "r5", "r10", "mAP", "mINP", "seconds")},
                         indent=2), flush=True)
    out = os.path.join(OUT_DIR, f"offline_calibration_{args.variant}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
