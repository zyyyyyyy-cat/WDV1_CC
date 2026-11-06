"""Offline (training-free / post-hoc) feature-transform lab for UAD.

Fits transforms on TRAIN features, applies them to query/gallery features,
and evaluates the official WHU-MARS-1000 protocol (overall, 9 modality
pairs, AG-ReID). No model training or inference is performed.

Usage:
  python offline_transforms.py --transform zca_mod
  python offline_transforms.py --all
  python offline_transforms.py --all --fit gallery   # transductive (diagnostic)
"""
import argparse
import json
import os
import time
import numpy as np

import offline_eval as oe

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_RANK = 50


def load_all():
    d = np.load(oe.FEAT_NPZ, allow_pickle=True)
    f = d["features"].astype(np.float32)
    pids = d["pids"].astype(np.int64)
    cams = d["camids"].astype(np.int64)
    mods = d["modids"].astype(np.int64)
    splits = d["splits"].astype(str)
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    q = splits == "query"
    g = splits == "gallery"
    t = splits == "train"
    return (f[t], pids[t], cams[t], mods[t],
            f[q], pids[q], cams[q], mods[q],
            f[g], pids[g], cams[g], mods[g])


def sqrtm_psd(S, eps=1e-6):
    w, V = np.linalg.eigh(S.astype(np.float64))
    w = np.clip(w, eps, None)
    return (V * np.sqrt(w)) @ V.T


def invsqrtm_psd(S, eps=1e-6):
    w, V = np.linalg.eigh(S.astype(np.float64))
    w = np.clip(w, eps, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def mean_cov(X, eps=1e-4):
    mu = X.mean(axis=0)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(len(X) - 1, 1)
    C = C.astype(np.float64)
    # shrinkage toward diagonal for numerical stability
    tr = np.trace(C) / C.shape[0]
    C = (1 - eps) * C + eps * np.eye(C.shape[0], dtype=np.float64) * tr
    return mu, C


def fit(name, ft, pt, ct, mt, d=256, eps=1e-4):
    """Return a transform dict fitted on training features."""
    params = {"name": name}
    if name == "none":
        return params
    if name == "center_global":
        params["mu"] = ft.mean(0)
        return params
    if name == "center_mod":
        params["mu"] = {m: ft[mt == m].mean(0) for m in (1, 2, 3)}
        return params
    if name == "center_cam":
        params["mu"] = {c: ft[ct == c].mean(0) for c in sorted(set(ct.tolist()))}
        return params
    if name in ("zca_global", "pca_global"):
        mu, C = mean_cov(ft)
        params["mu"] = mu
        if name == "zca_global":
            params["W"] = invsqrtm_psd(C).astype(np.float32)
        else:
            w, V = np.linalg.eigh(C)
            idx = np.argsort(w)[::-1][:d]
            params["W"] = V[:, idx].astype(np.float32)
        return params
    if name in ("zca_mod", "pca_mod"):
        params["mu"], params["W"] = {}, {}
        for m in (1, 2, 3):
            mu, C = mean_cov(ft[mt == m])
            params["mu"][m] = mu
            if name == "zca_mod":
                params["W"][m] = invsqrtm_psd(C).astype(np.float32)
            else:
                w, V = np.linalg.eigh(C)
                idx = np.argsort(w)[::-1][:d]
                params["W"][m] = V[:, idx].astype(np.float32)
        return params
    if name == "coral_mod_to_rgb":
        mu_r, C_r = mean_cov(ft[mt == 1])
        Sr = sqrtm_psd(C_r)
        params["mu_ref"] = mu_r
        params["mu"], params["W"] = {}, {}
        for m in (1, 2, 3):
            mu, C = mean_cov(ft[mt == m])
            params["mu"][m] = mu
            params["W"][m] = (invsqrtm_psd(C) @ Sr).astype(np.float32)
        return params
    if name == "center_mod_cam":
        params["mu"] = {}
        for m in (1, 2, 3):
            for c in sorted(set(ct.tolist())):
                sel = (mt == m) & (ct == c)
                if sel.sum() >= 8:
                    params["mu"][(m, c)] = ft[sel].mean(0)
        return params
    if name in ("center_platform", "zca_platform", "pca_platform"):
        pt = (ct >= 5).astype(np.int64)
        params["mu"] = {}
        params["W"] = {}
        for p in (0, 1):
            sel = pt == p
            mu, C = mean_cov(ft[sel])
            params["mu"][p] = mu
            if name == "zca_platform":
                params["W"][p] = invsqrtm_psd(C).astype(np.float32)
            elif name == "pca_platform":
                w, V = np.linalg.eigh(C)
                idx = np.argsort(w)[::-1][:d]
                params["W"][p] = V[:, idx].astype(np.float32)
        return params
    if name in ("center_mod_platform", "zca_mod_platform", "pca_mod_platform"):
        pt = (ct >= 5).astype(np.int64)
        params["mu"], params["W"] = {}, {}
        for m in (1, 2, 3):
            for p in (0, 1):
                sel = (mt == m) & (pt == p)
                mu, C = mean_cov(ft[sel])
                params["mu"][(m, p)] = mu
                if name == "zca_mod_platform":
                    params["W"][(m, p)] = invsqrtm_psd(C).astype(np.float32)
                elif name == "pca_mod_platform":
                    w, V = np.linalg.eigh(C)
                    idx = np.argsort(w)[::-1][:d]
                    params["W"][(m, p)] = V[:, idx].astype(np.float32)
        return params
    if name == "coral_platform_to_ground":
        pt = (ct >= 5).astype(np.int64)
        params["mu"], params["W"] = {}, {}
        for m in (1, 2, 3):
            mug, Cg = mean_cov(ft[(mt == m) & (pt == 0)])
            mua, Ca = mean_cov(ft[(mt == m) & (pt == 1)])
            Sg = sqrtm_psd(Cg)
            params["mu"][(m, 0)] = mug
            params["W"][(m, 0)] = np.eye(ft.shape[1], dtype=np.float32)
            params["mu"][(m, 1)] = mua
            params["W"][(m, 1)] = (invsqrtm_psd(Ca) @ Sg).astype(np.float32)
        return params
    if name == "coral_mod_platform_to_rgb_ground":
        pt = (ct >= 5).astype(np.int64)
        mu0, C0 = mean_cov(ft[(mt == 1) & (pt == 0)])
        S0 = sqrtm_psd(C0)
        params["mu_ref"] = mu0
        params["mu"], params["W"] = {}, {}
        for m in (1, 2, 3):
            for p in (0, 1):
                mu, C = mean_cov(ft[(mt == m) & (pt == p)])
                params["mu"][(m, p)] = mu
                params["W"][(m, p)] = (invsqrtm_psd(C) @ S0).astype(np.float32)
        return params
    raise ValueError(name)


def apply(f, cams, mods, p):
    name = p["name"]
    if name == "none":
        return f
    if name == "center_global":
        return f - p["mu"]
    if name == "center_mod":
        out = f.copy()
        for m in (1, 2, 3):
            out[mods == m] -= p["mu"][m]
        return out
    if name == "center_cam":
        out = f.copy()
        for c, mu in p["mu"].items():
            out[cams == c] -= mu
        return out
    if name in ("zca_global", "pca_global"):
        return (f - p["mu"]) @ p["W"]
    if name in ("zca_mod", "pca_mod"):
        out = np.empty_like(f)
        for m in (1, 2, 3):
            sel = mods == m
            out[sel] = (f[sel] - p["mu"][m]) @ p["W"][m]
        return out
    if name == "coral_mod_to_rgb":
        out = np.empty_like(f)
        for m in (1, 2, 3):
            sel = mods == m
            out[sel] = (f[sel] - p["mu"][m]) @ p["W"][m] + p["mu_ref"]
        return out
    if name == "center_mod_cam":
        out = f.copy()
        for (m, c), mu in p["mu"].items():
            out[(mods == m) & (cams == c)] -= mu
        return out
    if name == "center_platform":
        pt = (cams >= 5).astype(np.int64)
        out = f.copy()
        for p_ in (0, 1):
            out[pt == p_] -= p["mu"][p_]
        return out
    if name in ("zca_platform", "pca_platform"):
        pt = (cams >= 5).astype(np.int64)
        out = np.empty_like(f)
        for p_ in (0, 1):
            sel = pt == p_
            out[sel] = (f[sel] - p["mu"][p_]) @ p["W"][p_]
        return out
    if name in ("center_mod_platform", "zca_mod_platform", "pca_mod_platform"):
        pt = (cams >= 5).astype(np.int64)
        out = f.copy()
        for m in (1, 2, 3):
            for p_ in (0, 1):
                sel = (mods == m) & (pt == p_)
                if sel.sum() == 0:
                    continue
                if name == "center_mod_platform":
                    out[sel] = f[sel] - p["mu"][(m, p_)]
                else:
                    out[sel] = (f[sel] - p["mu"][(m, p_)]) @ p["W"][(m, p_)]
        return out
    if name == "coral_platform_to_ground":
        pt = (cams >= 5).astype(np.int64)
        out = np.empty_like(f)
        for m in (1, 2, 3):
            for p_ in (0, 1):
                sel = (mods == m) & (pt == p_)
                if sel.sum() == 0:
                    continue
                out[sel] = (f[sel] - p["mu"][(m, p_)]) @ p["W"][(m, p_)] \
                    + p["mu"][(m, 0)]
        return out
    if name == "coral_mod_platform_to_rgb_ground":
        pt = (cams >= 5).astype(np.int64)
        out = np.empty_like(f)
        for m in (1, 2, 3):
            for p_ in (0, 1):
                sel = (mods == m) & (pt == p_)
                if sel.sum() == 0:
                    continue
                out[sel] = (f[sel] - p["mu"][(m, p_)]) @ p["W"][(m, p_)] \
                    + p["mu_ref"]
        return out
    raise ValueError(name)


def evaluate_overall(Q, G, qp, qc, gp, gc, chunk=256):
    nq = Q.shape[0]
    cmc_sum = np.zeros(MAX_RANK)
    aps, inps = [], []
    for s in range(0, nq, chunk):
        e = min(s + chunk, nq)
        D = -(Q[s:e] @ G.T)
        for i in range(e - s):
            out = oe._eval_one_query(D[i], gp, gc, qp[s + i], qc[s + i])
            if out is None:
                continue
            c, a, ip, _ = out
            cmc_sum += c
            aps.append(a)
            inps.append(ip)
    n = len(aps)
    cmc_sum /= max(n, 1)
    return dict(r1=float(cmc_sum[0]), r5=float(cmc_sum[4]), r10=float(cmc_sum[9]),
                mAP=float(np.mean(aps)), mINP=float(np.mean(inps)), n=n)


def evaluate_pairs(Q, G, qp, qc, qm, gp, gc, gm):
    out = {}
    for qmod in (1, 2, 3):
        qi = np.where(qm == qmod)[0]
        for gmod in (1, 2, 3):
            gi = np.where(gm == gmod)[0]
            D = -(Q[qi] @ G[gi].T)
            r1, r5, r10, mAP, mINP, first, n = oe.evaluate(
                D, qp[qi], qc[qi], gp[gi], gc[gi])
            out[f"{qmod}->{gmod}"] = dict(r1=r1, r5=r5, r10=r10,
                                          mAP=mAP, mINP=mINP, n=n)
    return out


def evaluate_ag(Q, G, qp, qc, gp, gc):
    out = {}
    for name, qsel, gsel in (("A->G", qc >= 5, gc < 5), ("G->A", qc < 5, gc >= 5)):
        qi, gi = np.where(qsel)[0], np.where(gsel)[0]
        D = -(Q[qi] @ G[gi].T)
        r1, r5, r10, mAP, mINP, first, n = oe.evaluate(
            D, qp[qi], qc[qi], gp[gi], gc[gi])
        out[name] = dict(r1=r1, r10=r10, mAP=mAP, mINP=mINP, n=n)
    return out


def run_transform(name, fit_on="train", d=256):
    ft, pt, ct, mt, qf, qp, qc, qm, gf, gp, gc, gm = load_all()
    if fit_on == "train":
        p = fit(name, ft, pt, ct, mt, d=d)
    elif fit_on == "gallery":
        p = fit(name, gf, gp, gc, gm, d=d)
    else:
        raise ValueError(fit_on)
    t0 = time.time()
    Q = apply(qf, qc, qm, p)
    G = apply(gf, gc, gm, p)
    Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    G = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-12)
    overall = evaluate_overall(Q, G, qp, qc, gp, gc)
    pairs = evaluate_pairs(Q, G, qp, qc, qm, gp, gc, gm)
    ag = evaluate_ag(Q, G, qp, qc, gp, gc)
    return dict(transform=name, fit_on=fit_on, d=d, overall=overall,
                pairs=pairs, ag=ag, seconds=time.time() - t0)


def run_qe(k=10, alpha=3.0, per_modality=False, exclude_same_cam=True, lam=1.0):
    ft, pt, ct, mt, qf, qp, qc, qm, gf, gp, gc, gm = load_all()
    t0 = time.time()
    Q = qf.copy()
    G = gf.copy()
    nq = Q.shape[0]
    for i in range(nq):
        s = Q[i] @ G.T
        if exclude_same_cam:
            s = s.copy()
            s[gc == qc[i]] = -np.inf
        sel = np.argpartition(-s, k)[:k]
        w = np.maximum(s[sel], 0) ** alpha
        if w.sum() <= 0:
            continue
        new = Q[i] + lam * (G[sel] * w[:, None]).sum(0) / max(w.sum(), 1e-12)
        Q[i] = new / max(np.linalg.norm(new), 1e-12)
    overall = evaluate_overall(Q, G, qp, qc, gp, gc)
    pairs = evaluate_pairs(Q, G, qp, qc, qm, gp, gc, gm)
    ag = evaluate_ag(Q, G, qp, qc, gp, gc)
    return dict(transform=f"qe_k{k}_a{alpha}_lam{lam}_excl{int(exclude_same_cam)}",
                overall=overall, pairs=pairs, ag=ag, seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transform", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--qe", action="store_true")
    ap.add_argument("--fit", default="train", choices=["train", "gallery"])
    ap.add_argument("--d", type=int, default=256)
    args = ap.parse_args()
    names = ["none", "center_global", "center_mod", "center_cam",
             "center_mod_cam", "zca_global", "zca_mod", "pca_global",
             "pca_mod", "coral_mod_to_rgb"]
    if args.transform:
        names = [args.transform]
    if args.all:
        names = names
    results = []
    for n in names:
        try:
            r = run_transform(n, fit_on=args.fit, d=args.d)
            results.append(r)
            print(json.dumps(r, indent=2), flush=True)
        except Exception as e:
            print("ERR", n, repr(e), flush=True)
    if args.qe:
        for k, a in ((5, 1.0), (10, 1.0), (10, 3.0), (20, 3.0)):
            r = run_qe(k=k, alpha=a)
            results.append(r)
            print(json.dumps(r, indent=2), flush=True)
    out = os.path.join(OUT_DIR, "offline_transforms_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
