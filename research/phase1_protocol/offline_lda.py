"""Post-hoc supervised linear metric (Fisher LDA / PCA+LDA) on UAD features.

Fits on TRAIN identities, applies to query/gallery, evaluates the official
WHU-MARS-1000 protocol. Diagnostic only; no model training.
"""
import json
import os
import time
import numpy as np

import offline_eval as oe
import offline_transforms as ot

OUT = os.path.dirname(os.path.abspath(__file__))


def fit_lda(X, y, shrink=0.1, d=256, eps=1e-6):
    C = int(y.max()) + 1
    mu = X.mean(0)
    Xc = X - mu
    Sw = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    means = np.zeros((C, X.shape[1]), dtype=np.float64)
    counts = np.zeros(C, dtype=np.float64)
    for c in range(C):
        sel = y == c
        n = int(sel.sum())
        if n == 0:
            continue
        mc = X[sel].mean(0)
        means[c] = mc
        counts[c] = n
        Xcc = X[sel] - mc
        Sw += Xcc.T @ Xcc
    Sw /= max(len(X) - C, 1)
    Sb = np.zeros_like(Sw)
    for c in range(C):
        if counts[c] == 0:
            continue
        diff = means[c] - mu
        Sb += counts[c] * np.outer(diff, diff)
    Sb /= max(len(X), 1)
    tr = np.trace(Sw) / Sw.shape[0]
    Sw = (1 - shrink) * Sw + shrink * np.eye(Sw.shape[0]) * tr
    # Solve generalized eigenproblem Sb v = lambda Sw v
    A = np.linalg.solve(Sw + eps * np.eye(Sw.shape[0]), Sb)
    w, V = np.linalg.eig(A)
    w = np.real(w)
    V = np.real(V)
    idx = np.argsort(w)[::-1][:d]
    W = V[:, idx].astype(np.float32)
    # normalize columns
    W /= np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)
    return mu.astype(np.float32), W


def run(shrink=0.1, d=256, center_mod=False):
    ft, pt, ct, mt, qf, qp, qc, qm, gf, gp, gc, gm = ot.load_all()
    if center_mod:
        mu_m = {m: ft[mt == m].mean(0) for m in (1, 2, 3)}
        ft = ft.copy(); qf = qf.copy(); gf = gf.copy()
        for m in (1, 2, 3):
            ft[mt == m] -= mu_m[m]
            qf[qm == m] -= mu_m[m]
            gf[gm == m] -= mu_m[m]
    mu, W = fit_lda(ft, pt, shrink=shrink, d=d)
    Q = (qf - mu) @ W
    G = (gf - mu) @ W
    Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    G = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-12)
    t0 = time.time()
    overall = ot.evaluate_overall(Q, G, qp, qc, gp, gc)
    pairs = ot.evaluate_pairs(Q, G, qp, qc, qm, gp, gc, gm)
    ag = ot.evaluate_ag(Q, G, qp, qc, gp, gc)
    return dict(shrink=shrink, d=d, center_mod=center_mod,
                overall=overall, pairs=pairs, ag=ag, seconds=time.time() - t0)


if __name__ == "__main__":
    out = []
    for shrink, d in ((0.1, 256), (0.3, 256), (0.1, 128)):
        print("=====", shrink, d, "=====", flush=True)
        r = run(shrink=shrink, d=d)
        out.append(r)
        print(json.dumps(r, indent=1), flush=True)
    with open(os.path.join(OUT, "offline_lda_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
