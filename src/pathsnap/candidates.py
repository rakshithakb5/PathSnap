"""Candidate generation + feature engineering  (the SAME code runs in training and in the API).

PROBLEM REFRAMING (the key design decision)
-------------------------------------------
Arc ids are meaningless across trips (each trip has its own city/network), so we cannot do
"classify the GPS point into one of N roads". Instead:

  1. For every GPS point, generate a small set of CANDIDATE arcs (the K nearest within R metres).
  2. Describe every (point, candidate) pair with geometric/contextual features.
  3. Train a binary classifier:  P(this candidate is the arc the vehicle is really on | features).
  4. Per point, the highest-probability candidate is the Top-1 prediction.
     The probabilities then become HMM emission scores (see hmm.py).

This turns one huge, trip-specific multi-class problem into one small, transferable binary problem.
All features are label-free, so inference on unseen evaluator trips needs no ground truth.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from .config import CAND_RADIUS_M, CAND_K, ARC_SAMPLE_STEP_M, HEADING_WINDOW, CONTEXT_OFFSETS
from .geo import point_seg_dist, bearing, ang_diff

FEATURES = [
    "d", "d_rank", "d_rel", "d_ratio", "n_cand",
    "frac_along", "along_m", "arc_len", "dist_to_end", "dist_to_start",
    "heading_cos", "heading_abs", "moving",
    "speed", "dt_prev", "dt_next",
    "out_deg_to", "in_deg_from", "has_twin", "is_dead_end",
    "ctx_m5", "ctx_m2", "ctx_p2", "ctx_p5", "ctx_mean", "ctx_max",
    "ctx_rel_mean",
]


def _clip_arcs_to_track(trip, buf):
    """Keep only arcs whose bounding box touches the track bbox (+buf). Huge networks shrink a lot."""
    lo = trip.xy.min(0) - buf; hi = trip.xy.max(0) + buf
    a, b = trip.arc_a, trip.arc_b
    mn = np.minimum(a, b); mx = np.maximum(a, b)
    keep = ((mx >= lo) & (mn <= hi)).all(1)
    return np.flatnonzero(keep)


def _densify(a, b, ids, step):
    """Sample points every `step` metres along the kept arcs -> (xy_samples, arc_id_of_sample)."""
    L = np.linalg.norm(b[ids] - a[ids], axis=1)
    n = np.ceil(L / step).astype(int) + 1
    arc_rep = np.repeat(ids, n)
    start = np.cumsum(n) - n
    k = np.arange(n.sum()) - np.repeat(start, n)
    frac = k / np.repeat(np.maximum(n - 1, 1), n)
    pts = a[arc_rep] + frac[:, None] * (b[arc_rep] - a[arc_rep])
    return pts, arc_rep


def _point_context(xy, t):
    """Per-point speed, heading from a +-W window. Heading is 'unknown' when the vehicle is ~still."""
    T = len(xy); w = HEADING_WINDOW
    i0 = np.clip(np.arange(T) - w, 0, T - 1); i1 = np.clip(np.arange(T) + w, 0, T - 1)
    disp = xy[i1] - xy[i0]
    span = np.maximum(t[i1] - t[i0], 1e-6)
    dist = np.linalg.norm(disp, axis=1)
    speed = dist / span
    head = np.arctan2(disp[:, 1], disp[:, 0])
    moving = dist > 3.0                                      # < 3 m over the window = standing/noise
    dt_prev = np.diff(t, prepend=t[0]); dt_next = np.diff(t, append=t[-1])
    return speed, head, moving, dt_prev, dt_next


def generate(trip, K=CAND_K, radius=CAND_RADIUS_M):
    """Returns (cand_pt, cand_arc, cand_d, cand_frac, cand_along) flat arrays sorted by point."""
    a_all, b_all = trip.arc_a, trip.arc_b
    keep = _clip_arcs_to_track(trip, radius + 50)
    samp, samp_arc = _densify(a_all, b_all, keep, ARC_SAMPLE_STEP_M)
    tree = cKDTree(samp)
    r_query = radius + ARC_SAMPLE_STEP_M / 2
    hits = tree.query_ball_point(trip.xy, r=r_query, return_sorted=False)
    fb_d, fb_i = tree.query(trip.xy, k=8)                     # fallback when nothing within radius

    P, A_, D, F, AL = [], [], [], [], []
    for i, h in enumerate(hits):
        arcs = np.unique(samp_arc[h]) if len(h) else np.unique(samp_arc[fb_i[i]])
        d, fr, al = point_seg_dist(trip.xy[i], a_all[arcs], b_all[arcs])
        if len(h):
            ok = d <= radius
            if ok.any():
                arcs, d, fr, al = arcs[ok], d[ok], fr[ok], al[ok]
        order = np.argsort(d)[:K]
        P.append(np.full(len(order), i)); A_.append(arcs[order]); D.append(d[order])
        F.append(fr[order]); AL.append(al[order])
    return (np.concatenate(P), np.concatenate(A_), np.concatenate(D), np.concatenate(F), np.concatenate(AL))


def build_trip_table(trip, labels=None, K=CAND_K):
    """Candidate x feature table for one trip.
    labels: optional dict from labeling.align_points_to_route (TRAIN/STUDENT only)."""
    pt, arc, d, frac, along = generate(trip, K=K)
    T = len(trip.xy)
    n = len(pt)
    A, B = trip.arc_a, trip.arc_b
    arc_len = np.linalg.norm(B[arc] - A[arc], axis=1)
    arc_head = bearing(A[arc], B[arc])

    speed, head, moving, dt_prev, dt_next = _point_context(trip.xy, trip.t)
    mv = moving[pt]
    dh = ang_diff(arc_head, head[pt])
    heading_cos = np.where(mv, np.cos(dh), 0.0)               # 0 = "no information" when standing still
    heading_abs = np.where(mv, np.abs(dh), np.pi / 2)

    # ---- per-point candidate-set statistics ----
    df = pd.DataFrame({"pt": pt, "arc": arc, "d": d})
    g = df.groupby("pt")["d"]
    d_min = g.transform("min").to_numpy()
    d_rank = g.rank(method="first").to_numpy() - 1
    n_cand = g.transform("size").to_numpy()

    # ---- network topology of each candidate arc ----
    N = len(trip.nodes)
    out_deg = np.bincount(trip.arcs[:, 0], minlength=N); in_deg = np.bincount(trip.arcs[:, 1], minlength=N)
    u, v = trip.arcs[arc, 0], trip.arcs[arc, 1]
    pair = set(map(tuple, trip.arcs.tolist()))
    # has_twin: the reverse-direction arc exists (two-way road)
    twin = np.fromiter(((int(vv), int(uu)) in pair for uu, vv in zip(u, v)), bool, count=n)

    # ---- temporal context: how far are the neighbouring GPS points from the SAME arc? ----
    ctx = {}
    for o in CONTEXT_OFFSETS:
        j = np.clip(pt + o, 0, T - 1)
        dd, _, _ = point_seg_dist(trip.xy[j], A[arc], B[arc])
        ctx[o] = dd
    ctx_arr = np.column_stack([ctx[o] for o in CONTEXT_OFFSETS])
    ctx_mean = ctx_arr.mean(1); ctx_max = ctx_arr.max(1)
    cm_pt = pd.Series(ctx_mean).groupby(pt).transform("min").to_numpy()

    out = pd.DataFrame({
        "pt": pt, "arc": arc,
        "d": d, "d_rank": d_rank, "d_rel": d - d_min, "d_ratio": d / (d_min + 1.0), "n_cand": n_cand,
        "frac_along": frac, "along_m": along, "arc_len": arc_len,
        "dist_to_end": arc_len - along, "dist_to_start": along,
        "heading_cos": heading_cos, "heading_abs": heading_abs, "moving": mv.astype(float),
        "speed": speed[pt], "dt_prev": dt_prev[pt], "dt_next": dt_next[pt],
        "out_deg_to": out_deg[v], "in_deg_from": in_deg[u],
        "has_twin": twin.astype(float),
        "is_dead_end": (out_deg[v] == 0).astype(float),
        "ctx_m5": ctx[-5], "ctx_m2": ctx[-2], "ctx_p2": ctx[2], "ctx_p5": ctx[5],
        "ctx_mean": ctx_mean, "ctx_max": ctx_max, "ctx_rel_mean": ctx_mean - cm_pt,
    })
    if labels is not None:
        out["label"] = (arc == labels["arc"][pt]).astype(np.int8)
    return out
