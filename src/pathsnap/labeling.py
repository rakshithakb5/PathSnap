"""Turn the REFERENCE ROUTE (one ordered list of arcs per trip) into PER-GPS-POINT labels.

The raw data does NOT say which arc each GPS row belongs to - only which arcs the vehicle drove,
in order. So we must align points to the route. Doing "nearest route arc" independently per point is
fragile (parallel carriageways, U-turns, revisited arcs, junction corners). Instead we run a small
monotone alignment HMM *restricted to the reference route*:

    state      = index j into the route (0..R-1)
    emission   = Gaussian on distance(point, route[j]) with sigma = LABEL_SIGMA_M (clipped at 60 m)
    transition = stay on j, or move forward by s<=LABEL_MAX_SKIP arcs, penalty LABEL_SKIP_PENALTY*s
                 (after a GPS gap > LABEL_GAP_S the vehicle may have moved any distance forward)

Why monotone-forward-only: a vehicle follows the route in order, so labels can never "go back".
This is what makes the labels consistent where the route revisits a road or passes close to itself.

IMPORTANT (be honest in the report): these labels are *derived from a reference route that is itself
imperfect* (a few trips have small route discontinuities). We flag low-confidence labels
(distance to labelled arc > LABEL_FAR_M) and report their share. Labels are used for TRAIN/STUDENT
data only. They are never computed for evaluator trips at inference time.
"""
import numpy as np
from .config import LABEL_SIGMA_M, LABEL_MAX_SKIP, LABEL_SKIP_PENALTY, LABEL_GAP_S, LABEL_FAR_M
from .geo import point_seg_dist

NEG = -1e18


def _route_distance_matrix(xy, a, b, chunk=2048):
    """(T,R) float32 distance of every point to every route arc."""
    T, R = len(xy), len(a)
    out = np.empty((T, R), np.float32)
    for s in range(0, T, chunk):
        p = xy[s:s + chunk][:, None, :]
        d, _, _ = point_seg_dist(p, a[None], b[None])
        out[s:s + chunk] = d
    return out


def align_points_to_route(trip, sigma=LABEL_SIGMA_M, max_skip=LABEL_MAX_SKIP,
                          skip_pen=LABEL_SKIP_PENALTY, gap_s=LABEL_GAP_S):
    """Returns dict(route_idx (T,), arc (T,), dist (T,), far (T,) bool)."""
    route = trip.route
    R, T = len(route), len(trip.xy)
    a, b = trip.arc_a[route], trip.arc_b[route]
    D = _route_distance_matrix(trip.xy, a, b)
    logE = -0.5 * (np.minimum(D, 60.0) / sigma) ** 2           # (T,R)
    dt = np.diff(trip.t, prepend=trip.t[0])

    score = logE[0].astype(np.float64)
    bp = np.zeros((T, R), dtype=np.int32)                       # previous route index
    idx = np.arange(R)
    S = min(max_skip, R - 1)
    for t in range(1, T):
        if dt[t] > gap_s:                                       # long gap: any forward move, cost 0
            cm = np.maximum.accumulate(score)
            is_new = score >= cm
            best_idx = np.maximum.accumulate(np.where(is_new, idx, 0))
            new = cm
            bp[t] = best_idx
        else:
            new = score.copy(); arg = idx.copy()                # s = 0 (stay)
            for s in range(1, S + 1):
                cand = np.full(R, NEG); cand[s:] = score[:R - s] - skip_pen * s
                better = cand > new
                new = np.where(better, cand, new)
                arg = np.where(better, idx - s, arg)
            bp[t] = arg
        score = new + logE[t]
    # backtrack
    j = int(np.argmax(score)); path = np.empty(T, np.int32)
    for t in range(T - 1, -1, -1):
        path[t] = j; j = bp[t, j]
    dist = D[np.arange(T), path]
    return dict(route_idx=path, arc=route[path], dist=dist, far=dist > LABEL_FAR_M)
