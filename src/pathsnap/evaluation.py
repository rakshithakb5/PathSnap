"""Server-side scoring of a predicted route against the reference route (needs ground truth => never used by the
prediction path). Used by Phase 5 (student-test) and by the API (evaluator trips, server side only)."""
import numpy as np
from .labeling import align_points_to_route


def continuity(seq, arcs):
    seq = np.asarray(seq); seq = seq[seq >= 0]
    if len(seq) == 0: return float("nan")
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]]
    if len(seq) < 2: return 1.0
    return float((arcs[seq[:-1], 1] == arcs[seq[1:], 0]).mean())


def route_set_f1(pred_seq, ref_arcs):
    """Sequence-level quality: precision/recall/F1 between the SET of predicted arcs and the reference route arcs."""
    p = set(int(x) for x in pred_seq if x >= 0); r = set(int(x) for x in ref_arcs)
    tp = len(p & r); prec = tp / max(len(p), 1); rec = tp / max(len(r), 1)
    return prec, rec, (2 * prec * rec / (prec + rec) if tp else 0.0)


def evaluate_prediction(trip_with_route, preds: dict):
    """preds: {name: (T,) predicted arc per GPS point}. Points whose reference label is >50 m away
    (reference route shorter than the track) are excluded, exactly as in training/CV."""
    lab = align_points_to_route(trip_with_route)
    ok = ~lab["far"]; true = lab["arc"]; arcs = trip_with_route.arcs
    out = {"n_points": int(len(true)), "n_scored_points": int(ok.sum())}
    for name, pr in preds.items():
        pr = np.asarray(pr)
        prec, rec, f1 = route_set_f1(pr[ok], trip_with_route.route)
        out[name] = dict(top1_accuracy=float((pr[ok] == true[ok]).mean()) if ok.any() else float("nan"),
                         route_continuity=continuity(pr[ok], arcs),
                         route_precision=prec, route_recall=rec, route_f1=f1)
    out["reference_label_continuity"] = continuity(true[ok], arcs)
    return out
