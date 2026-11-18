"""All metrics required by Models and Training Guidelines.pdf section 5.

Two levels - keep them straight in the report:
  * CANDIDATE level (binary: is this candidate the true arc?) -> accuracy/precision/recall/F1/macro-F1/log-loss/confusion matrix
  * OBSERVATION level (one decision per GPS point)            -> Top-1 accuracy, candidate miss rate, route continuity
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                             log_loss, confusion_matrix)


def candidate_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    return dict(cand_accuracy=accuracy_score(y, yhat), precision=pr, recall=rc, f1_pos=f1,
                macro_f1=f1_score(y, yhat, average="macro", zero_division=0),
                log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))


def top1_predictions(df, p):
    """df has columns trip, pt, arc. Returns DataFrame(trip, pt, pred_arc) = argmax-prob candidate per point."""
    t = df[["trip", "pt", "arc"]].copy(); t["p"] = p
    idx = t.groupby(["trip", "pt"])["p"].idxmax()
    out = t.loc[idx, ["trip", "pt", "arc"]].rename(columns={"arc": "pred_arc"}).reset_index(drop=True)
    return out


def observation_top1(points, pred):
    """points: per-point table (trip, pt, true_arc, has_pos). pred: output of top1_predictions.
    A point whose true arc was NOT among the candidates counts as wrong (that is the candidate miss rate)."""
    m = points.merge(pred, on=["trip", "pt"], how="left")
    ok = (m.pred_arc == m.true_arc)
    return float(ok.mean()), m


def route_continuity(pred_pts, arcs_by_trip):
    """Share of arc-to-arc transitions in the predicted sequence that are physically connected
    (to_node(a) == from_node(b)). Consecutive repeats of the same arc are collapsed first.
    Calling this on the TRUE label sequence gives the ceiling achievable at 1 Hz sampling."""
    ok = tot = 0
    for trip, g in pred_pts.groupby("trip"):
        seq = g.sort_values("pt")["pred_arc"].to_numpy()
        keep = np.r_[True, seq[1:] != seq[:-1]]; seq = seq[keep]
        if len(seq) < 2: continue
        arcs = arcs_by_trip[trip]
        ok += int((arcs[seq[:-1], 1] == arcs[seq[1:], 0]).sum()); tot += len(seq) - 1
    return ok / tot if tot else float("nan")


def confusion(y, p, thr=0.5):
    return confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
