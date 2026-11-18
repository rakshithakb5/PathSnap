"""Shared helpers for Phase 3/4/5 scripts: load the Phase-2 tables, subsample, quick metrics."""
import numpy as np, pandas as pd
from .config import OUT_DIR, SEED
from .candidates import FEATURES
from . import metrics as M

DATA = OUT_DIR / "data"


def load_split(part, stride=1):
    """Candidate rows + point table for TRUSTED points only (label within 50 m of the GPS point)."""
    cand = pd.read_parquet(DATA / f"{part}_candidates.parquet")
    pts = pd.read_parquet(DATA / f"{part}_points.parquet")
    pts = pts[~pts.label_far]
    if stride > 1:
        pts = pts[pts.pt % stride == 0]
    cand = cand.merge(pts[["trip", "pt"]], on=["trip", "pt"], how="inner").reset_index(drop=True)
    cand["arc"] = cand["arc"].astype(np.int64)
    return cand, pts.reset_index(drop=True)


def XY(cand):
    return cand[FEATURES].to_numpy(np.float32), cand["label"].to_numpy()


def quick_scores(cand, pts, p):
    """Top-1, macro-F1, log-loss for probabilities p aligned with cand rows."""
    cm = M.candidate_metrics(cand["label"].to_numpy(), p)
    top1, _ = M.observation_top1(pts, M.top1_predictions(cand, p))
    return dict(top1=top1, macro_f1=cm["macro_f1"], log_loss=cm["log_loss"])
