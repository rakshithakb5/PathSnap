"""PHASE 2 - Build the candidate x feature tables for train and student_test.

Run:  python scripts/phase2_build_dataset.py
Writes: outputs/data/{train,student_test}_candidates.parquet   (one row per point-candidate pair)
        outputs/data/{train,student_test}_points.parquet       (one row per GPS point: label + quality)
        outputs/data/label_quality.csv, candidate_recall.csv

Only the 80 training trips and 10 student-test trips are touched. The evaluator partition is never read.
Nothing is 'fitted' here (no scalers/encoders), so there is no leakage between partitions; feature scaling,
where a model needs it, is fitted inside the sklearn Pipeline on training folds only.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from pathsnap.config import DATA_DIR, OUT_DIR
from pathsnap.io import load_trip, list_trips
from pathsnap.labeling import align_points_to_route
from pathsnap.candidates import build_trip_table

OUT = OUT_DIR / "data"; OUT.mkdir(parents=True, exist_ok=True)


def process(trip_dir):
    trip = load_trip(trip_dir)
    lab = align_points_to_route(trip)
    tab = build_trip_table(trip, lab)
    tab.insert(0, "trip", int(trip.trip_id))
    haspos = tab.groupby("pt")["label"].max().reindex(range(len(trip.xy)), fill_value=0).to_numpy()
    pts = pd.DataFrame({"trip": int(trip.trip_id), "pt": np.arange(len(trip.xy)),
                        "true_arc": lab["arc"], "label_dist": lab["dist"], "label_far": lab["far"],
                        "has_pos": haspos.astype(bool)})
    return tab, pts


def main():
    for part in ("train", "student_test"):
        t0 = time.time()
        res = Parallel(n_jobs=-1)(delayed(process)(d) for d in list_trips(DATA_DIR / part))
        cand = pd.concat([r[0] for r in res], ignore_index=True)
        pts = pd.concat([r[1] for r in res], ignore_index=True)
        for c in cand.columns:
            if cand[c].dtype == np.float64: cand[c] = cand[c].astype(np.float32)
        cand.to_parquet(OUT / f"{part}_candidates.parquet"); pts.to_parquet(OUT / f"{part}_points.parquet")
        q = pts.groupby("trip").agg(n_points=("pt", "size"), label_far_frac=("label_far", "mean"),
                                    label_dist_med=("label_dist", "median"),
                                    candidate_recall=("has_pos", "mean"))
        q.to_csv(OUT / f"label_quality_{part}.csv")
        print(f"[{part}] trips={pts.trip.nunique()} points={len(pts):,} cand_rows={len(cand):,} "
              f"positive_rate={cand.label.mean():.3f} time={time.time()-t0:.0f}s")
        tr_ = pts[~pts.label_far]          # "trusted" points: labelled arc within 50 m of the GPS point
        print(f"   low-confidence labels (>50 m from labelled arc) = {pts.label_far.mean():.4f} "
              f"(excluded from training/scoring; see docs/GUIDE.md 'reference-route limitations')")
        print(f"   candidate recall on trusted points = {tr_.has_pos.mean():.4f} "
              f"-> candidate MISS RATE = {1-tr_.has_pos.mean():.4f}")
        bad = q[q.label_far_frac > 0.05]
        if len(bad): print("   trips with >5% untrusted labels (route shorter than track):", list(bad.index))


if __name__ == "__main__":
    main()
