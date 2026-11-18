"""PHASE 5 - HMM/Viterbi route reconstruction: tune the decoder, compare classifier-only vs classifier+HMM.

Run:   python scripts/phase5_hmm.py                 (~8-15 min)
Quick: python scripts/phase5_hmm.py --folds 3 --tune_trips 10

Steps
 1. Out-of-fold (OOF) classifier probabilities for every trusted train point (group k-fold by trip, so the decoder is
    tuned on probabilities the model did NOT fit on - tuning on in-sample probabilities would be over-optimistic).
 2. Grid-search decoder parameters (graph-hop cost alpha, unconnected-jump penalty) on a subset of TRAIN trips.
    Evaluator data is never used.
 3. Fit the final classifier on all 80 train trips, save artifacts/final/{model.pkl, hmm_config.json}.
 4. Run the REAL inference pipeline (same code as the API) on the 10 student-test trips and compare
    classifier-only vs classifier+HMM: Top-1, route continuity, route-level precision/recall/F1.
Outputs: outputs/hmm/{decoder_grid.csv, comparison.csv, per_trip.csv, map_*.html}
"""
import sys, json, time, pickle, argparse, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from pathsnap.config import OUT_DIR, ARTIFACT_DIR, DATA_DIR, SEED
from pathsnap.evalutil import load_split, XY
from pathsnap.models import REGISTRY
from pathsnap.hmm import HMMParams, RoadGraph, viterbi, group_candidates, argmax_arcs
from pathsnap.evaluation import continuity, evaluate_prediction
from pathsnap.io import load_trip, list_trips
from pathsnap.pipeline import PathSnapPipeline
from pathsnap.viz import make_map

HMM_OUT = OUT_DIR / "hmm"; HMM_OUT.mkdir(parents=True, exist_ok=True)
FINAL = ARTIFACT_DIR / "final"; FINAL.mkdir(parents=True, exist_ok=True)
FIXED = dict(subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1)


def lgbm_params():
    bp = FINAL / "best_params.json"
    if bp.exists():
        print("Using Bayesian-tuned LightGBM params from Phase 4"); return json.load(open(bp))
    cfg = REGISTRY["lgbm"]["build"]().get_params()
    print("Phase-4 params not found -> using Phase-3 LightGBM defaults")
    return {k: cfg[k] for k in ["n_estimators", "learning_rate", "num_leaves", "subsample", "colsample_bytree"]}


def read_arcs_t(part, trip):
    d = DATA_DIR / part / f"{trip:08d}"
    return np.loadtxt(d / f"{trip:08d}.arcs", dtype=int, ndmin=2), np.loadtxt(d / f"{trip:08d}.track", ndmin=2)[:, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--tune_trips", type=int, default=25)
    ap.add_argument("--max_pts", type=int, default=4000, help="only tune the decoder on trips with <= this many points")
    a = ap.parse_args()
    params = lgbm_params()

    # ---------- 1. OOF probabilities on train (fit on stride rows, predict ALL trusted points) ----------
    fit_all, _ = load_split("train", a.stride); full, full_pts = load_split("train", 1)
    trips = np.array(sorted(full["trip"].unique()))
    gkf = GroupKFold(n_splits=a.folds); oof = np.zeros(len(full))
    t0 = time.time()
    for k, (i_tr, i_va) in enumerate(gkf.split(trips, groups=trips)):
        te = set(trips[i_va]); ftr = fit_all[~fit_all.trip.isin(te)]
        m = LGBMClassifier(**params, **FIXED).fit(*XY(ftr))
        msk = full.trip.isin(te).to_numpy(); oof[msk] = m.predict_proba(XY(full[msk])[0])[:, 1]
        print(f"  OOF fold {k} done ({time.time()-t0:.0f}s)")
    full = full.assign(p=oof)

    # ---------- 2. tune decoder on a subset of train trips ----------
    sizes = full_pts.groupby("trip").size()
    pool = sizes[sizes <= a.max_pts].index.to_numpy()
    rng = np.random.default_rng(SEED); chosen = rng.choice(pool, min(a.tune_trips, len(pool)), replace=False)
    data = {}
    for tid in chosen:
        g = full[full.trip == tid]; pp = full_pts[full_pts.trip == tid].set_index("pt")
        arcs, t = read_arcs_t("train", int(tid)); T = len(t)
        data[tid] = dict(g=g, truth=pp["true_arc"], arcs=arcs, t=t, T=T, graph=RoadGraph(arcs))
    def score(decode):
        ok = n = 0; cont = []
        for tid, d in data.items():
            pred = decode(d); idx = d["truth"].index.to_numpy()
            ok += int((pred[idx] == d["truth"].to_numpy()).sum()); n += len(idx); cont.append(continuity(pred[idx], d["arcs"]))
        return ok / n, float(np.nanmean(cont))
    def clf_decode(d): return argmax_arcs(d["g"].pt.to_numpy(), d["g"].arc.to_numpy(), d["g"].p.to_numpy(), d["T"])
    base_top1, base_cont = score(clf_decode)
    print(f"\nDecoder tuning on {len(data)} train trips | classifier-only: Top-1={base_top1:.4f} continuity={base_cont:.4f}")
    grid = []
    for alpha, jump in itertools.product([0.1, 0.3, 0.6, 1.0], [2.0, 5.0, 10.0, 20.0]):
        prm = HMMParams(alpha=alpha, jump_pen=jump, uturn_pen=max(jump, 4.0))
        def dec(d, prm=prm):
            al, lp = group_candidates(d["g"].pt.to_numpy(), d["g"].arc.to_numpy(), d["g"].p.to_numpy(), d["T"], prm.eps)
            return viterbi(al, lp, d["t"], d["graph"], prm)
        t1 = time.time(); top1, cont = score(dec)
        grid.append(dict(alpha=alpha, jump_pen=jump, uturn_pen=prm.uturn_pen, top1=top1, continuity=cont, secs=time.time() - t1))
        print(f"  alpha={alpha} jump={jump:>4}: Top-1={top1:.4f} continuity={cont:.4f}")
    gdf = pd.DataFrame(grid); gdf.loc[len(gdf)] = dict(alpha=np.nan, jump_pen=np.nan, uturn_pen=np.nan, top1=base_top1, continuity=base_cont, secs=0)
    gdf.to_csv(HMM_OUT / "decoder_grid.csv", index=False)
    best = gdf.iloc[:-1].sort_values(["top1", "continuity"], ascending=False).iloc[0]
    hp = HMMParams(alpha=float(best.alpha), jump_pen=float(best.jump_pen), uturn_pen=float(best.uturn_pen))
    print(f"Best decoder: {hp.to_dict()}")

    # ---------- 3. final classifier on all 80 trips ----------
    final = LGBMClassifier(**params, **FIXED).fit(*XY(fit_all))
    with open(FINAL / "model.pkl", "wb") as f: pickle.dump(final, f)
    json.dump(hp.to_dict(), open(FINAL / "hmm_config.json", "w"), indent=1)

    # ---------- 4. real pipeline on the 10 student-test trips ----------
    pipe = PathSnapPipeline.load(FINAL); rows = []
    for d in list_trips(DATA_DIR / "student_test"):
        trip = load_trip(d, with_route=False)                       # prediction side: NO ground truth
        res = pipe.run(trip)
        ref = load_trip(d, with_route=True)                         # evaluation side: ground truth
        ev = evaluate_prediction(ref, {"classifier_only": res["clf_pred"], "classifier_plus_hmm": res["hmm_pred"]})
        for name in ("classifier_only", "classifier_plus_hmm"):
            rows.append(dict(trip=d.name, method=name, n_scored=ev["n_scored_points"], **ev[name],
                             hmm_s=res["timings"]["hmm_s"], ref_label_continuity=ev["reference_label_continuity"]))
        print(f"  trip {d.name}: Top-1 clf={ev['classifier_only']['top1_accuracy']:.4f} -> hmm={ev['classifier_plus_hmm']['top1_accuracy']:.4f} | "
              f"continuity {ev['classifier_only']['route_continuity']:.3f} -> {ev['classifier_plus_hmm']['route_continuity']:.3f}")
        if d.name == sorted(p.name for p in list_trips(DATA_DIR / "student_test"))[0]:
            make_map(ref, pred_arcs=res["clf_pred"], out_html=HMM_OUT / f"map_{d.name}_classifier_only.html", title=f"Classifier only - trip {d.name}")
            make_map(ref, pred_arcs=res["hmm_pred"], out_html=HMM_OUT / f"map_{d.name}_classifier_plus_hmm.html", title=f"Classifier + HMM - trip {d.name}")
    pt = pd.DataFrame(rows); pt.to_csv(HMM_OUT / "per_trip.csv", index=False)
    w = pt.n_scored
    comp = pt.groupby("method").apply(lambda g: pd.Series(dict(
        top1_accuracy=np.average(g.top1_accuracy, weights=g.n_scored), route_continuity=g.route_continuity.mean(),
        route_precision=g.route_precision.mean(), route_recall=g.route_recall.mean(), route_f1=g.route_f1.mean(),
        ref_label_continuity_ceiling=g.ref_label_continuity.mean())), include_groups=False)
    comp.to_csv(HMM_OUT / "comparison.csv"); print("\nSTUDENT-TEST: classifier-only vs classifier+HMM\n" + comp.round(4).to_string())
    hurt = pt.pivot(index="trip", columns="method", values="top1_accuracy"); hurt["delta"] = hurt.classifier_plus_hmm - hurt.classifier_only
    print("\nPer-trip Top-1 change (HMM - classifier):\n" + hurt.round(4).to_string())


if __name__ == "__main__":
    main()
