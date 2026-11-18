"""PHASE 3 - Train & compare >=5 models (group-aware CV on the 80 train trips, then student_test).

Run (all models):        python scripts/phase3_train_compare.py
Run a subset / faster:   python scripts/phase3_train_compare.py --models logreg rf lgbm --folds 3 --stride 6

Protocol (matches Models and Training Guidelines.pdf section 4):
  * fit + cross-validate ONLY on the 80 training trips; CV is GroupKFold with group = trip id, so no trip
    is ever split between fit and validation folds (a random row split would leak - neighbouring 1 Hz
    points of one trip are almost identical).
  * the 10 student_test trips are used for development-stage comparison only.
  * the evaluator partition is never touched.
  * untrusted-label points (reference route shorter than the track) are excluded from fitting and scoring.
  * every model is fitted with the SAME features, SAME folds, SAME seed.
"""
import sys, time, json, argparse, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from pathsnap.config import OUT_DIR, ARTIFACT_DIR, DATA_DIR, SEED, TRAIN_STRIDE, CV_FOLDS
from pathsnap.candidates import FEATURES
from pathsnap.models import REGISTRY
from pathsnap.io import load_arcs_by_trip
from pathsnap import metrics as M

DATA = OUT_DIR / "data"; EXP = OUT_DIR / "experiments"; EXP.mkdir(parents=True, exist_ok=True)
(ARTIFACT_DIR / "models").mkdir(parents=True, exist_ok=True)


def load(part, stride):
    cand = pd.read_parquet(DATA / f"{part}_candidates.parquet")
    pts = pd.read_parquet(DATA / f"{part}_points.parquet")
    pts = pts[~pts.label_far]                                   # trusted points only
    if stride > 1: pts = pts[pts.pt % stride == 0]
    cand = cand.merge(pts[["trip", "pt"]], on=["trip", "pt"], how="inner").reset_index(drop=True)
    cand["arc"] = cand["arc"].astype(np.int64)
    return cand, pts.reset_index(drop=True)


def cap_rows(cand, max_rows, seed=SEED):
    """Subsample whole POINTS (keeping each point's full candidate set) to roughly max_rows rows."""
    if max_rows is None or len(cand) <= max_rows: return cand
    key = cand["trip"].astype(np.int64) * 10_000_000 + cand["pt"]
    uk = key.unique(); rng = np.random.default_rng(seed)
    chosen = rng.choice(uk, int(len(uk) * max_rows / len(cand)), replace=False)
    return cand[key.isin(chosen)]


def evaluate(model, cand, pts, arcs_by_trip):
    t0 = time.perf_counter()
    p = model.predict_proba(cand[FEATURES].to_numpy(np.float32))[:, 1]
    inf = time.perf_counter() - t0
    cm = M.candidate_metrics(cand["label"].to_numpy(), p)
    pred = M.top1_predictions(cand, p)
    top1, merged = M.observation_top1(pts, pred)
    cm.update(top1=top1, route_continuity=M.route_continuity(pred, arcs_by_trip),
              infer_ms_per_1000_pts=inf / len(pts) * 1e6,        # seconds/point -> milliseconds per 1000 points
              n_points=len(pts), n_rows=len(cand))
    return cm, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(REGISTRY))
    ap.add_argument("--folds", type=int, default=CV_FOLDS)
    ap.add_argument("--stride", type=int, default=TRAIN_STRIDE)
    a = ap.parse_args()

    tr, tr_pts = load("train", a.stride)
    st, st_pts = load("student_test", 1)                        # student-test: ALL trusted points
    arcs_tr = load_arcs_by_trip(DATA_DIR / "train"); arcs_st = load_arcs_by_trip(DATA_DIR / "student_test")
    print(f"train rows={len(tr):,} pts={len(tr_pts):,} | student rows={len(st):,} pts={len(st_pts):,} | stride={a.stride}")

    # ceiling for route continuity at this sampling rate = continuity of the TRUE label sequence
    ceil = M.route_continuity(st_pts.rename(columns={"true_arc": "pred_arc"}), arcs_st)
    print(f"route-continuity of the reference label sequence itself (ceiling) = {ceil:.4f}")

    folds = list(GroupKFold(n_splits=a.folds).split(tr, groups=tr["trip"]))
    rows, fold_rows, curves, cms, log = [], [], {}, {}, {"seed": SEED, "stride": a.stride, "folds": a.folds, "features": FEATURES}
    for name in a.models:
        spec = REGISTRY[name]; print(f"\n=== {name}: {spec['config']}")
        # ---- group-CV ----
        cvm = []
        for k, (i_tr, i_va) in enumerate(folds):
            ctr = cap_rows(tr.iloc[i_tr], spec["max_rows"]); cva = tr.iloc[i_va]
            pva = tr_pts.merge(cva[["trip", "pt"]].drop_duplicates(), on=["trip", "pt"])
            m = spec["build"](); t0 = time.perf_counter()
            m.fit(ctr[FEATURES].to_numpy(np.float32), ctr["label"].to_numpy())
            fit_s = time.perf_counter() - t0
            r, _ = evaluate(m, cva, pva, arcs_tr); r.update(model=name, fold=k, fit_s=fit_s)
            cvm.append(r); fold_rows.append(r)
            print(f"  fold {k}: macroF1={r['macro_f1']:.4f} top1={r['top1']:.4f} logloss={r['log_loss']:.4f} fit={fit_s:.0f}s")
        cv = pd.DataFrame(cvm)
        # ---- final fit on all 80 trips -> student_test ----
        ctr = cap_rows(tr, spec["max_rows"]); m = spec["build"](); t0 = time.perf_counter()
        m.fit(ctr[FEATURES].to_numpy(np.float32), ctr["label"].to_numpy()); fit_s = time.perf_counter() - t0
        rs, ps = evaluate(m, st, st_pts, arcs_st)
        # overfitting evidence: score on (a sample of) the data the model was fitted on
        smp = cap_rows(ctr, 60_000, seed=1)
        rtr, _ = evaluate(m, smp, tr_pts.merge(smp[["trip", "pt"]].drop_duplicates(), on=["trip", "pt"]), arcs_tr)
        rows.append(dict(model=name, family=spec["family"], config=spec["config"],
                         cv_macro_f1_mean=cv.macro_f1.mean(), cv_macro_f1_std=cv.macro_f1.std(),
                         cv_top1_mean=cv.top1.mean(), cv_top1_std=cv.top1.std(), cv_logloss_mean=cv.log_loss.mean(),
                         cv_route_continuity=cv.route_continuity.mean(),
                         st_top1=rs["top1"], st_macro_f1=rs["macro_f1"], st_precision=rs["precision"], st_recall=rs["recall"],
                         st_f1_pos=rs["f1_pos"], st_log_loss=rs["log_loss"], st_route_continuity=rs["route_continuity"],
                         infer_ms_per_1000_pts=rs["infer_ms_per_1000_pts"], fit_s=fit_s,
                         train_macro_f1=rtr["macro_f1"], train_top1=rtr["top1"], train_log_loss=rtr["log_loss"]))
        cms[name] = M.confusion(st["label"].to_numpy(), ps)
        with open(ARTIFACT_DIR / "models" / f"{name}.pkl", "wb") as f: pickle.dump(m, f)
        est = m[-1] if hasattr(m, "steps") else m
        if hasattr(est, "loss_curve_"):
            curves[name] = dict(train=list(est.loss_curve_), val=[1 - s for s in est.validation_scores_] if getattr(est, "validation_scores_", None) else None)
        if hasattr(est, "feature_importances_"):
            pd.Series(est.feature_importances_, index=FEATURES).sort_values(ascending=False).to_csv(EXP / f"feature_importance_{name}.csv")
        print(f"  CV top1={cv.top1.mean():.4f}±{cv.top1.std():.4f} | STUDENT top1={rs['top1']:.4f} macroF1={rs['macro_f1']:.4f} "
              f"logloss={rs['log_loss']:.4f} cont={rs['route_continuity']:.3f} | train top1={rtr['top1']:.4f} | "
              f"infer={rs['infer_ms_per_1000_pts']:.0f} ms/1k pts fit={fit_s:.0f}s")
        # LightGBM epoch history (held-out fold 0) - only models with a real training history may show a curve
        if name == "lgbm":
            i_tr, i_va = folds[0]; ctr0 = tr.iloc[i_tr]; cva0 = tr.iloc[i_va]
            mm = spec["build"](); mm.fit(ctr0[FEATURES].to_numpy(np.float32), ctr0["label"].to_numpy(),
                eval_set=[(ctr0[FEATURES].to_numpy(np.float32), ctr0["label"].to_numpy()), (cva0[FEATURES].to_numpy(np.float32), cva0["label"].to_numpy())],
                eval_metric="binary_logloss")
            ev = mm.evals_result_; ks = list(ev.keys())      # [train-set key, validation-set key]
            curves["lgbm"] = dict(train=ev[ks[0]]["binary_logloss"], val=ev[ks[-1]]["binary_logloss"])

    res = pd.DataFrame(rows); res.to_csv(EXP / "model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(EXP / "cv_fold_results.csv", index=False)
    json.dump({**log, "route_continuity_ceiling_student": ceil}, open(EXP / "experiment_log.json", "w"), indent=1)
    plots(res, cms, curves)
    show = res[["model", "cv_macro_f1_mean", "cv_top1_mean", "cv_logloss_mean", "st_top1", "st_macro_f1", "st_log_loss",
                "st_route_continuity", "infer_ms_per_1000_pts", "fit_s", "train_top1"]].round(4)
    print("\n" + show.to_string(index=False)); print("\nSaved to outputs/experiments/")


def plots(res, cms, curves):
    names = res.model.tolist(); x = np.arange(len(names))
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    ax[0].bar(x, res.cv_macro_f1_mean, yerr=res.cv_macro_f1_std, capsize=3); ax[0].set_title("CV Macro-F1 (group k-fold, train)")
    w = 0.38
    ax[1].bar(x - w / 2, res.st_top1, w, label="Top-1 acc"); ax[1].bar(x + w / 2, res.st_macro_f1, w, label="Macro-F1"); ax[1].legend()
    ax[1].set_title("Student-test Top-1 & Macro-F1")
    ax[2].bar(x, res.st_log_loss); ax[2].set_title("Student-test Log Loss (lower=better)")
    ax[3].scatter(res.infer_ms_per_1000_pts, res.st_top1)
    for n, xx, yy in zip(names, res.infer_ms_per_1000_pts, res.st_top1): ax[3].annotate(n, (xx, yy))
    ax[3].set_xscale("log"); ax[3].set_xlabel("inference ms / 1000 points"); ax[3].set_ylabel("Top-1"); ax[3].set_title("Accuracy vs speed")
    for a_ in ax[:3]: a_.set_xticks(x); a_.set_xticklabels(names, rotation=30)
    plt.tight_layout(); plt.savefig(EXP / "model_comparison.png", dpi=130); plt.close()
    fig, ax = plt.subplots(1, len(cms), figsize=(3.2 * len(cms), 3.2), squeeze=False)
    for a_, (n, cm) in zip(ax[0], cms.items()):
        a_.imshow(cm, cmap="Blues"); a_.set_title(n)
        for (i, j), v in np.ndenumerate(cm): a_.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8)
        a_.set_xticks([0, 1]); a_.set_yticks([0, 1]); a_.set_xlabel("pred"); a_.set_ylabel("true")
    plt.tight_layout(); plt.savefig(EXP / "confusion_matrices_student.png", dpi=130); plt.close()
    for n, c in curves.items():
        plt.figure(figsize=(5, 3.5)); plt.plot(c["train"], label="train")
        if c["val"] is not None: plt.plot(c["val"], label="validation")
        plt.xlabel("epoch / boosting round"); plt.ylabel("loss"); plt.legend(); plt.title(f"{n}: training history")
        plt.tight_layout(); plt.savefig(EXP / f"loss_curve_{n}.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
