"""PHASE 4 - Bayesian hyperparameter optimisation of the chosen model (LightGBM) with Optuna (TPE).

Run:   python scripts/phase4_bayes_tune.py                       (30 trials, ~15-25 min)
Quick: python scripts/phase4_bayes_tune.py --trials 8 --stride 12 --folds 3

Why Bayesian (TPE) instead of grid/random search?  Each trial = a full group-CV of the model (minutes).
TPE keeps a probabilistic model of which hyperparameter regions gave good scores and samples there more often,
so it needs far fewer trials than a grid.   Objective: MEAN GROUP-CV TOP-1 over the 80 training trips
(same primary metric as Phase 3).  Evaluator data is never used.

Outputs: outputs/tuning/{trials.csv, best_params.json, untuned_vs_tuned.csv, history.png, param_importance.csv}
         artifacts/final/{best_params.json, lgbm_tuned.pkl}   (used by Phase 5 and the API)
"""
import sys, json, time, argparse, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, optuna
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from pathsnap.config import OUT_DIR, ARTIFACT_DIR, SEED
from pathsnap.evalutil import load_split, XY, quick_scores
from pathsnap.models import REGISTRY

TUNE = OUT_DIR / "tuning"; TUNE.mkdir(parents=True, exist_ok=True)
FINAL = ARTIFACT_DIR / "final"; FINAL.mkdir(parents=True, exist_ok=True)
FIXED = dict(subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1)


def cv_score(params, cand, pts, folds):
    res = []
    for i_tr, i_va in folds:
        tr, va = cand.iloc[i_tr], cand.iloc[i_va]
        X, y = XY(tr); m = LGBMClassifier(**params, **FIXED).fit(X, y)
        Xv, _ = XY(va); p = m.predict_proba(Xv)[:, 1]
        pv = pts.merge(va[["trip", "pt"]].drop_duplicates(), on=["trip", "pt"])
        res.append(quick_scores(va, pv, p))
    return pd.DataFrame(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--stride", type=int, default=6, help="point stride used INSIDE the search (speed)")
    ap.add_argument("--final_stride", type=int, default=3, help="stride for the untuned-vs-tuned comparison + final fit")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    tr_s, pts_s = load_split("train", a.stride)
    folds_s = list(GroupKFold(n_splits=a.folds).split(tr_s, groups=tr_s["trip"]))

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        sc = cv_score(params, tr_s, pts_s, folds_s)
        trial.set_user_attr("cv_macro_f1", float(sc.macro_f1.mean()))
        trial.set_user_attr("cv_log_loss", float(sc.log_loss.mean()))
        return float(sc.top1.mean())                      # OBJECTIVE: mean group-CV Top-1 (maximise)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                study_name="lgbm_pathsnap", storage=f"sqlite:///{(TUNE / 'study.db').as_posix()}",
                                load_if_exists=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials, callbacks=[lambda s, t: print(f"trial {t.number:>3}  CV Top-1={t.value:.4f}  best={s.best_value:.4f}")])
    search_s = time.time() - t0
    df = study.trials_dataframe(); df.to_csv(TUNE / "trials.csv", index=False)
    best = study.best_params; json.dump(best, open(TUNE / "best_params.json", "w"), indent=1)
    json.dump(best, open(FINAL / "best_params.json", "w"), indent=1)
    try:
        imp = pd.Series(optuna.importance.get_param_importances(study)); imp.to_csv(TUNE / "param_importance.csv")
    except Exception as e:
        print("importance skipped:", e)
    print(f"\n{len(study.trials)} trials in {search_s/60:.1f} min. Best CV Top-1={study.best_value:.4f}\nBest params: {best}")

    # ---- history plot ----
    v = df["value"].to_numpy()
    plt.figure(figsize=(6, 3.8)); plt.scatter(range(len(v)), v, s=14, label="trial")
    plt.plot(np.maximum.accumulate(v), color="r", label="best so far"); plt.xlabel("trial"); plt.ylabel("CV Top-1")
    plt.title("Bayesian (TPE) optimisation history"); plt.legend(); plt.tight_layout(); plt.savefig(TUNE / "history.png", dpi=130); plt.close()

    # ---- untuned vs tuned: SAME folds, SAME data, SAME metrics ----
    tr, pts = load_split("train", a.final_stride); st, st_pts = load_split("student_test", 1)
    folds = list(GroupKFold(n_splits=a.folds).split(tr, groups=tr["trip"]))
    base_cfg = REGISTRY["lgbm"]["build"]().get_params()
    untuned = {k: base_cfg[k] for k in ["n_estimators", "learning_rate", "num_leaves", "subsample", "colsample_bytree"]}
    rows = []
    for name, params in (("untuned (Phase-3 defaults)", untuned), ("tuned (Bayesian best)", best)):
        cv = cv_score(params, tr, pts, folds)
        X, y = XY(tr); t0 = time.perf_counter(); m = LGBMClassifier(**params, **FIXED).fit(X, y); fit_s = time.perf_counter() - t0
        Xs, _ = XY(st); t0 = time.perf_counter(); p = m.predict_proba(Xs)[:, 1]; inf = time.perf_counter() - t0
        sc = quick_scores(st, st_pts, p)
        rows.append(dict(model=name, cv_macro_f1=cv.macro_f1.mean(), cv_top1=cv.top1.mean(), cv_log_loss=cv.log_loss.mean(),
                         student_top1=sc["top1"], student_macro_f1=sc["macro_f1"], student_log_loss=sc["log_loss"],
                         inference_ms_per_1000_pts=inf / len(st_pts) * 1e6, fit_s=fit_s))
        if name.startswith("tuned"):
            with open(FINAL / "lgbm_tuned.pkl", "wb") as f: pickle.dump(m, f)
    out = pd.DataFrame(rows); out.to_csv(TUNE / "untuned_vs_tuned.csv", index=False)
    print("\n" + out.round(4).to_string(index=False))
    print("\nSaved: outputs/tuning/ and artifacts/final/lgbm_tuned.pkl")


if __name__ == "__main__":
    main()
