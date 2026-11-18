"""Model registry: >=5 models from DIFFERENT algorithm families (guideline requirement).

Why each one is in the line-up (use this reasoning in the report - and verify it with YOUR results):
  logreg   linear/probabilistic  -> baseline. Fast, well-calibrated probabilities. If it already does well,
                                   the features (not the model) carry the signal.
  knn      instance-based        -> "find similar past (point,arc) situations". Sensitive to scaling; slow at
                                   inference (stores all training rows) - shows the cost of lazy learners.
  rf       bagged trees          -> nonlinear interactions (distance x heading x context) with little tuning;
                                   robust to unscaled, heterogeneous features. Probabilities are averaged votes
                                   (often under-confident).
  et       extra-trees           -> like RF but more randomised splits: lower variance, faster to train.
  hgb      boosting (sklearn)    -> sequential error-correction; usually the strongest on tabular data.
  lgbm     boosting (LightGBM)   -> same family, faster, native early-stopping history for loss curves.
  mlp      neural net            -> learned nonlinear representation; needs scaling; gives an epoch-wise
                                   training/validation loss curve (the only models that may show loss plots).

NOTE ON CLASS IMBALANCE: ~11% of candidates are positives. We deliberately do NOT use class_weight='balanced':
the HMM needs honest probabilities, and re-weighting distorts them. Instead we evaluate with macro-F1,
Top-1 and log-loss, which expose imbalance problems.
"""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from .config import SEED

# max_rows: cap on training rows for models whose cost explodes with data (documented limitation)
REGISTRY = {
    "logreg": dict(family="Linear / probabilistic", max_rows=None,
        build=lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500, random_state=SEED)),
        config="StandardScaler + LogisticRegression(C=1.0, L2, lbfgs, max_iter=500)"),
    "knn": dict(family="Instance-based", max_rows=120_000,
        build=lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
        config="StandardScaler + KNN(k=25, distance-weighted, euclidean); train rows capped at 120k"),
    "rf": dict(family="Bagging / tree ensemble", max_rows=None,
        build=lambda: RandomForestClassifier(n_estimators=100, max_depth=18, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        config="RandomForest(n_estimators=100, max_depth=18, min_samples_leaf=5)"),
    "et": dict(family="Randomised tree ensemble", max_rows=None,
        build=lambda: ExtraTreesClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        config="ExtraTrees(n_estimators=100, max_depth=20, min_samples_leaf=5)"),
    "hgb": dict(family="Boosting (histogram GB)", max_rows=None,
        build=lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, max_depth=None, max_leaf_nodes=31, random_state=SEED),
        config="HistGradientBoosting(lr=0.1, max_iter=200, max_leaf_nodes=31)"),
    "lgbm": dict(family="Boosting (LightGBM)", max_rows=None,
        build=lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1),
        config="LightGBM(n_estimators=300, lr=0.05, num_leaves=31, subsample=0.8, colsample=0.8)"),
    "mlp": dict(family="Neural network (MLP)", max_rows=300_000,
        build=lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512,
                                    learning_rate_init=1e-3, max_iter=60, early_stopping=True, n_iter_no_change=6, random_state=SEED)),
        config="StandardScaler + MLP(64,32, relu, adam lr=1e-3, alpha=1e-4, early_stopping)"),
}
