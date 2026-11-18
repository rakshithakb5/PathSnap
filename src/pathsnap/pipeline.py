"""The ONE inference pipeline used by Phase 5 AND the API (no train/serve skew).

raw trip (track+nodes+arcs, NO route) -> candidates+features -> classifier probabilities -> [argmax | HMM/Viterbi]
The trip object given to run() must not contain ground truth; the API loads it with with_route=False."""
import json, pickle, time
import numpy as np
from .config import ARTIFACT_DIR
from .candidates import build_trip_table, FEATURES
from .hmm import HMMParams, RoadGraph, viterbi, group_candidates, argmax_arcs


class PathSnapPipeline:
    def __init__(self, model, hmm_params: HMMParams):
        self.model, self.hmm = model, hmm_params

    @classmethod
    def load(cls, art_dir=None):
        d = (art_dir or ARTIFACT_DIR / "final")
        mp = d / "model.pkl"
        if not mp.exists():
            mp = d / "lgbm_tuned.pkl"
        if not mp.exists():
            mp = ARTIFACT_DIR / "models" / "lgbm.pkl"          # fallback: Phase-3 untuned LightGBM
            print(f"[pipeline] WARNING: tuned model not found, falling back to {mp}")
        model = pickle.load(open(mp, "rb"))
        hp = HMMParams()
        hc = d / "hmm_config.json"
        if hc.exists(): hp = HMMParams(**json.load(open(hc)))
        else: print("[pipeline] WARNING: hmm_config.json missing, using default HMM params")
        print(f"[pipeline] model={mp.name} hmm={hp.to_dict()}")
        return cls(model, hp)

    def run(self, trip):
        T = len(trip.xy); tm = {}
        t0 = time.perf_counter(); tab = build_trip_table(trip); tm["features_s"] = time.perf_counter() - t0
        t0 = time.perf_counter(); p = self.model.predict_proba(tab[FEATURES].to_numpy(np.float32))[:, 1]; tm["model_s"] = time.perf_counter() - t0
        pt, arc = tab["pt"].to_numpy(), tab["arc"].to_numpy()
        clf = argmax_arcs(pt, arc, p, T)
        t0 = time.perf_counter()
        arcs_l, lp_l = group_candidates(pt, arc, p, T, self.hmm.eps)
        hmm = viterbi(arcs_l, lp_l, trip.t, RoadGraph(trip.arcs), self.hmm); tm["hmm_s"] = time.perf_counter() - t0
        return dict(table=tab, proba=p, clf_pred=clf, hmm_pred=hmm, timings=tm)
