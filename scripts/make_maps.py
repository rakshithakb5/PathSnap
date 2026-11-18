"""Regenerate the Folium maps (GPS + road network + reference route + predicted route) WITHOUT retraining.
Needs artifacts/final/ (from Phase 5).   Run:  python scripts/make_maps.py            (first student trip)
                                              python scripts/make_maps.py 00000056 00000077
Writes outputs/maps/map_<trip>_classifier_only.html and map_<trip>_classifier_plus_hmm.html
(Reference route is shown ONLY because these are student-test/dev trips; never do this in the API/UI.)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.config import DATA_DIR, OUT_DIR
from pathsnap.io import load_trip, list_trips
from pathsnap.pipeline import PathSnapPipeline
from pathsnap.viz import make_map

out = OUT_DIR / "maps"; out.mkdir(parents=True, exist_ok=True)
ids = sys.argv[1:] or [list_trips(DATA_DIR / "student_test")[0].name]
pipe = PathSnapPipeline.load()
for tid in ids:
    d = DATA_DIR / "student_test" / tid
    res = pipe.run(load_trip(d, with_route=False))            # prediction side: no labels
    ref = load_trip(d, with_route=True)                       # drawing side: reference layer
    make_map(ref, pred_arcs=res["clf_pred"], out_html=out / f"map_{tid}_classifier_only.html", title=f"Classifier only - trip {tid}")
    make_map(ref, pred_arcs=res["hmm_pred"], out_html=out / f"map_{tid}_classifier_plus_hmm.html", title=f"Classifier + HMM - trip {tid}")
    print("written:", out / f"map_{tid}_classifier_plus_hmm.html")
