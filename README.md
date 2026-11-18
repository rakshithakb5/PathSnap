# PathSnap

Snap noisy GPS tracks onto a road network and reconstruct the route that was driven.

PathSnap treats map matching as **candidate ranking**. For every GPS point it takes the nearest road segments, scores each with a LightGBM classifier, and then uses an **HMM with Viterbi decoding** to choose the most likely connected path. A FastAPI backend serves predictions and a single-page Leaflet UI shows and scores them.


## Results

Development set: 10 held-out trips (19,160 GPS points, trusted labels only).

| Stage | Top-1 accuracy | Route continuity |
| --- | --- | --- |
| LightGBM classifier alone | 0.900 | 0.794 |
| Classifier + HMM/Viterbi | 0.922 | 0.901 |

Route F1 rises from 0.905 to 0.929. The label ceiling for continuity is 0.917.

Seven models from six families were compared with 5-fold cross-validation grouped by trip: logistic regression, KNN, random forest, extra trees, HistGradientBoosting, LightGBM and an MLP. LightGBM ranked first on CV Top-1, macro-F1 and log loss.

Bayesian hyperparameter search (Optuna TPE, 12 trials) did not clearly beat the default configuration: it scored slightly lower on the development trips (Top-1 0.900 vs 0.904) and had worse log loss. The deployed model was refit with the tuned parameters. To switch back to the defaults, rename `artifacts/final/best_params.json` and rerun `scripts/phase5_hmm.py`.

## How it works

1. **Candidates.** Each trip is projected to local metres. For each point, the 10 nearest arcs within 100 m become candidates. The true arc is among them for over 99% of points.
2. **Features (27, label-free).** Distance and rank, position along the arc, heading agreement, speed and time gaps, road topology (degrees, twin arcs, dead ends), and the distance of neighbouring points to the same arc.
3. **Classifier.** A binary model predicts P(candidate is the arc the vehicle is on). Arc ids differ per trip, so ranking candidates is possible where multi-class prediction is not.
4. **HMM + Viterbi.** States are candidate arcs. Emissions are the classifier's log-probabilities. Transitions penalise extra hops, U-turns and unreachable jumps using the road graph. Viterbi returns the best path.
5. **Serving.** FastAPI runs the pipeline. The UI shows GPS points, the predicted route and the metrics.

Training labels come from the supplied `.route` files, aligned to GPS points with a monotone alignment HMM. Points more than 50 m from their aligned arc are treated as untrusted and excluded from fitting and scoring.

## Repository layout

```
api/              FastAPI app (/health, /trips, /predict)
ui/index.html     Single-file frontend (Leaflet / OpenStreetMap)
scripts/          phase1_eda ... phase5_hmm, make_maps
src/pathsnap/     io, geo, labeling, candidates, models, hmm, pipeline, evaluation
artifacts/final/  Trained model and decoder settings
docs/             GUIDE.md (design rationale), DEPLOY.md (two-machine setup)
outputs/          Plots, tables and maps from our runs
```

## Data

The dataset is **not included** in this repository (it is large). To run anything beyond the demo files, you need data in this layout:

```
data/
  train/
    00000001/00000001.track  .nodes  .arcs  .route
    ...
  student_test/
    00000000/00000000.track  .nodes  .arcs  .route
    ...
```

File formats (no headers):

| File | Columns |
| --- | --- |
| `.track` | longitude, latitude, timestamp (seconds) |
| `.nodes` | longitude, latitude (row index = node id) |
| `.arcs` | from_node, to_node (row index = arc id, directed) |
| `.route` | ordered arc ids driven (training and development data only) |

Either place the folders in `./data`, or point to them with an environment variable:

```bash
export PATHSNAP_DATA=/path/to/data          # Linux / macOS
set PATHSNAP_DATA=C:\path\to\data           # Windows CMD
```

For the API, `PATHSNAP_EVAL_DIR` selects the folder of trips to serve (default `data/student_test`). Trips served by the API need `.track`, `.nodes` and `.arcs`; the `.route` file is only used for scoring when present.

## Quick start

```bash
pip install -r requirements.txt
```

| Step | Command | Output |
| --- | --- | --- |
| EDA | `python scripts/phase1_eda.py` | `outputs/eda/` |
| Labels, candidates, features | `python scripts/phase2_build_dataset.py` | `outputs/data/` (regenerable, not committed) |
| Model comparison | `python scripts/phase3_train_compare.py` | `outputs/experiments/`, `artifacts/models/` |
| Bayesian tuning | `python scripts/phase4_bayes_tune.py` | `outputs/tuning/` |
| HMM decoder and final model | `python scripts/phase5_hmm.py` | `outputs/hmm/`, `artifacts/final/` |
| Folium maps | `python scripts/make_maps.py <trip_id>` | `outputs/maps/` |

Run the phases in order. The trained model in `artifacts/final/` is already included, so you can skip straight to the API if you only have trips to serve.

## Run the API and UI

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `ui/index.html` in a browser (it can be on another machine), enter the backend address, pick a trip and run the prediction. The reference route stays on the server and is only used to compute metrics. See `docs/DEPLOY.md` for the two-laptop setup and firewall notes.

## Limitations

- Some reference routes are shorter than their GPS tracks, so about 11% of training points have no trustworthy label.
- The decoder's best penalty setting sat at the edge of the tuning grid; a wider search may help.
- One development trip (trip 9) is clearly harder than the rest.
- Not tested on cities outside the supplied data.

Seeds, configurations and fold results are logged in `outputs/experiments/`.
