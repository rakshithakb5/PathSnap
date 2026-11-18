# PathSnap – Team 10 phase-by-phase guide

Each phase = **what we do → why this way (and not the alternative) → what it outputs → what to say in the presentation / viva**.
Schedule hooks: **Phases 1–3 = 11:30 AM evaluation** (preprocessing + features 5 marks, model selection + training 5 marks).
**Phases 4–6 = 3:00 PM evaluation** (tuning 5, deployment 5, demo 5). Phase 7 = report + submission form.

> Integrity reminder: this code is a starting scaffold. The numbers in your report must come from **your own runs**
> (`outputs/experiments/*.csv`), and every team member must be able to explain every phase below. Disclose AI assistance as the
> event policy requires. Don't copy another team's configs – change settings you can justify and log why.

```
raw files ──► Phase 1 EDA ──► Phase 2 label + candidates + features ──► Phase 3 compare ≥5 models
                                                                              │
 API/UI ◄── Phase 6 ◄── Phase 5 HMM/Viterbi ◄── Phase 4 Bayesian tuning ◄────┘
```

---
## Phase 0 – Understand the problem (9:30–10:00)
**One sentence:** given a noisy 1 Hz GPS track and a road network, output the road segment (arc) the car was on at every point, and a connected route.

**The key insight that shapes everything:** every trip has its **own road network in a different city** (we checked: 90 different node files, trips on several continents). So arc id 17630 means nothing outside its trip, and "classify each GPS point into one of N roads" is impossible. We therefore reframe:

| Naive idea | Why it fails | What we do instead |
|---|---|---|
| Multi-class classifier over arc ids | ids are trip-specific, 10k–300k classes | **Candidate ranking**: for each point take the K nearest arcs, classify each (point, arc) pair as *right / wrong* |
| Nearest arc only | two-way roads give twin arcs at identical distance; parallel streets | classifier uses **heading, context, topology**, not just distance (nearest-arc baseline got ~46 % Top-1 on one trip) |
| Classifier only | picks can jump between unconnected roads | **HMM/Viterbi** (Phase 5) enforces a connected route |

---
## Phase 1 – Data understanding & EDA  (`scripts/phase1_eda.py`)
**Run:** `python scripts/phase1_eda.py` → `outputs/eda/` (inventory CSVs, plots, Folium map).

**What it checks (the checklist in *Dataset Exploration.pdf*):** trips per partition (80 / 10 / 10 hidden), points per trip, NaN/invalid lon-lat, timestamp order, sampling gaps, implied speeds, extent, #nodes/#arcs, connectivity (scipy connected components), isolated nodes, route availability and route connectivity.

**What we found (cite your own CSVs):**
- No NaNs, no invalid coordinates, timestamps strictly increasing → no cleaning/imputation needed. (Say this explicitly – "missing-value analysis".)
- ~97–100 % of steps are exactly 1 s, but **real gaps exist** (up to 153 064 s in one train trip, many >60 s) → features must use real `dt`, and the sequence model must handle gaps.
- **GPS jumps**: a few trips have dozens of implied speeds >70 m/s (e.g. one train trip has ~95) → outliers; robust features (distance-to-arc, windowed heading) instead of raw point-to-point speed.
- Networks are **not connected graphs** (5–100 components) → "no path" must be a valid state in routing.
- Reference routes are mostly 100 % connected, but a few trips have tiny breaks (0.98–0.997) → the route is *reference*, not perfect truth (dataset PDF §6 says exactly this).

**Why a local projection (`geo.py`):** degrees ≠ metres (longitude shrinks with cos(lat)). Each trip gets its own equirectangular projection around its median point: error ≪ GPS noise at city scale, no pyproj/UTM-zone logic needed for a world-wide dataset. Rejected: Euclidean on degrees (wrong), Haversine everywhere (slow, and we need point-to-*segment* geometry).

**Say in the presentation:** "We verified file semantics ourselves – columns, 0-based ids, directed arcs – and found the route has no per-point timestamps, which is the central difficulty of Phase 2."

---
## Phase 2 – Preprocessing, labelling, candidates, features (`scripts/phase2_build_dataset.py`, `src/pathsnap/{labeling,candidates}.py`)
**Run:** `python scripts/phase2_build_dataset.py` (~30 s) → `outputs/data/*.parquet`.

### 2a. Labels from the reference route (`labeling.py`)
**Problem:** `.route` is an ordered list of arcs for the whole trip, *not* one arc per GPS row. We must align them.
**Method:** a small **monotone alignment HMM restricted to the route**: state = position in the route; emission = Gaussian on distance to that arc (σ = 15 m); transition = stay or move forward ≤ 12 arcs (any distance after a >10 s gap). Solved with Viterbi/dynamic programming.
**Why not "nearest route arc per point":** fails on U-turns, parallel carriageways, revisited roads. The forward-only constraint is physically true (the car follows the route in order) and resolves these.
**Result you can quote:** labels are monotone, median distance to labelled arc ≈ 4–6 m on all trips tested.
**Honest limitation:** in 5 train trips (2, 33, 43, 54, 66) the route covers only part of the GPS track (e.g. trip 54: route 31 km, track 169 km) – we verified even order-free nearest-arc fails there, so it is the data, not our alignment. Those ~11 % of train points are flagged `label_far` (>50 m from labelled arc) and **excluded from fitting and scoring**; student-test has only 0.14 % such points. *Always state this – the PDF explicitly asks you not to overstate label certainty.*

### 2b. Candidate generation (`candidates.py::generate`)
Arcs are densified every 10 m, put in a KD-tree; for each GPS point we take arcs within 100 m, keep the **K = 10 nearest**. Arcs outside the track's bounding box are dropped first (some networks have 300 k arcs).
**Why K=10 / 100 m:** it is a recall-vs-size trade-off. Candidate recall (reference arc ∈ candidates) = **99.5 % train / 99.85 % student**, i.e. **candidate miss rate 0.5 % / 0.15 %**. K=5 lost ~1 %; K=20+ added rows but almost no recall.
**Why KD-tree on sampled points, not "nearest node":** a node can be far from the road you are on if the arc is long; point-to-segment distance is exact.

### 2c. Features (27, all label-free → identical at inference) 
| Group | Features | Why |
|---|---|---|
| Geometry | perpendicular distance `d`, rank, `d_rel`=d−min d, ratio, #candidates | the basic evidence; relative distance matters in dense areas |
| Along-arc | fraction/metres along, arc length, distance to ends | where on the arc – junction corners behave differently |
| **Heading** | cos/abs of angle between arc direction and GPS heading, `moving` flag | **separates twin (opposite-direction) arcs and parallel roads** – the single most valuable group |
| Motion | speed, dt prev/next | slow/gappy points are noisier |
| Topology | out-degree of end node, in-degree of start, has-twin, dead-end | intersections/two-way roads |
| **Temporal context** | distance of points t±2, t±5 to *the same arc*, their mean/max, mean relative to best candidate | a true arc stays close for neighbouring points; a crossing street doesn't. Gives the classifier a sliding window without a sequence model |

**Rejected features:** raw lon/lat (trip-specific, would memorise cities), arc id (meaningless), anything using `.route` (not available for evaluator trips = leakage).
**Leakage control:** no scaler/encoder is fitted here; scaling lives inside sklearn Pipelines fitted only on training folds. Splits are by **trip** everywhere.

**Say in the presentation:** "The same `build_trip_table()` function is used offline and inside the API, so there is no train/serve skew."

---
## Phase 3 – Model experimentation & selection (`scripts/phase3_train_compare.py`, `src/pathsnap/models.py`)
**Run:** `python scripts/phase3_train_compare.py` (add `--models logreg rf lgbm --folds 3 --stride 6` for a quick pass) → `outputs/experiments/`.

**Protocol:** train on the 80 trips only; **GroupKFold(5), group = trip** (a random row split leaks: adjacent 1 Hz points are near-duplicates); same features, folds and seed for every model; student_test used for development comparison; evaluator data never touched. Training uses every 3rd point (`TRAIN_STRIDE`) because 1 Hz neighbours are nearly identical – this cuts cost ~3× with negligible information loss (you can check: run `--stride 1` for one model).

**Models (7, six families – ≥5 required):** Logistic Regression (linear), KNN (instance), Random Forest (bagging), Extra Trees (randomised), HistGradientBoosting + LightGBM (boosting), MLP (neural). Reasons for each are in `models.py`.
**No `class_weight='balanced'`:** only ~11 % of candidates are positive, but the HMM needs honest probabilities; re-weighting distorts them. We watch macro-F1 and log-loss instead.

**Metrics:** candidate level – accuracy, precision, recall, F1, **macro-F1**, **log loss**, confusion matrix; observation level – **Top-1 accuracy** (argmax candidate vs reference arc; a missed candidate counts as wrong), candidate miss rate, **route continuity** (share of consecutive distinct predicted arcs that are physically connected; the true label sequence itself scores ≈0.90 at 1 Hz, so that is the ceiling), training time, inference ms/1000 points.

**Primary selection metric = CV Top-1 accuracy** (threshold-free, equals the task: "which road is the car on?"), with **log loss as tiebreaker/HMM-compatibility check** (the decoder consumes probabilities), macro-F1 reported as required. Rejected as primary: candidate accuracy (a model that says "never" gets 88 %), F1 at 0.5 (depends on an arbitrary threshold).
**Over/under-fitting evidence** = train vs CV vs student gap (columns `train_top1`, `cv_top1_*`, `st_top1`) and fold std.
**Loss curves** are produced only for models with real epoch history (MLP, LightGBM) – the PDF forbids fabricated curves.

**What our scaffold check showed (reduced run: 3 folds, stride 12 – re-run yours with the full settings; quote only your numbers):**
- Boosting (LightGBM / HistGB) led: ≈0.89–0.90 Top-1, log-loss ≈0.064 vs LogReg ≈0.87 Top-1, log-loss ≈0.10 → the problem has non-linear interactions (distance × heading × context).
- **KNN overfits badly**: train Top-1 ≈0.99 vs CV/student ≈0.865, and inference is ~70× slower than trees → a good "overfitting evidence + cost" example for the report.
- LogReg/MLP are fast but under-perform trees; RF/ET are close to boosting but a bit worse on log-loss (averaged votes are less sharp).
- Top-1 plateaus around 0.90 and route continuity is only ≈0.77 vs a 0.905 ceiling: classifier-only still jumps between unconnected arcs → this is exactly the gap **Phase 5 (HMM)** should close.
- Full-settings runtime: ≈40+ min on a 2-core machine (RF/ET/KNN dominate). Use `--models` to start with logreg, hgb, lgbm, then add the rest while you write.

---
## Phase 4 – Bayesian hyperparameter optimisation (3 PM; to build next)
Optuna TPE on the best model, objective = mean **group-CV Top-1 (or macro-F1)** on 80 trips, 30–50 trials, store `study.trials_dataframe()` as the "search record". Compare untuned vs tuned on CV, student Top-1, macro-F1, log loss, inference time. **Why Bayesian (TPE) not grid/random:** each trial costs minutes; TPE models which regions are promising and wastes fewer trials.

## Phase 5 – HMM + Viterbi route reconstruction (to build next)
Hidden state = candidate arc at time t; emission = classifier probability; transition = high for same/next-connected arc, small penalty for disconnected jumps, shortest-path bridging across big gaps. Report classifier-only vs classifier+HMM (Top-1, continuity). Decoder parameters tuned on train + student only.

## Phase 6 – FastAPI + UI + two-laptop deployment (to build next)
`POST /predict {trip_id}` → loads raw files → `build_trip_table` → model → Viterbi → returns route; accuracy computed **server-side** from reference data the UI never sees. Log every request on Laptop 1.

## Phase 7 – Report + submission
Map every report section to files in `outputs/`; commit code, CSVs, logs, Folium HTML; no secrets; README with run steps.

---
### Likely viva questions (practise answering from your own results)
1. Why can't we classify points into arc ids? 2. How did you build point labels from the route, and what are its limits? 3. Why GroupKFold by trip? 4. What is the candidate miss rate and what bounds your Top-1? 5. Why those features – which group mattered most (see `feature_importance_*.csv`)? 6. Why did model X beat model Y here? 7. Why is no class weighting used? 8. What would happen if you evaluated on a random row split?
