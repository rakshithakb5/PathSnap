# PathSnap
GPS map-matching: noisy GPS track + road network -> road segment per point -> connected route (classifier + HMM/Viterbi).
Read `docs/GUIDE.md` first: it explains every phase and why each design choice was made.

## Setup
    pip install -r requirements.txt
    # put the hackathon data in ./data  (folders: train/, student_test/)  or set PATHSNAP_DATA=/path/to/Student_folder

## Run order (each phase writes to outputs/)
| Phase | Command | Output |
|---|---|---|
| 1 EDA | `python scripts/phase1_eda.py` | outputs/eda/ (inventory CSVs, plots, Folium map) |
| 2 Labels + candidates + features | `python scripts/phase2_build_dataset.py` | outputs/data/*.parquet, label/candidate quality |
| 3 Compare >=5 models | `python scripts/phase3_train_compare.py` | outputs/experiments/ (CSVs, plots, logs), artifacts/models/ |
| 4 Bayesian tuning | _to be added_ | |
| 5 HMM / Viterbi | _to be added_ | |
| 6 FastAPI + UI | _to be added_ | |

Seeds, configs and fold results are logged in outputs/experiments/. Evaluator data and labels are never read by any training script.
