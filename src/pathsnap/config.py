"""Central configuration. Every magic number lives here so it can be justified in the report
and changed in ONE place. Record any change you make (and why) in your experiment log."""
from pathlib import Path

SEED = 42

# ---- paths (override with env var PATHSNAP_DATA if your data lives elsewhere) ----
import os
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", ROOT / "data"))   # contains train/ and student_test/
OUT_DIR = ROOT / "outputs"
ARTIFACT_DIR = ROOT / "artifacts"

# ---- candidate generation ----
CAND_RADIUS_M = 100.0     # only arcs within 100 m of the GPS point are candidates
CAND_K = 10               # keep the K nearest of those (K=10 gave ~99.9% candidate recall in our check)
ARC_SAMPLE_STEP_M = 10.0  # arcs are densified every 10 m for the KD-tree radius query

# ---- label construction (aligning GPS points to the reference route) ----
LABEL_SIGMA_M = 15.0      # emission std-dev (m) for the alignment HMM
LABEL_MAX_SKIP = 12       # max route arcs a point may advance in ONE normal (<=GAP_S) step
LABEL_SKIP_PENALTY = 0.5  # log-penalty per skipped arc (prefers the smallest forward move)
LABEL_GAP_S = 10.0        # dt above this => vehicle may have moved any distance forward
LABEL_FAR_M = 50.0        # a point farther than this from its labelled arc is flagged "low-confidence label"

# ---- feature engineering ----
HEADING_WINDOW = 3        # heading/speed estimated from points t-3 and t+3
CONTEXT_OFFSETS = (-5, -2, 2, 5)   # distance of neighbouring points to the SAME candidate arc

# ---- training ----
TRAIN_STRIDE = 3          # use every 3rd GPS point of a train trip for fitting (1 Hz points are near-duplicates)
CV_FOLDS = 5              # GroupKFold, group = trip id
