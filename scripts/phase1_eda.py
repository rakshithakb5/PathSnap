"""PHASE 1 - Dataset inventory, data-quality checks, plots and a Folium map.

Run:  python scripts/phase1_eda.py
Writes: outputs/eda/inventory_<partition>.csv, plots (*.png), eda_map_<trip>.html

This answers the 'Initial Dataset Exploration' checklist from Dataset Exploration.pdf:
trip counts, points per trip, dtypes, missing/invalid GPS, timestamp order & gaps, extent,
network size, connectivity, isolated nodes, components, anomalies, route availability.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pathsnap.config import DATA_DIR, OUT_DIR
from pathsnap.io import load_trip, list_trips
from pathsnap.viz import make_map

EDA = OUT_DIR / "eda"; EDA.mkdir(parents=True, exist_ok=True)


def trip_row(trip, partition):
    tr, nd, ar, rt = trip.track, trip.nodes, trip.arcs, trip.route
    dt = np.diff(tr[:, 2])
    xy = trip.xy
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    speed = step / np.maximum(dt, 1e-6)
    N, A = len(nd), len(ar)
    g = coo_matrix((np.ones(A), (ar[:, 0], ar[:, 1])), shape=(N, N))
    ncomp, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    deg = np.bincount(ar.ravel(), minlength=N)
    row = dict(
        partition=partition, trip_id=trip.trip_id,
        n_points=len(tr), duration_s=tr[-1, 2] - tr[0, 2],
        nan_gps=int(np.isnan(tr).any(1).sum()),
        lonlat_invalid=int(((np.abs(tr[:, 0]) > 180) | (np.abs(tr[:, 1]) > 90)).sum()),
        ts_non_increasing=int((dt <= 0).sum()),
        dt_eq_1_frac=float(np.mean(dt == 1)), dt_max=float(dt.max()),
        n_gaps_gt10s=int((dt > 10).sum()),
        speed_p99_mps=float(np.percentile(speed[dt > 0], 99)), speed_max_mps=float(speed[dt > 0].max()),
        n_jumps_gt70mps=int((speed > 70).sum()),
        track_len_km=float(step.sum() / 1000),
        extent_km_x=float(np.ptp(xy[:, 0]) / 1000), extent_km_y=float(np.ptp(xy[:, 1]) / 1000),
        n_nodes=N, n_arcs=A,
        arcs_bad_node_ref=int(((ar < 0) | (ar >= N)).any(1).sum()),
        self_loops=int((ar[:, 0] == ar[:, 1]).sum()),
        isolated_nodes=int((deg == 0).sum()),
        n_components=int(ncomp), largest_component_frac=float(sizes.max() / N),
        route_len_arcs=-1, route_connected_frac=np.nan,
    )
    if rt is not None:
        row["route_len_arcs"] = len(rt)
        row["route_connected_frac"] = float(np.mean(ar[rt[:-1], 1] == ar[rt[1:], 0])) if len(rt) > 1 else 1.0
        row["route_arc_ids_valid"] = bool(((rt >= 0) & (rt < A)).all())
    return row


def main():
    allrows = []
    for part in ("train", "student_test"):
        rows = []
        for d in list_trips(DATA_DIR / part):
            rows.append(trip_row(load_trip(d), part))
        df = pd.DataFrame(rows)
        df.to_csv(EDA / f"inventory_{part}.csv", index=False)
        allrows.append(df)
        print(f"\n=== {part}: {len(df)} trips ===")
        print(df[["n_points", "duration_s", "dt_eq_1_frac", "dt_max", "n_gaps_gt10s", "n_jumps_gt70mps",
                  "n_nodes", "n_arcs", "n_components", "isolated_nodes", "route_len_arcs",
                  "route_connected_frac"]].describe().T[["min", "50%", "max"]].round(3).to_string())
        print("Missing GPS rows:", int(df.nan_gps.sum()), "| invalid lon/lat:", int(df.lonlat_invalid.sum()),
              "| non-increasing ts:", int(df.ts_non_increasing.sum()))
        bad = df[(df.dt_max > 60) | (df.n_jumps_gt70mps > 0) | (df.route_connected_frac < 1)]
        print("Anomalous trips (dt>60s, speed jumps, or disconnected route):")
        print(bad[["trip_id", "dt_max", "n_jumps_gt70mps", "route_connected_frac"]].to_string(index=False))
    df = pd.concat(allrows)

    # ---------- plots ----------
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    tr_df = df[df.partition == "train"]
    ax[0, 0].hist(tr_df.n_points, bins=25); ax[0, 0].set_title("GPS points per trip (train)")
    ax[0, 1].hist(np.log10(tr_df.n_arcs), bins=25); ax[0, 1].set_title("log10(#arcs) per trip (train)")
    ax[0, 2].hist(tr_df.route_len_arcs, bins=25); ax[0, 2].set_title("Reference route length, arcs (train)")
    ax[1, 0].hist(tr_df.speed_p99_mps, bins=25); ax[1, 0].set_title("99th pct implied speed m/s (train)")
    ax[1, 1].scatter(tr_df.n_points, tr_df.track_len_km, s=10); ax[1, 1].set_title("points vs track km (train)")
    ax[1, 1].set_xlabel("#points"); ax[1, 1].set_ylabel("km")
    ax[1, 2].bar(["train", "student_test"], [len(tr_df), (df.partition == "student_test").sum()])
    ax[1, 2].set_title("Trips per partition (evaluator: 10 hidden)")
    plt.tight_layout(); plt.savefig(EDA / "eda_overview.png", dpi=130); plt.close()

    # sampling interval histogram on one mid-size train trip + GPS-vs-route residual sketch
    d0 = list_trips(DATA_DIR / "train")[0]
    t0 = load_trip(d0)
    dt = np.diff(t0.track[:, 2])
    plt.figure(figsize=(6, 4)); plt.hist(np.clip(dt, 0, 10), bins=40)
    plt.title(f"Sampling interval, trip {t0.trip_id} (clipped at 10 s)"); plt.xlabel("seconds")
    plt.tight_layout(); plt.savefig(EDA / "sampling_interval.png", dpi=130); plt.close()

    # ---------- folium map: a mid-size train trip with its reference route ----------
    mid = tr_df.sort_values("n_points").iloc[len(tr_df) // 2].trip_id
    trip = load_trip(DATA_DIR / "train" / mid)
    make_map(trip, out_html=EDA / f"eda_map_train_{mid}.html", title=f"EDA: train trip {mid}")
    print(f"\nFolium map written: outputs/eda/eda_map_train_{mid}.html")
    summary = {"n_train": int((df.partition == 'train').sum()), "n_student": int((df.partition == 'student_test').sum())}
    json.dump(summary, open(EDA / "summary.json", "w"))


if __name__ == "__main__":
    main()
