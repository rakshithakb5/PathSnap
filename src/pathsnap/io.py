"""Raw-file loading + a Trip container.

File formats (verified against the real data, see docs/GUIDE.md):
  .track : lon lat t(seconds)           (no header, whitespace separated)
  .nodes : lon lat                      (row index = node id)
  .arcs  : from_node to_node            (row index = arc id, DIRECTED)
  .route : arc_id (one per line)        (ordered reference route for the whole trip)
"""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from .geo import LocalProjection


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (T,3) lon, lat, t
    nodes: np.ndarray            # (N,2) lon, lat
    arcs: np.ndarray             # (A,2) int
    route: np.ndarray | None     # (R,) int arc ids, None when unavailable (evaluator side)
    proj: LocalProjection = field(repr=False, default=None)
    xy: np.ndarray = field(repr=False, default=None)        # (T,2) projected metres
    node_xy: np.ndarray = field(repr=False, default=None)   # (N,2) projected metres
    t: np.ndarray = field(repr=False, default=None)         # (T,) seconds

    @property
    def arc_a(self): return self.node_xy[self.arcs[:, 0]]
    @property
    def arc_b(self): return self.node_xy[self.arcs[:, 1]]


def _load(path: Path, dtype=float, ncols=None):
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if ncols is not None and arr.shape[1] != ncols:
        raise ValueError(f"{path.name}: expected {ncols} columns, got {arr.shape[1]}")
    return arr


def load_trip(trip_dir: str | Path, with_route: bool = True) -> Trip:
    d = Path(trip_dir)
    tid = d.name
    track = _load(d / f"{tid}.track", float, 3)
    nodes = _load(d / f"{tid}.nodes", float, 2)
    arcs = _load(d / f"{tid}.arcs", int, 2)
    route = None
    rp = d / f"{tid}.route"
    if with_route and rp.exists():
        route = np.loadtxt(rp, dtype=int, ndmin=1)
    # one projection per trip, centred on the GPS track (every trip is in a different city)
    proj = LocalProjection.from_lonlat(track[:, 0], track[:, 1])
    trip = Trip(tid, track, nodes, arcs, route, proj)
    trip.xy = proj.forward(track[:, 0], track[:, 1])
    trip.node_xy = proj.forward(nodes[:, 0], nodes[:, 1])
    trip.t = track[:, 2]
    return trip


def list_trips(partition_dir: str | Path):
    p = Path(partition_dir)
    return sorted(d for d in p.iterdir() if d.is_dir())


def load_arcs_by_trip(partition_dir) -> dict:
    """{trip_id(int): arcs array} - used for route-continuity checks without loading everything else."""
    out = {}
    for d in list_trips(partition_dir):
        out[int(d.name)] = np.loadtxt(d / f"{d.name}.arcs", dtype=int, ndmin=2)
    return out
