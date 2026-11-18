"""Geometry helpers. All distances are in METRES in a per-trip local projection.

Why a local equirectangular projection (not Euclidean on degrees, not UTM)?
  * 1 degree of longitude is shorter than 1 degree of latitude by cos(lat), so raw-degree
    distances are wrong and direction-dependent (the dataset guide explicitly warns about this).
  * Every trip is in a different city and spans only a few km to a few tens of km. Over that
    scale the equirectangular error is well below GPS noise (<0.1%), and it needs no extra
    dependency (pyproj) and no UTM-zone logic (trips are on every continent).
"""
from dataclasses import dataclass
import numpy as np

R_EARTH = 6371008.8


@dataclass
class LocalProjection:
    lon0: float
    lat0: float

    @classmethod
    def from_lonlat(cls, lon, lat):
        return cls(float(np.median(lon)), float(np.median(lat)))

    def forward(self, lon, lat):
        lon = np.asarray(lon, float); lat = np.asarray(lat, float)
        x = np.deg2rad(lon - self.lon0) * R_EARTH * np.cos(np.deg2rad(self.lat0))
        y = np.deg2rad(lat - self.lat0) * R_EARTH
        return np.column_stack([x, y])

    def inverse(self, xy):
        xy = np.asarray(xy, float)
        lon = self.lon0 + np.rad2deg(xy[:, 0] / (R_EARTH * np.cos(np.deg2rad(self.lat0))))
        lat = self.lat0 + np.rad2deg(xy[:, 1] / R_EARTH)
        return np.column_stack([lon, lat])


def point_seg_dist(p, a, b):
    """Distance from point(s) p to segment(s) a-b. Broadcasts; returns (dist, frac_along, along_m).
    p:(...,2) a,b:(...,2)"""
    ab = b - a
    L2 = np.maximum((ab * ab).sum(-1), 1e-9)
    t = np.clip(((p - a) * ab).sum(-1) / L2, 0.0, 1.0)
    proj = a + t[..., None] * ab
    d = np.linalg.norm(p - proj, axis=-1)
    return d, t, t * np.sqrt(L2)


def bearing(a, b):
    """Direction of a->b as angle (radians, math convention atan2(dy,dx)) in the local plane."""
    d = b - a
    return np.arctan2(d[..., 1], d[..., 0])


def ang_diff(a, b):
    """Smallest signed difference a-b in (-pi, pi]."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi
