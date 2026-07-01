"""Elevation / terrain support for LuminaMaps.

Fetches a Digital Elevation Model (DEM) for the mapped area and turns it into a
solid, printable terrain base that the city is draped onto — so hilly cities
(San Francisco, Rome, Lisbon…) actually read as hilly.

Data source
-----------
Elevation is sampled on a regular grid via a public elevation API (batched HTTP,
no API key): OpenTopoData (SRTM 30 m) first, then Open-Elevation as a fallback.
Both are cached, so changing terrain *display* options never re-downloads.

Coordinate spaces
-----------------
* ``TerrainGrid`` stores elevation on a regular grid in **centred UTM metres**
  (easting/northing minus the map centre), matching how ``geometry_processor``
  centres its building geometry.
* ``TileTerrain`` is a per-tile resampling into **tile millimetre space** (the
  same space the extruded buildings live in), ready for the mesh stage to build
  a terrain solid and to look up the ground height under each building.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import streamlit as st
from pyproj import Transformer

USER_AGENT = "LuminaMaps/1.0 (3D map manufacturing tool)"
OPENTOPODATA_URL = "https://api.opentopodata.org/v1/srtm30m"
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
_BATCH = 100                 # locations per request (OpenTopoData public limit)
_REQUEST_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Global DEM grid (centred UTM metres)
# ---------------------------------------------------------------------------
@dataclass
class TerrainGrid:
    xs_m: np.ndarray         # 1D, increasing, centred easting (metres)
    ys_m: np.ndarray         # 1D, increasing, centred northing (metres)
    z_m: np.ndarray          # 2D [len(ys), len(xs)] elevation (metres)

    @property
    def elev_min(self) -> float:
        return float(np.nanmin(self.z_m))

    @property
    def elev_max(self) -> float:
        return float(np.nanmax(self.z_m))

    def sample(self, x_m: float, y_m: float) -> float:
        """Bilinearly sample elevation at a centred-metre coordinate (clamped)."""
        return _bilinear(self.xs_m, self.ys_m, self.z_m, x_m, y_m)


# ---------------------------------------------------------------------------
# Per-tile terrain in millimetre space
# ---------------------------------------------------------------------------
@dataclass
class TileTerrain:
    gx_mm: np.ndarray        # 1D increasing tile-x (mm)
    gy_mm: np.ndarray        # 1D increasing tile-y (mm)
    z_mm: np.ndarray         # 2D [len(gy), len(gx)] surface height (mm), >= base
    base_mm: float

    def sample(self, x_mm: float, y_mm: float) -> float:
        return _bilinear(self.gx_mm, self.gy_mm, self.z_mm, x_mm, y_mm)


def _bilinear(xs, ys, z, x, y) -> float:
    """Clamped bilinear interpolation on a regular grid."""
    x = min(max(x, xs[0]), xs[-1])
    y = min(max(y, ys[0]), ys[-1])
    i = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
    j = int(np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2))
    x0, x1 = xs[i], xs[i + 1]
    y0, y1 = ys[j], ys[j + 1]
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
    z00, z10 = z[j, i], z[j, i + 1]
    z01, z11 = z[j + 1, i], z[j + 1, i + 1]
    return float((z00 * (1 - tx) + z10 * tx) * (1 - ty)
                 + (z01 * (1 - tx) + z11 * tx) * ty)


# ---------------------------------------------------------------------------
# Elevation API backends
# ---------------------------------------------------------------------------
def _query_opentopodata(latlons: list[tuple[float, float]]) -> list[float] | None:
    import requests

    out: list[float] = []
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    for start in range(0, len(latlons), _BATCH):
        chunk = latlons[start:start + _BATCH]
        locs = "|".join(f"{lat:.6f},{lon:.6f}" for lat, lon in chunk)
        resp = sess.get(OPENTOPODATA_URL, params={"locations": locs},
                        timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        for r in resp.json().get("results", []):
            e = r.get("elevation")
            out.append(0.0 if e is None else float(e))
    return out


def _query_open_elevation(latlons: list[tuple[float, float]]) -> list[float] | None:
    import requests

    out: list[float] = []
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    for start in range(0, len(latlons), _BATCH):
        chunk = latlons[start:start + _BATCH]
        body = {"locations": [{"latitude": lat, "longitude": lon} for lat, lon in chunk]}
        resp = sess.post(OPEN_ELEVATION_URL, json=body, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        for r in resp.json().get("results", []):
            e = r.get("elevation")
            out.append(0.0 if e is None else float(e))
    return out


def _query_elevations(latlons: list[tuple[float, float]]) -> list[float] | None:
    """Try each elevation backend in turn; return None if all fail."""
    for backend in (_query_opentopodata, _query_open_elevation):
        try:
            values = backend(latlons)
            if values and len(values) == len(latlons):
                return values
        except Exception:  # noqa: BLE001 - try the next backend
            continue
    return None


# ---------------------------------------------------------------------------
# Fetch (cached) — the only networked entry point
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_terrain(center_latlon: tuple[float, float], radius_m: float,
                  utm_crs: str, grid_n: int = 56) -> TerrainGrid | None:
    """Fetch a DEM over the mapped square and return a ``TerrainGrid`` (or None).

    Cached on the geographic inputs, so terrain display options never re-fetch.
    """
    lat, lon = center_latlon
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_ll = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    cx, cy = to_utm.transform(lon, lat)

    xs = np.linspace(cx - radius_m, cx + radius_m, grid_n)
    ys = np.linspace(cy - radius_m, cy + radius_m, grid_n)

    latlons: list[tuple[float, float]] = []
    for y in ys:
        for x in xs:
            plon, plat = to_ll.transform(x, y)
            latlons.append((plat, plon))

    elevs = _query_elevations(latlons)
    if elevs is None:
        return None

    z = np.asarray(elevs, dtype=float).reshape(grid_n, grid_n)
    return TerrainGrid(xs_m=xs - cx, ys_m=ys - cy, z_m=z)


# ---------------------------------------------------------------------------
# Per-tile resampling into millimetre space
# ---------------------------------------------------------------------------
def build_tile_terrain(grid: TerrainGrid, *, tcx_m: float, tcy_m: float, scale: float,
                       tile_w_mm: float, tile_h_mm: float, base_mm: float,
                       elev_ref: float, exaggeration: float, cap_mm: float,
                       res: int = 48) -> TileTerrain:
    """Resample the global DEM onto a tile's mm grid, as surface heights.

    ``elev_ref`` (usually the whole map's minimum elevation) maps to exactly
    ``base_mm`` so the lowest ground still has a solid base; relief above it is
    scaled by the map scale and ``exaggeration`` and clamped to ``cap_mm``.
    Sampling at exact tile-box coordinates keeps neighbouring tiles aligned.
    """
    gx = np.linspace(-tile_w_mm / 2.0, tile_w_mm / 2.0, res)
    gy = np.linspace(-tile_h_mm / 2.0, tile_h_mm / 2.0, res)
    z = np.empty((res, res), dtype=float)
    for jj, Y in enumerate(gy):
        yc = tcy_m + Y / scale
        for ii, X in enumerate(gx):
            xc = tcx_m + X / scale
            relief = max(0.0, grid.sample(xc, yc) - elev_ref) * scale * exaggeration
            if cap_mm > 0:
                relief = min(relief, cap_mm)
            z[jj, ii] = base_mm + relief
    return TileTerrain(gx_mm=gx, gy_mm=gy, z_mm=z, base_mm=base_mm)
