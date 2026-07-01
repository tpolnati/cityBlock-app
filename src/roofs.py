"""Procedural roof-shape reconstruction for LuminaMaps.

OSM "Simple 3D Buildings" tags a building's ``roof:shape`` (``dome``, ``pyramidal``,
``gabled``, ``onion``, ``spire``…), ``roof:height`` and ``roof:direction``. Flat
extrusion throws this away — but the roof is exactly what makes a cathedral,
mosque or clock-tower recognisable. This module turns those tags into a roof
solid that sits on top of the wall extrusion.

Each builder returns a self-contained, watertight ``trimesh.Trimesh`` in tile
millimetre space (or ``None`` if the shape can't be built), to be concatenated
onto the city mesh. Footprints may be arbitrary polygons; ridge/dome shapes use
the minimum rotated rectangle for a clean, recognisable result.
"""

from __future__ import annotations

import numpy as np
import trimesh
from shapely.geometry import Polygon

# Roof shapes we render; anything else falls back to flat (None).
SUPPORTED_SHAPES = {
    "pyramidal", "cone", "spire", "dome", "onion",
    "gabled", "hipped", "round", "skillion", "pitched",
}


def _exterior_xy(poly):
    """Return the exterior ring of a (Multi)Polygon's largest part as an (N,2) array."""
    if poly is None or poly.is_empty:
        return None
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.geom_type != "Polygon":
        return None
    coords = np.asarray(poly.exterior.coords)[:-1]   # drop closing dup
    return coords if len(coords) >= 3 else None


def _apex_roof(poly, z0: float, height: float, apex_xy=None) -> trimesh.Trimesh | None:
    """Pyramidal / conical / spire roof: footprint ring lifted to a single apex."""
    ring = _exterior_xy(poly)
    if ring is None:
        return None
    apex = np.asarray(apex_xy if apex_xy is not None else ring.mean(axis=0))
    n = len(ring)

    verts = [[x, y, z0] for x, y in ring]          # 0..n-1 base ring
    verts.append([apex[0], apex[1], z0 + height])  # n = apex
    verts = np.asarray(verts, dtype=float)

    faces = [[i, (i + 1) % n, n] for i in range(n)]  # sides -> apex
    verts = list(verts)
    # Bottom cap: triangulate the base polygon (merge_vertices later welds the
    # duplicated z0 ring vertices, keeping the solid watertight).
    try:
        v2d, f2d = trimesh.creation.triangulate_polygon(Polygon(ring), engine="earcut")
        off = len(verts)
        verts.extend([[float(x), float(y), z0] for x, y in v2d])
        faces.extend([[off + int(b), off + int(a), off + int(c)] for a, b, c in f2d])
    except Exception:  # noqa: BLE001 - fan cap fallback (fine for convex footprints)
        faces.extend([[0, (i + 1), i] for i in range(1, n - 1)])

    return _finish(np.asarray(verts, dtype=float), faces)


def _dome_roof(poly, z0: float, height: float, onion: bool = False) -> trimesh.Trimesh | None:
    """Hemispherical dome (optionally an onion with a pointed tip) over the bbox."""
    ring = _exterior_xy(poly)
    if ring is None:
        return None
    minx, miny = ring.min(axis=0)
    maxx, maxy = ring.max(axis=0)
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    rx, ry = max((maxx - minx) / 2.0, 1e-3), max((maxy - miny) / 2.0, 1e-3)

    rings, segs = 10, 24
    verts, faces = [], []
    # Latitude rings, excluding the exact top (which would be a degenerate ring).
    for i in range(rings):
        t = i / rings                        # 0 at base -> nearly 1 near top
        if onion:                            # bulge out then pinch to a point
            r = np.cos(t * np.pi / 2) * (1.0 + 0.35 * np.sin(t * np.pi))
            z = z0 + height * (t ** 0.7)
        else:
            r = np.cos(t * np.pi / 2)
            z = z0 + height * np.sin(t * np.pi / 2)
        for j in range(segs):
            a = 2 * np.pi * j / segs
            verts.append([cx + rx * r * np.cos(a), cy + ry * r * np.sin(a), z])
    apex = len(verts)
    verts.append([cx, cy, z0 + height])      # single apex vertex
    for i in range(rings - 1):               # quads between successive rings
        for j in range(segs):
            a = i * segs + j
            b = i * segs + (j + 1) % segs
            c = (i + 1) * segs + j
            d = (i + 1) * segs + (j + 1) % segs
            faces.extend([[a, b, d], [a, d, c]])
    last = (rings - 1) * segs                 # top ring -> apex
    for j in range(segs):
        faces.append([last + j, last + (j + 1) % segs, apex])
    base_c = len(verts)                       # flat base disk -> closed solid
    verts.append([cx, cy, z0])
    for j in range(segs):
        faces.append([base_c, (j + 1) % segs, j])
    return _finish(np.asarray(verts, dtype=float), faces)


def _ridge_roof(poly, z0: float, height: float, direction=None, hipped: bool = False):
    """Gabled / hipped roof over the footprint's minimum rotated rectangle."""
    try:
        rect = poly.minimum_rotated_rectangle
        corners = np.asarray(rect.exterior.coords)[:-1]
    except Exception:  # noqa: BLE001
        return None
    if len(corners) != 4:
        return None

    # Identify long vs short edges; the ridge runs along the long axis.
    e0 = np.linalg.norm(corners[1] - corners[0])
    e1 = np.linalg.norm(corners[2] - corners[1])
    if e0 >= e1:
        p0, p1, p2, p3 = corners[0], corners[1], corners[2], corners[3]
    else:
        p0, p1, p2, p3 = corners[1], corners[2], corners[3], corners[0]
    # Short-edge midpoints -> ridge ends.
    mA = (p0 + p3) / 2.0
    mB = (p1 + p2) / 2.0
    inset = 0.0
    if hipped:
        inset = min(0.25, 0.25)  # pull ridge in by 25% of length for hips
        mA, mB = mA + (mB - mA) * inset, mB + (mA - mB) * inset

    rA = [mA[0], mA[1], z0 + height]
    rB = [mB[0], mB[1], z0 + height]
    verts = [
        [p0[0], p0[1], z0], [p1[0], p1[1], z0],
        [p2[0], p2[1], z0], [p3[0], p3[1], z0],
        rA, rB,
    ]
    # 0..3 base corners, 4 = ridge@A side, 5 = ridge@B side
    faces = [
        [0, 1, 5], [0, 5, 4],   # long slope p0-p1
        [2, 3, 4], [2, 4, 5],   # long slope p2-p3
        [1, 2, 5],              # hip/gable near B
        [3, 0, 4],              # hip/gable near A
        [0, 3, 2], [0, 2, 1],   # base cap
    ]
    return _finish(np.asarray(verts, dtype=float), faces)


def _finish(verts, faces) -> trimesh.Trimesh | None:
    """Build, clean and validate a roof mesh."""
    try:
        m = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=True)
        if m.is_empty or len(m.faces) == 0:
            return None
        m.merge_vertices()
        m.fix_normals()
        return m
    except Exception:  # noqa: BLE001
        return None


def build_roof(poly, z0: float, roof_shape: str, roof_height: float,
               roof_direction: float | None = None) -> trimesh.Trimesh | None:
    """Dispatch to the right roof builder. Returns None for flat/unsupported."""
    if not roof_shape or roof_height <= 0:
        return None
    shape = str(roof_shape).strip().lower()
    if shape in ("pyramidal", "cone", "spire", "round"):
        return _apex_roof(poly, z0, roof_height)
    if shape == "dome":
        return _dome_roof(poly, z0, roof_height, onion=False)
    if shape == "onion":
        return _dome_roof(poly, z0, roof_height, onion=True)
    if shape in ("gabled", "pitched"):
        return _ridge_roof(poly, z0, roof_height, roof_direction, hipped=False)
    if shape in ("hipped", "skillion"):
        return _ridge_roof(poly, z0, roof_height, roof_direction, hipped=True)
    return None
