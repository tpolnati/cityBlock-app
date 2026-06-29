"""3D mesh generation + STL export for LuminaMaps (Step 2).

All ``trimesh`` work lives here. Consumes millimetre-space ``TileGeom`` objects
from ``geometry_processor`` and produces watertight, manifold-friendly STL
meshes, one per printed tile.

Construction per tile
----------------------
* Solid rectangular **base plate**: bottom flush at Z=0, top at Z=base_mm.
* **Buildings**: extruded prisms, slightly embedded into the base for a solid
  weld, rising from the base top.
* **Parks**: a thin raised pad on the base top (subtle land texture).
* **Water / roads**: carved as recessed channels into the base top via boolean
  difference (the "negative space"), with a graceful fallback if the boolean
  engine struggles on pathological geometry.
* Everything is centred at X=0, Y=0 and the bottom sits flush at Z=0.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import trimesh

from src import landmark_models, roofs
from src.geometry_processor import LandmarkRecord, TileGeom

# Depth (mm) of carved water/road channels, capped relative to base thickness.
CARVE_MAX_MM = 1.2
# How far (mm) buildings/parks sink into the base top to guarantee a solid weld.
EMBED_MM = 0.2
# Thin raised park layer thickness (mm).
PARK_MAX_MM = 0.8


@dataclass
class TileMesh:
    """A finished tile mesh plus a few stats for the UI."""

    name: str
    mesh: trimesh.Trimesh
    # Human-readable note per landmark that got a detailed downloaded model.
    model_sources: list[str] = field(default_factory=list)

    @property
    def triangles(self) -> int:
        return len(self.mesh.faces)

    @property
    def watertight(self) -> bool:
        return bool(self.mesh.is_watertight)

    @property
    def dims_mm(self) -> tuple[float, float, float]:
        ext = self.mesh.extents
        return (float(ext[0]), float(ext[1]), float(ext[2]))

    def to_stl_bytes(self) -> bytes:
        return self.mesh.export(file_type="stl")


# ---------------------------------------------------------------------------
# Polygon -> mesh helpers
# ---------------------------------------------------------------------------
def _iter_polys(geom):
    """Yield individual shapely Polygons from any (Multi)Polygon / collection."""
    if geom is None or geom.is_empty:
        return
    gtype = geom.geom_type
    if gtype == "Polygon":
        yield geom
    elif gtype in ("MultiPolygon", "GeometryCollection"):
        for g in geom.geoms:
            yield from _iter_polys(g)


def _extrude(geom, height: float, z0: float = 0.0):
    """Extrude a (Multi)Polygon to a given height, optionally lifted to ``z0``.

    Returns a single concatenated ``Trimesh`` or ``None``. Individual polygon
    failures are skipped so one bad ring can't sink the tile.
    """
    if height <= 0:
        return None
    meshes = []
    for poly in _iter_polys(geom):
        if poly.area <= 0:
            continue
        poly = poly.buffer(0)            # heal self-touching rings
        for clean in _iter_polys(poly):
            if clean.area <= 0:
                continue
            try:
                m = trimesh.creation.extrude_polygon(clean, height)
            except Exception:            # noqa: BLE001 - degenerate triangulation
                continue
            if z0:
                m.apply_translation([0.0, 0.0, z0])
            meshes.append(m)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def _base_plate(width_mm: float, height_mm: float, thickness_mm: float) -> trimesh.Trimesh:
    """Solid centred base box: bottom at Z=0, top at Z=thickness."""
    plate = trimesh.creation.box(extents=[width_mm, height_mm, thickness_mm])
    plate.apply_translation([0.0, 0.0, thickness_mm / 2.0])
    return plate


def _carve(base: trimesh.Trimesh, cutter_geom, depth: float, top_z: float):
    """Subtract recessed channels (cutter extruded down from ``top_z``) from base.

    Returns (mesh, carved?). On any boolean failure, returns the untouched base
    so the tile still exports.
    """
    if cutter_geom is None or depth <= 0:
        return base, False
    cutter = _extrude(cutter_geom, depth)
    if cutter is None:
        return base, False
    cutter.apply_translation([0.0, 0.0, top_z - depth])
    try:
        result = trimesh.boolean.difference([base, cutter])
        if result is None or result.is_empty or len(result.faces) == 0:
            return base, False
        return result, True
    except Exception:                    # noqa: BLE001 - boolean engine hiccup
        return base, False


# ---------------------------------------------------------------------------
# Tile assembly
# ---------------------------------------------------------------------------
def _build_landmark(lm: LandmarkRecord, base_t: float, *,
                    fetch_models: bool, sketchfab_token: str | None):
    """Return (meshes, source_note). Tries a detailed downloaded model, else
    falls back to extruded walls plus an optional procedural roof."""
    # 1) Detailed model injection (Wikidata/Commons -> Sketchfab).
    if fetch_models and (lm.qid or (lm.name and sketchfab_token)):
        try:
            model, source = landmark_models.fetch_landmark_mesh(lm.qid, lm.name, sketchfab_token)
        except Exception:                # noqa: BLE001
            model, source = None, None
        if model is not None:
            placed = landmark_models.fit_model(
                model,
                centroid_mm=lm.centroid_mm,
                target_height_mm=lm.target_height_mm,
                base_top_mm=base_t,
                embed_mm=EMBED_MM,
                max_footprint_mm=lm.max_footprint_mm * 1.6,
            )
            if placed is not None and len(placed.faces) > 0:
                return [placed], f"{lm.name or lm.qid}: {source}"

    # 2) Procedural fallback: extruded walls (+ roof if tagged).
    meshes = []
    for z_bottom, z_top, geom in lm.wall_pieces:
        height = (z_top - z_bottom) + EMBED_MM
        if height <= 0:
            continue
        wm = _extrude(geom, height, z0=base_t + z_bottom - EMBED_MM)
        if wm is not None:
            meshes.append(wm)
    if lm.roof_shape and lm.roof_height_mm > 0:
        roof = roofs.build_roof(
            lm.footprint_mm, base_t + lm.roof_base_mm - EMBED_MM,
            lm.roof_shape, lm.roof_height_mm + EMBED_MM, lm.roof_direction,
        )
        if roof is not None:
            meshes.append(roof)
    return meshes, None


def build_tile(tile: TileGeom, *, carve_features: bool = True, weld: bool = False,
               fetch_models: bool = False, sketchfab_token: str | None = None) -> TileMesh:
    """Assemble a single printable tile mesh from its prepared geometry."""
    base_t = tile.base_mm
    parts: list[trimesh.Trimesh] = []
    sources: list[str] = []

    # --- Base plate, optionally carved with water + roads ---------------------
    base = _base_plate(tile.size_w_mm, tile.size_h_mm, base_t)
    if carve_features:
        cutters = [g for g in (tile.water, tile.roads) if g is not None]
        if cutters:
            from shapely.ops import unary_union

            depth = min(CARVE_MAX_MM, base_t * 0.6)
            base, _ = _carve(base, unary_union(cutters), depth, base_t)
    parts.append(base)

    # --- Thin raised park pads ------------------------------------------------
    if tile.parks is not None:
        park_t = min(PARK_MAX_MM, base_t * 0.4)
        pm = _extrude(tile.parks, park_t, z0=base_t)
        if pm is not None:
            parts.append(pm)

    # --- Bulk buildings (slightly embedded into the base) ---------------------
    for height_mm, geom in tile.building_bins:
        bm = _extrude(geom, height_mm + EMBED_MM, z0=base_t - EMBED_MM)
        if bm is not None:
            parts.append(bm)

    # --- Generic 3D parts (not named landmarks): stacked z_bottom..z_top -------
    for z_bottom, z_top, geom in tile.detail_buildings:
        height = (z_top - z_bottom) + EMBED_MM
        if height <= 0:
            continue
        dm = _extrude(geom, height, z0=base_t + z_bottom - EMBED_MM)
        if dm is not None:
            parts.append(dm)

    # --- Named landmarks: detailed model injection or procedural fallback ------
    for lm in tile.landmarks:
        meshes, source = _build_landmark(
            lm, base_t, fetch_models=fetch_models, sketchfab_token=sketchfab_token)
        parts.extend(meshes)
        if source:
            sources.append(source)

    # --- Combine --------------------------------------------------------------
    if weld and len(parts) > 1:
        try:
            mesh = trimesh.boolean.union(parts)
            if mesh is None or mesh.is_empty:
                mesh = trimesh.util.concatenate(parts)
        except Exception:                # noqa: BLE001
            mesh = trimesh.util.concatenate(parts)
    else:
        mesh = trimesh.util.concatenate(parts)

    _finalize(mesh)
    return TileMesh(name=tile.name, mesh=mesh, model_sources=sources)


def _finalize(mesh: trimesh.Trimesh) -> None:
    """Clean up + guarantee the mesh is centred in XY and flush at Z=0."""
    mesh.merge_vertices()
    # trimesh 4.x: filter faces via boolean/index masks rather than removed helpers.
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.fix_normals()
    # Centre X/Y on the origin; drop the bottom to exactly Z=0.
    minx, miny, minz = mesh.bounds[0]
    maxx, maxy, _ = mesh.bounds[1]
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    mesh.apply_translation([-cx, -cy, -minz])


def build_all(tiles: list[TileGeom], *, carve_features: bool = True, weld: bool = False,
              fetch_models: bool = False, sketchfab_token: str | None = None) -> list[TileMesh]:
    """Build every tile in a preset (1 tile normally, 4 for the Tier-3 mega map)."""
    return [
        build_tile(t, carve_features=carve_features, weld=weld,
                   fetch_models=fetch_models, sketchfab_token=sketchfab_token)
        for t in tiles
    ]
