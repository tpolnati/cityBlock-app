"""2D geometry processing for LuminaMaps (Step 2 + landmark detail).

Pure ``shapely`` / ``geopandas`` operations. Turns the raw, projected OSM layers
(in UTM metres, from ``osm_fetcher``) into clean, millimetre-space geometry ready
for extrusion by ``mesh_generator``.

Landmark / detail handling
--------------------------
Ordinary buildings are unioned by height bin and extruded as simple prisms — fast
and clean. But major attractions (Burj Khalifa, cathedrals, towers…) must be
recognisable in the print, so they get special treatment:

* ``building:part`` polygons (the OSM "Simple 3D Buildings" scheme) are extruded
  individually using their own ``height`` + ``min_height`` — this reproduces the
  stacked setbacks and tapering spire that make a skyscraper identifiable.
* Landmark buildings are detected from tags (``tourism``/``historic``/
  ``man_made``/``wikidata``…, notable ``building`` values, or sheer height) and
  are kept at full footprint fidelity (little/no simplification) and never merged
  into neighbours.
* A **Landmark emphasis** multiplier lets these features tower above the urban
  fabric so they visually pop.

The ``Detail Level`` chosen in the UI selects how aggressively to apply the above.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.affinity import scale as shp_scale
from shapely.affinity import translate as shp_translate
from shapely.geometry import box
from shapely.ops import unary_union

from src.osm_fetcher import MapData

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
LEVEL_HEIGHT_M = 3.0            # 1 building level == 3 metres (per brief)
DEFAULT_MIN_M = 3.0            # random fallback height range (metres)
DEFAULT_MAX_M = 9.0
HEIGHT_BIN_MM = 0.5            # height quantisation for bulk footprint unioning
MIN_BUILDING_MM = 0.6            # floor on printed building height (mm)
MIN_PART_MM = 0.4            # floor on printed thickness of a single 3D part
DEFAULT_MAX_HEIGHT_MM = 50.0     # printable ceiling for the tallest feature (0 = off)

# A building counts as a landmark (recognisable attraction) if it carries any of
# these tags, has a notable building value, or is simply very tall.
LANDMARK_TAG_KEYS = ("tourism", "historic", "wikidata", "wikipedia")
LANDMARK_BUILDING_VALUES = {
    "cathedral", "church", "mosque", "temple", "synagogue", "chapel", "basilica",
    "stadium", "tower", "castle", "monument", "government", "palace",
    "train_station", "transportation", "museum", "city_hall", "skyscraper",
}
LANDMARK_NAMED_HEIGHT_M = 60.0    # a *named* building this tall is a landmark
LANDMARK_ANY_HEIGHT_M = 120.0     # anything this tall is a landmark regardless

# Detail Level presets. ``use_parts`` enables Simple-3D-Buildings detail;
# ``simplify_*`` are shapely tolerances in metres (0 = no simplification).
DETAIL_LEVELS: dict[str, dict] = {
    "Standard": {
        "use_parts": False, "landmarks": False,
        "simplify_bulk": 0.30, "simplify_detail": 0.30,
        "blurb": "Fast. Flat-topped building prisms; no 3D landmark detail.",
    },
    "High": {
        "use_parts": True, "landmarks": True,
        "simplify_bulk": 0.20, "simplify_detail": 0.05,
        "blurb": "Landmarks kept sharp; skyscraper setbacks via building:part.",
    },
    "Maximum": {
        "use_parts": True, "landmarks": True,
        "simplify_bulk": 0.10, "simplify_detail": 0.0,
        "blurb": "Every node preserved. Highest fidelity, most triangles.",
    },
}
DEFAULT_DETAIL_LEVEL = "High"

# Road carve widths by OSM highway class (full width, metres).
ROAD_WIDTH_M = {
    "motorway": 16.0, "motorway_link": 10.0,
    "trunk": 14.0, "primary": 12.0, "primary_link": 8.0,
    "secondary": 10.0, "secondary_link": 7.0,
    "tertiary": 8.0, "residential": 6.0, "unclassified": 6.0,
    "living_street": 5.0, "service": 4.0, "pedestrian": 4.0,
}
ROAD_WIDTH_DEFAULT_M = 6.0

WATERWAY_WIDTH_M = {
    "river": 14.0, "canal": 10.0, "stream": 4.0, "dock": 12.0, "riverbank": 16.0,
}
WATERWAY_WIDTH_DEFAULT_M = 6.0


# ---------------------------------------------------------------------------
# Output container — one per printed STL tile
# ---------------------------------------------------------------------------
@dataclass
class TileGeom:
    """Clean, millimetre-space geometry for a single printable tile."""

    name: str
    size_w_mm: float
    size_h_mm: float
    base_mm: float
    # Ordinary buildings: (height_mm, geometry), footprints unioned per height bin.
    building_bins: list[tuple[float, object]] = field(default_factory=list)
    # Landmarks / 3D parts: (z_bottom_mm, z_top_mm, geometry), extruded individually.
    detail_buildings: list[tuple[float, float, object]] = field(default_factory=list)
    water: object | None = None
    roads: object | None = None
    parks: object | None = None

    @property
    def building_count_bins(self) -> int:
        return len(self.building_bins)

    @property
    def detail_count(self) -> int:
        return len(self.detail_buildings)


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------
def project_center(center_latlon: tuple[float, float], utm_crs: str) -> tuple[float, float]:
    """Project the queried (lat, lon) into the map's UTM CRS -> (easting, northing)."""
    lat, lon = center_latlon
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)


# ---------------------------------------------------------------------------
# Tag parsing
# ---------------------------------------------------------------------------
def _parse_float(value) -> float | None:
    """Best-effort parse of an OSM numeric tag like '12', '12 m', '12.5'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        token = str(value).strip().split()[0].replace(",", ".")
        return float(token)
    except (ValueError, IndexError):
        return None


def _truthy(value) -> bool:
    """True if an OSM tag is set to a meaningful (non-empty, non-'no') value."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip().lower()
    return text not in ("", "no", "false", "0", "nan")


def _get(row, key):
    return row[key] if key in row.index else None


def building_z_range(row, rng: random.Random) -> tuple[float, float]:
    """Real-world (z_bottom, z_top) in metres for a building or building:part.

    ``height`` / ``building:levels`` give the top; ``min_height`` / ``min_level``
    give the floating base of an upper 3D part. Missing tops fall back to a
    random 3–9 m so plain buildings still have relief.
    """
    top = _parse_float(_get(row, "height"))
    if top is None:
        lv = _parse_float(_get(row, "building:levels"))
        if lv and lv > 0:
            top = lv * LEVEL_HEIGHT_M

    bottom = _parse_float(_get(row, "min_height"))
    if bottom is None:
        ml = _parse_float(_get(row, "min_level"))
        if ml and ml > 0:
            bottom = ml * LEVEL_HEIGHT_M
    bottom = max(0.0, bottom or 0.0)

    if top is None or top <= bottom:
        top = bottom + rng.uniform(DEFAULT_MIN_M, DEFAULT_MAX_M)
    return bottom, top


def is_landmark(row, top_m: float) -> bool:
    """Heuristic: is this building a recognisable attraction worth extra detail?"""
    if any(_truthy(_get(row, k)) for k in LANDMARK_TAG_KEYS):
        return True
    if _get(row, "man_made") is not None and str(_get(row, "man_made")).strip().lower() == "tower":
        return True
    bval = _get(row, "building")
    if isinstance(bval, str) and bval.strip().lower() in LANDMARK_BUILDING_VALUES:
        return True
    if top_m >= LANDMARK_ANY_HEIGHT_M:
        return True
    if top_m >= LANDMARK_NAMED_HEIGHT_M and _truthy(_get(row, "name")):
        return True
    return False


# ---------------------------------------------------------------------------
# Line -> polygon buffering for land features
# ---------------------------------------------------------------------------
def _buffer_layer(gdf: gpd.GeoDataFrame, width_for_row, default_width: float):
    if gdf is None or gdf.empty:
        return None
    pieces = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        gtype = geom.geom_type
        if gtype in ("LineString", "MultiLineString"):
            width = width_for_row(row) or default_width
            pieces.append(geom.buffer(width / 2.0, cap_style=2, join_style=1))
        elif gtype in ("Polygon", "MultiPolygon"):
            cleaned = geom.buffer(0)
            if not cleaned.is_empty:
                pieces.append(cleaned)
    if not pieces:
        return None
    merged = unary_union(pieces)
    return merged if not merged.is_empty else None


def _road_width(row) -> float:
    hw = _get(row, "highway")
    if isinstance(hw, list):
        hw = hw[0] if hw else None
    return ROAD_WIDTH_M.get(hw, ROAD_WIDTH_DEFAULT_M)


def _water_width(row) -> float:
    ww = _get(row, "waterway")
    if isinstance(ww, list):
        ww = ww[0] if ww else None
    return WATERWAY_WIDTH_M.get(ww, WATERWAY_WIDTH_DEFAULT_M)


# ---------------------------------------------------------------------------
# Small geometry utilities
# ---------------------------------------------------------------------------
def _only_polygons(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        return unary_union(polys) if polys else None
    return None


def _soft_cap(height_mm: float, cap_mm: float) -> float:
    """Smoothly bound a height toward ``cap_mm`` while staying near-linear when low.

    Uses ``cap * tanh(h / cap)``: a 5 mm feature under a 50 mm cap is essentially
    unchanged, but an 800 m skyscraper asymptotes toward the cap instead of
    printing a metre tall — and because the map is monotonic, stacked setbacks
    keep their relative proportions, so a landmark's silhouette survives.
    """
    if cap_mm <= 0 or height_mm <= 0:
        return height_mm
    return cap_mm * math.tanh(height_mm / cap_mm)


def _to_tile_mm(geom, tcx_m: float, tcy_m: float, scale: float):
    if geom is None or geom.is_empty:
        return None
    g = shp_translate(geom, xoff=-tcx_m, yoff=-tcy_m)
    g = shp_scale(g, xfact=scale, yfact=scale, origin=(0, 0))
    return g if not g.is_empty else None


def _crop(geom, tile_box):
    if geom is None or geom.is_empty:
        return None
    return _only_polygons(geom.intersection(tile_box))


# ---------------------------------------------------------------------------
# Building classification: bulk prisms vs detailed landmark/part pieces
# ---------------------------------------------------------------------------
def _classify_buildings(buildings: gpd.GeoDataFrame, level: dict, seed: int):
    """Split buildings into (bulk_gdf, detail_gdf).

    bulk_gdf   : ordinary footprints -> binned/unioned simple prisms. Carries a
                 ``height_m`` column.
    detail_gdf : landmarks + 3D parts -> extruded individually with full detail.
                 Carries ``zb_m`` / ``zt_m`` columns.
    """
    rng = random.Random(seed)
    use_parts = level["use_parts"]
    want_landmarks = level["landmarks"]

    is_part, is_outline, zb, zt, landmark = [], [], [], [], []
    for _, row in buildings.iterrows():
        part = _truthy(_get(row, "building:part")) and not _truthy(_get(row, "building"))
        outline = _truthy(_get(row, "building"))
        b, t = building_z_range(row, rng)
        is_part.append(part)
        is_outline.append(outline or not part)   # treat ambiguous rows as outlines
        zb.append(b)
        zt.append(t)
        landmark.append(want_landmarks and is_landmark(row, t))

    work = buildings.copy()
    work["_is_part"] = is_part
    work["_is_outline"] = is_outline
    work["zb_m"] = zb
    work["zt_m"] = zt
    work["_landmark"] = landmark

    parts = work[work["_is_part"]]
    outlines = work[work["_is_outline"]]

    if not use_parts:
        # Ignore 3D parts entirely; every outline is a bulk prism.
        bulk = outlines.copy()
        bulk["height_m"] = bulk["zt_m"]
        detail = work.iloc[0:0].copy()
        return bulk, detail

    # Detail mode: drop outlines that are covered by parts (avoid double walls).
    kept_outlines = outlines
    if not parts.empty and not outlines.empty:
        try:
            joined = gpd.sjoin(outlines, parts[["geometry"]], predicate="intersects", how="left")
            covered = joined.index[joined["index_right"].notna()].unique()
            kept_outlines = outlines.drop(index=covered)
        except Exception:  # noqa: BLE001 - spatial join hiccup -> keep all outlines
            kept_outlines = outlines

    landmark_outlines = kept_outlines[kept_outlines["_landmark"]]
    bulk_outlines = kept_outlines[~kept_outlines["_landmark"]]

    detail = pd.concat([parts, landmark_outlines])
    detail = gpd.GeoDataFrame(detail, geometry="geometry", crs=buildings.crs)

    bulk = bulk_outlines.copy()
    bulk["height_m"] = bulk["zt_m"]
    return bulk, detail


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def prepare_tiles(
    map_data: MapData,
    preset: dict,
    z_mult: float,
    *,
    detail_level: str = DEFAULT_DETAIL_LEVEL,
    landmark_emphasis: float = 1.0,
    max_height_mm: float = DEFAULT_MAX_HEIGHT_MM,
    include_water: bool = True,
    include_roads: bool = True,
    include_parks: bool = True,
    seed: int = 42,
) -> list[TileGeom]:
    """Turn fetched OSM layers into a list of millimetre-space ``TileGeom`` tiles.

    Re-running with new ``z_mult`` / ``detail_level`` / ``landmark_emphasis``
    recomputes geometry only — it never touches the network.
    """
    level = DETAIL_LEVELS.get(detail_level, DETAIL_LEVELS[DEFAULT_DETAIL_LEVEL])
    cx, cy = project_center(map_data.center_latlon, map_data.utm_crs)

    # --- Buildings: classify, recentre on full-crop centre, simplify ----------
    buildings = map_data.buildings
    buildings = buildings[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    bulk = detail = None
    if not buildings.empty:
        bulk, detail = _classify_buildings(buildings, level, seed)
        for gdf, tol in ((bulk, level["simplify_bulk"]), (detail, level["simplify_detail"])):
            if gdf is not None and not gdf.empty:
                geom = gdf.geometry.buffer(0).translate(xoff=-cx, yoff=-cy)
                if tol and tol > 0:
                    geom = geom.simplify(tol, preserve_topology=True)
                gdf.set_geometry(geom, inplace=True)

    # --- Land features: buffer to polygons, recentre (metres) -----------------
    water_geom = _buffer_layer(map_data.water, _water_width, WATERWAY_WIDTH_DEFAULT_M) if include_water else None
    roads_geom = _buffer_layer(map_data.roads, _road_width, ROAD_WIDTH_DEFAULT_M) if include_roads else None
    parks_geom = _buffer_layer(map_data.parks, lambda r: None, 0.0) if include_parks else None
    water_geom = shp_translate(water_geom, -cx, -cy) if water_geom is not None else None
    roads_geom = shp_translate(roads_geom, -cx, -cy) if roads_geom is not None else None
    parks_geom = shp_translate(parks_geom, -cx, -cy) if parks_geom is not None else None

    # --- Crop-box geometry sized to the preset's physical aspect ratio --------
    full_w_mm, full_h_mm = preset["size_mm"]
    radius = map_data.radius_m
    aspect = full_w_mm / full_h_mm
    if aspect >= 1.0:
        half_w_m, half_h_m = radius, radius / aspect
    else:
        half_w_m, half_h_m = radius * aspect, radius
    scale = full_w_mm / (2.0 * half_w_m)          # mm per metre (uniform)

    tiles_x, tiles_y = preset["tiles"]
    tile_w_m = (2.0 * half_w_m) / tiles_x
    tile_h_m = (2.0 * half_h_m) / tiles_y
    tile_w_mm, tile_h_mm = preset["tile_size_mm"]

    detail_mult = z_mult * max(0.0, landmark_emphasis)
    if detail is not None and not detail.empty:
        detail = _assign_detail_heights(detail, detail_mult, scale, max_height_mm)

    results: list[TileGeom] = []
    for j in range(tiles_y):
        for i in range(tiles_x):
            tcx = -half_w_m + tile_w_m * (i + 0.5)
            tcy = half_h_m - tile_h_m * (j + 0.5)
            tile_box = box(
                tcx - tile_w_m / 2.0, tcy - tile_h_m / 2.0,
                tcx + tile_w_m / 2.0, tcy + tile_h_m / 2.0,
            )

            name = "tile" if (tiles_x * tiles_y) == 1 else f"tile_r{j}_c{i}"
            tile = TileGeom(
                name=name,
                size_w_mm=tile_w_mm, size_h_mm=tile_h_mm, base_mm=preset["base_mm"],
                water=_to_tile_mm(_crop(water_geom, tile_box), tcx, tcy, scale),
                roads=_to_tile_mm(_crop(roads_geom, tile_box), tcx, tcy, scale),
                parks=_to_tile_mm(_crop(parks_geom, tile_box), tcx, tcy, scale),
            )

            if bulk is not None and not bulk.empty:
                in_tile = gpd.clip(bulk, tile_box)
                in_tile = in_tile[in_tile.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
                tile.building_bins = _bin_buildings(in_tile, tcx, tcy, scale, z_mult, max_height_mm)

            if detail is not None and not detail.empty:
                in_tile = gpd.clip(detail, tile_box)
                in_tile = in_tile[in_tile.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
                tile.detail_buildings = _detail_pieces(in_tile, tcx, tcy, scale, detail_mult, max_height_mm)

            results.append(tile)

    return results


def _bin_buildings(in_tile: gpd.GeoDataFrame, tcx, tcy, scale, z_mult, max_height_mm):
    """Group cropped bulk buildings by quantised height and union each group."""
    if in_tile.empty:
        return []
    bins: dict[float, list] = {}
    for _, row in in_tile.iterrows():
        geom = _to_tile_mm(row.geometry, tcx, tcy, scale)
        if geom is None:
            continue
        height_mm = _soft_cap(row["height_m"] * z_mult * scale, max_height_mm)
        height_mm = max(MIN_BUILDING_MM, height_mm)
        key = max(MIN_BUILDING_MM, round(round(height_mm / HEIGHT_BIN_MM) * HEIGHT_BIN_MM, 3))
        bins.setdefault(key, []).append(geom)
    out = []
    for height_mm, geoms in sorted(bins.items()):
        merged = _only_polygons(unary_union(geoms))
        if merged is not None and not merged.is_empty:
            out.append((height_mm, merged))
    return out


def _assign_detail_heights(detail: gpd.GeoDataFrame, mult: float, scale: float, cap: float):
    """Add printed ``pz_b`` / ``pz_t`` (mm) columns to landmark/part features.

    A landmark's overall top is soft-capped to the printable ceiling, but its
    internal 3D parts are then scaled *proportionally to that landmark's own real
    height*. That keeps the tapered silhouette (setbacks, spire) intact instead
    of letting ``tanh`` saturation flatten every upper part to the same level.

    Parts are grouped into landmarks by connected (overlapping) footprints.
    """
    detail = detail.copy()
    pz_b = pd.Series(0.0, index=detail.index)
    pz_t = pd.Series(0.0, index=detail.index)

    parts = detail[detail["_is_part"]]
    outlines = detail[~detail["_is_part"]]

    # Landmark outlines without 3D parts: a single capped prism from the ground.
    for idx, row in outlines.iterrows():
        pz_t[idx] = _soft_cap(row["zt_m"] * mult * scale, cap)

    if not parts.empty:
        try:
            merged = unary_union(list(parts.geometry.values))
            clusters = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
            cgdf = gpd.GeoDataFrame({"cid": range(len(clusters))}, geometry=clusters, crs=parts.crs)
            pts = parts.copy()
            pts.set_geometry(parts.geometry.representative_point(), inplace=True)
            joined = gpd.sjoin(pts, cgdf, predicate="within", how="left")
            groups = joined.groupby("index_right")
        except Exception:  # noqa: BLE001 - clustering failed; cap each part on its own
            groups = [(None, parts)]

        for _cid, grp in groups:
            idxs = list(grp.index)
            h_real = max(float(parts.loc[i, "zt_m"]) for i in idxs)
            if h_real <= 0:
                continue
            target = _soft_cap(h_real * mult * scale, cap)   # printed mm for the top
            for i in idxs:
                pz_t[i] = target * (float(parts.loc[i, "zt_m"]) / h_real)
                pz_b[i] = target * (max(0.0, float(parts.loc[i, "zb_m"])) / h_real)

    detail["pz_b"] = pz_b
    detail["pz_t"] = pz_t
    return detail


def _detail_pieces(in_tile: gpd.GeoDataFrame, tcx, tcy, scale, mult, max_height_mm):
    """Each landmark/part kept individually as (z_bottom_mm, z_top_mm, geom).

    Printed heights were pre-computed per landmark in ``_assign_detail_heights``
    (so the taper is preserved); here we just read them and place the geometry.
    """
    if in_tile.empty:
        return []
    out = []
    for _, row in in_tile.iterrows():
        geom = _to_tile_mm(row.geometry, tcx, tcy, scale)
        if geom is None:
            continue
        zb = max(0.0, float(row["pz_b"])) if "pz_b" in row.index else 0.0
        zt = float(row["pz_t"]) if "pz_t" in row.index else _soft_cap(row["zt_m"] * mult * scale, max_height_mm)
        if zt - zb < MIN_PART_MM:
            zt = zb + MIN_PART_MM
        out.append((zb, zt, geom))
    return out
