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
import re
from collections import defaultdict
from dataclasses import dataclass, field

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.affinity import scale as shp_scale
from shapely.affinity import translate as shp_translate
from shapely.geometry import box
from shapely.ops import unary_union

from src import roofs, terrain as terrain_mod
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
DEFAULT_TERRAIN_EXAGG = 2.0      # vertical exaggeration for terrain relief
DEFAULT_TERRAIN_CAP_MM = 25.0    # max printed terrain relief above the base (0 = off)
TERRAIN_TILE_RES = 48            # per-tile terrain heightmap resolution

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
LANDMARK_MANMADE = {"tower", "monument", "obelisk", "statue", "lighthouse", "campanile"}

# Standalone attractions (statues/monuments that aren't buildings).
MONUMENT_DEFAULT_HEIGHT_M = 30.0  # fallback print height for an untagged monument
MONUMENT_POINT_RADIUS_M = 6.0     # footprint radius given to a point attraction
MONUMENT_MAX_AREA_M2 = 8000.0     # skip huge attraction areas (theme parks etc.)

# Default share of a building's printed height given to its roof when the OSM
# roof:height tag is missing, keyed by roof shape.
DEFAULT_ROOF_FRAC = {
    "spire": 0.55, "pyramidal": 0.5, "cone": 0.5, "round": 0.5,
    "dome": 0.4, "onion": 0.45,
    "gabled": 0.35, "pitched": 0.35, "hipped": 0.3, "skillion": 0.3,
}

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

# Road widths by OSM highway class (full width, metres).
ROAD_WIDTH_M = {
    "motorway": 16.0, "motorway_link": 10.0,
    "trunk": 14.0, "trunk_link": 9.0, "primary": 12.0, "primary_link": 8.0,
    "secondary": 10.0, "secondary_link": 7.0,
    "tertiary": 8.0, "tertiary_link": 6.0, "residential": 6.0, "unclassified": 6.0,
    "living_street": 5.0, "service": 4.0, "pedestrian": 4.0, "road": 6.0,
}
ROAD_WIDTH_DEFAULT_M = 6.0

# Importance rank per highway class (higher = bigger road). Anything not listed
# (footway/path/track/cycleway/steps…) is a "super small road" and is dropped.
ROAD_CLASS_RANK = {
    "motorway": 5, "motorway_link": 5, "trunk": 5, "trunk_link": 5,
    "primary": 4, "primary_link": 4, "secondary": 4, "secondary_link": 4,
    "tertiary": 3, "tertiary_link": 3,
    "residential": 2, "unclassified": 2, "road": 2,
    "living_street": 1, "service": 1, "pedestrian": 1,
}
# "Road detail" presets -> minimum rank kept.
ROAD_DETAIL_LEVELS = {
    "Major roads only": 3,          # motorway..tertiary
    "Major + residential": 2,       # + residential/unclassified   (default)
    "All roads (incl. service)": 1,  # + service/living_street/pedestrian
}
DEFAULT_ROAD_DETAIL = "Major + residential"
BRIDGE_MIN_RANK = 2                  # ignore bridges on tiny/service-only ways

WATERWAY_WIDTH_M = {
    "river": 14.0, "canal": 10.0, "stream": 4.0, "dock": 12.0, "riverbank": 16.0,
}
WATERWAY_WIDTH_DEFAULT_M = 6.0


# ---------------------------------------------------------------------------
# A named landmark, grouped for detailed-model injection (built per tile)
# ---------------------------------------------------------------------------
@dataclass
class LandmarkRecord:
    """One landmark we'll try to render with a detailed downloaded model.

    All geometry is in tile millimetre space. If no model is found, the caller
    falls back to ``wall_pieces`` (extruded footprints) plus an optional
    procedural ``roof``.
    """

    key: str
    name: str | None
    qid: str | None
    centroid_mm: tuple[float, float]
    target_height_mm: float                 # full printed height (for model scaling)
    max_footprint_mm: float                 # real footprint extent (clamps model width)
    footprint_mm: object                    # shapely geometry (union of the group)
    wall_pieces: list[tuple[float, float, object]] = field(default_factory=list)
    roof_shape: str | None = None
    roof_base_mm: float = 0.0               # z (above base top) where the roof starts
    roof_height_mm: float = 0.0
    roof_direction: float | None = None


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
    # Generic 3D parts (not a named landmark): (z_bottom_mm, z_top_mm, geometry).
    detail_buildings: list[tuple[float, float, object]] = field(default_factory=list)
    # Named landmarks grouped for detailed-model injection / roof reconstruction.
    landmarks: list[LandmarkRecord] = field(default_factory=list)
    water: object | None = None
    roads: object | None = None
    # Bridges as centrelines: list of (LineString_mm, width_mm) for real decks/piers.
    bridges: list = field(default_factory=list)
    parks: object | None = None
    # Per-tile elevation surface (mm space); None = flat base.
    terrain: object | None = None

    @property
    def building_count_bins(self) -> int:
        return len(self.building_bins)

    @property
    def detail_count(self) -> int:
        return len(self.detail_buildings) + sum(len(lm.wall_pieces) for lm in self.landmarks)

    @property
    def landmark_count(self) -> int:
        return len(self.landmarks)


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
    if isinstance(value, pd.Series):
        value = value.iloc[0] if len(value) else None
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip().lower()
    return text not in ("", "no", "false", "0", "nan")


def _as_float(value, default: float = 0.0) -> float:
    """Coerce a cell to float, tolerating a Series (duplicate column labels)."""
    if isinstance(value, pd.Series):
        value = value.iloc[0] if len(value) else None
    try:
        f = float(value)
        return default if f != f else f      # guard NaN
    except (TypeError, ValueError):
        return default


def _get(row, key):
    return row[key] if key in row.index else None


def _first(value):
    """OSM tags can arrive as a list when an element has multiple values."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def wikidata_qid(value) -> str | None:
    """Extract a Wikidata Q-id (e.g. 'Q243') from an OSM ``wikidata`` tag."""
    value = _first(value)
    if not _truthy(value):
        return None
    match = re.search(r"Q\d+", str(value))
    return match.group(0) if match else None


def clean_name(value) -> str | None:
    value = _first(value)
    return str(value).strip() if _truthy(value) else None


def roof_shape_of(value) -> str | None:
    value = _first(value)
    return str(value).strip().lower() if _truthy(value) else None


def is_attraction(row) -> bool:
    """A standalone monument/statue: not a building, but a tourism/historic/man_made feature."""
    if _truthy(_get(row, "building")):
        return False
    return any(_truthy(_get(row, k)) for k in ("tourism", "historic", "man_made"))


def building_z_range(row, rng: random.Random) -> tuple[float, float]:
    """Real-world (z_bottom, z_top) in metres for a building, part or monument.

    ``height`` / ``building:levels`` give the top; ``min_height`` / ``min_level``
    give the floating base of an upper 3D part. Missing tops fall back to a
    monument default for attractions, else a random 3–9 m so plain buildings
    still have relief.
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
        if is_attraction(row):
            top = bottom + MONUMENT_DEFAULT_HEIGHT_M
        else:
            top = bottom + rng.uniform(DEFAULT_MIN_M, DEFAULT_MAX_M)
    return bottom, top


def is_landmark(row, top_m: float) -> bool:
    """Heuristic: is this a recognisable attraction worth a detailed model?"""
    if any(_truthy(_get(row, k)) for k in LANDMARK_TAG_KEYS):
        return True
    mm = _first(_get(row, "man_made"))
    if isinstance(mm, str) and mm.strip().lower() in LANDMARK_MANMADE:
        return True
    bval = _first(_get(row, "building"))
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


def _iter_lines(geom):
    """Yield LineStrings from any Line/MultiLine/GeometryCollection."""
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "LineString":
        yield geom
    elif geom.geom_type in ("MultiLineString", "GeometryCollection"):
        for g in geom.geoms:
            yield from _iter_lines(g)


def _build_roads(gdf: gpd.GeoDataFrame, min_rank: int):
    """Filter roads by importance and split bridges out.

    Returns ``(roads_geom, bridge_lines)`` where ``roads_geom`` is a buffered
    polygon in metres (or None) and ``bridge_lines`` is a list of
    ``(LineString, width_m)`` centrelines — kept as lines so the mesh stage can
    build a real elevated deck with piers. Tiny paths are dropped entirely.
    """
    if gdf is None or gdf.empty:
        return None, []
    road_pieces, bridge_lines, bridge_polys = [], [], []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        hw = _first(_get(row, "highway"))
        rank = ROAD_CLASS_RANK.get(hw)
        if rank is None or rank < min_rank:
            continue
        width = ROAD_WIDTH_M.get(hw, ROAD_WIDTH_DEFAULT_M)
        gtype = geom.geom_type
        is_bridge = _truthy(_get(row, "bridge")) and rank >= BRIDGE_MIN_RANK

        if gtype in ("LineString", "MultiLineString"):
            if is_bridge:
                for seg in _iter_lines(geom):
                    bridge_lines.append((seg, width))
                bridge_polys.append(geom.buffer(width / 2.0, cap_style=2, join_style=1))
            else:
                road_pieces.append(geom.buffer(width / 2.0, cap_style=2, join_style=1))
        elif gtype in ("Polygon", "MultiPolygon"):
            road_pieces.append(geom.buffer(0))   # area road/pedestrian square

    roads_geom = unary_union(road_pieces) if road_pieces else None
    # Don't draw the flat road under a bridge deck.
    if roads_geom is not None and bridge_polys:
        roads_geom = roads_geom.difference(unary_union(bridge_polys))
        if roads_geom.is_empty:
            roads_geom = None
    return roads_geom, bridge_lines


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


def _bridge_lines_to_tile(bridge_lines, tile_box, tcx, tcy, scale):
    """Crop bridge centrelines to a tile and convert to (LineString_mm, width_mm)."""
    out = []
    for line, width_m in bridge_lines:
        if line is None or line.is_empty:
            continue
        clipped = line.intersection(tile_box)
        for seg in _iter_lines(clipped):
            seg_mm = _to_tile_mm(seg, tcx, tcy, scale)
            if seg_mm is not None and seg_mm.length > 1.0:
                out.append((seg_mm, width_m * scale))
    return out


# ---------------------------------------------------------------------------
# Building classification: bulk prisms vs detailed landmark/part pieces
# ---------------------------------------------------------------------------
def _landmark_key(qid, name, fallback: str) -> str:
    """NaN/None-safe landmark group key (a bare ``x or y`` mishandles float NaN)."""
    if _truthy(qid):
        return str(qid)
    if _truthy(name):
        return f"name:{name}"
    return fallback


def _roof_frac(shape, roof_h_m, total_m) -> float:
    """Fraction of a building's height occupied by its roof."""
    if not shape or shape not in roofs.SUPPORTED_SHAPES:
        return 0.0
    if roof_h_m and total_m and total_m > 0:
        return max(0.05, min(0.7, roof_h_m / total_m))
    return DEFAULT_ROOF_FRAC.get(shape, 0.35)


def _annotate(buildings: gpd.GeoDataFrame, level: dict, seed: int) -> gpd.GeoDataFrame:
    """Add per-row geometry/identity/height columns used by all later stages."""
    rng = random.Random(seed)
    want_landmarks = level["landmarks"]
    work = buildings.copy()

    cols = {k: [] for k in (
        "_is_part", "_is_outline", "zb_m", "zt_m", "_landmark",
        "lm_qid", "lm_name", "roof_shape", "roof_height_m", "roof_dir", "roof_frac",
    )}
    for _, row in work.iterrows():
        part = _truthy(_get(row, "building:part")) and not _truthy(_get(row, "building"))
        outline = _truthy(_get(row, "building"))
        b, t = building_z_range(row, rng)
        shape = roof_shape_of(_get(row, "roof:shape"))
        rh = _parse_float(_get(row, "roof:height"))
        # Give untagged monuments/statues a tapered obelisk marker so they read
        # as landmarks even when no detailed model is downloaded.
        if shape is None and is_attraction(row):
            shape = "pyramidal"
        cols["_is_part"].append(part)
        cols["_is_outline"].append(outline or not part)
        cols["zb_m"].append(b)
        cols["zt_m"].append(t)
        cols["_landmark"].append(want_landmarks and is_landmark(row, t))
        cols["lm_qid"].append(wikidata_qid(_get(row, "wikidata")))
        cols["lm_name"].append(clean_name(_get(row, "name")))
        cols["roof_shape"].append(shape)
        cols["roof_height_m"].append(rh)
        cols["roof_dir"].append(_parse_float(_get(row, "roof:direction")))
        cols["roof_frac"].append(_roof_frac(shape, rh, t))
    for k, v in cols.items():
        work[k] = v
    return work


def _classify_buildings(buildings: gpd.GeoDataFrame, level: dict, seed: int):
    """Split buildings into (bulk_gdf, detail_gdf).

    bulk_gdf   : ordinary footprints -> binned simple prisms (``height_m``).
    detail_gdf : landmarks + 3D parts -> kept individually, tagged with a landmark
                 group key (``lm_key``), the group's real top height
                 (``cluster_h_m``) and inherited identity/roof columns so a tower's
                 wikidata id survives even when its outline is covered by parts.
    """
    work = _annotate(buildings, level, seed)
    parts = work[work["_is_part"]]
    outlines = work[work["_is_outline"]]

    if not level["use_parts"]:
        bulk = outlines.copy()
        bulk["height_m"] = bulk["zt_m"]
        return bulk, work.iloc[0:0].copy()

    cluster_h = work["zt_m"].astype(float).copy()
    lm_key = pd.Series([None] * len(work), index=work.index, dtype=object)

    # Each landmark outline is its own group.
    for idx, row in outlines.iterrows():
        if row["_landmark"]:
            lm_key[idx] = _landmark_key(row["lm_qid"], row["lm_name"], f"idx:{idx}")

    # Cluster parts into connected structures; transfer identity from any
    # overlapping landmark outline (covered or not) and record which outlines
    # are covered (so they drop out of the bulk layer).
    covered: set = set()
    if not parts.empty:
        try:
            merged = unary_union(list(parts.geometry.values))
            clusters = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
            cgdf = gpd.GeoDataFrame({"cid": range(len(clusters))}, geometry=clusters, crs=parts.crs)
            pts = parts.copy()
            pts.set_geometry(parts.geometry.representative_point(), inplace=True)
            joined = gpd.sjoin(pts, cgdf, predicate="within", how="left")
            cid_parts = defaultdict(list)
            for pidx, crow in joined.iterrows():
                cid_parts[crow["index_right"]].append(pidx)
        except Exception:  # noqa: BLE001 - clustering failed: one big group
            clusters = [None]
            cid_parts = {0: list(parts.index)}

        for cid, pidxs in cid_parts.items():
            try:
                cl_geom = clusters[int(cid)]
            except (TypeError, ValueError, IndexError):
                cl_geom = unary_union([work.loc[i, "geometry"] for i in pidxs])
            h_real = max(_as_float(work.loc[i, "zt_m"]) for i in pidxs)

            cand = outlines[outlines.intersects(cl_geom)] if cl_geom is not None else outlines.iloc[0:0]
            covered.update(cand.index.tolist())
            lm_cand = cand[cand["_landmark"]]
            chosen = lm_cand.iloc[0] if not lm_cand.empty else None
            key = None
            if chosen is not None:
                key = _landmark_key(chosen["lm_qid"], chosen["lm_name"], f"cid:{cid}")
                h_real = max(h_real, _as_float(chosen["zt_m"]))

            for i in pidxs:
                cluster_h[i] = h_real
                if key is not None:
                    lm_key[i] = key
                    work.at[i, "lm_qid"] = chosen["lm_qid"]
                    work.at[i, "lm_name"] = chosen["lm_name"]
                    work.at[i, "roof_shape"] = chosen["roof_shape"]
                    work.at[i, "roof_height_m"] = chosen["roof_height_m"]
                    work.at[i, "roof_dir"] = chosen["roof_dir"]
                    work.at[i, "roof_frac"] = chosen["roof_frac"]

    work["cluster_h_m"] = cluster_h
    work["lm_key"] = lm_key

    # Re-derive the layers from `work` *after* the new columns exist, so detail
    # rows carry lm_key / cluster_h_m / inherited identity.
    covered_mask = work.index.isin(covered)
    detail_mask = work["_is_part"] | (work["_is_outline"] & work["_landmark"] & ~covered_mask)
    bulk_mask = work["_is_outline"] & ~work["_landmark"] & ~covered_mask

    detail = gpd.GeoDataFrame(work[detail_mask].copy(), geometry="geometry", crs=buildings.crs)
    bulk = work[bulk_mask].copy()
    bulk["height_m"] = bulk["zt_m"]
    return bulk, detail


def _prepare_attractions(gdf: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame):
    """Normalise non-building attractions into small footprints for the landmark path.

    Point attractions (statues/monuments) are buffered into a small footprint;
    polygon attractions are kept if not enormous. Attractions that overlap an
    existing building are dropped (that building already represents them).
    """
    if gdf is None or gdf.empty:
        return None

    geoms, keep = [], []
    for _, row in gdf.iterrows():
        g = row.geometry
        if g is None or g.is_empty:
            geoms.append(None)
            keep.append(False)
            continue
        gt = g.geom_type
        if gt in ("Point", "MultiPoint"):
            poly = g.buffer(MONUMENT_POINT_RADIUS_M)
        elif gt in ("Polygon", "MultiPolygon"):
            poly = g.buffer(0)
            if poly.is_empty or poly.area > MONUMENT_MAX_AREA_M2:
                geoms.append(None)
                keep.append(False)
                continue
        else:                                   # lines etc. aren't monuments
            geoms.append(None)
            keep.append(False)
            continue
        geoms.append(poly)
        keep.append(True)

    out = gdf.copy()
    out["geometry"] = geoms
    out = out[pd.Series(keep, index=gdf.index)]
    if out.empty:
        return None
    out = out.set_geometry("geometry")

    # Drop attractions already represented by a building footprint.
    if buildings is not None and not buildings.empty:
        try:
            joined = gpd.sjoin(out, buildings[["geometry"]], predicate="intersects", how="left")
            covered = joined.index[joined["index_right"].notna()].unique()
            out = out.drop(index=covered)
        except Exception:  # noqa: BLE001 - dedup is best-effort
            pass
    return out if not out.empty else None


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
    terrain=None,
    terrain_exaggeration: float = DEFAULT_TERRAIN_EXAGG,
    terrain_cap_mm: float = DEFAULT_TERRAIN_CAP_MM,
    include_water: bool = True,
    include_roads: bool = True,
    include_parks: bool = True,
    road_detail: str = DEFAULT_ROAD_DETAIL,
    seed: int = 42,
) -> list[TileGeom]:
    """Turn fetched OSM layers into a list of millimetre-space ``TileGeom`` tiles.

    Re-running with new ``z_mult`` / ``detail_level`` / ``landmark_emphasis``
    recomputes geometry only — it never touches the network.
    """
    level = DETAIL_LEVELS.get(detail_level, DETAIL_LEVELS[DEFAULT_DETAIL_LEVEL])
    cx, cy = project_center(map_data.center_latlon, map_data.utm_crs)

    # --- Buildings (+ standalone attractions): classify, recentre, simplify ----
    buildings = map_data.buildings
    buildings = buildings[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    attractions = _prepare_attractions(getattr(map_data, "attractions", None), buildings)
    # Merge with a fresh unique index (ignore_index) and drop duplicate column
    # labels — real OSM data can repeat both, which otherwise makes scalar lookups
    # return a Series (and collapses distinct landmarks that share an osmid).
    frames = [f for f in (buildings, attractions) if f is not None and not f.empty]
    if frames:
        buildings = pd.concat(frames, ignore_index=True)
        buildings = buildings.loc[:, ~buildings.columns.duplicated()]
        buildings = gpd.GeoDataFrame(buildings, geometry="geometry", crs=map_data.utm_crs)
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
    if include_roads:
        min_rank = ROAD_DETAIL_LEVELS.get(road_detail, ROAD_DETAIL_LEVELS[DEFAULT_ROAD_DETAIL])
        roads_geom, bridge_lines = _build_roads(map_data.roads, min_rank)
    else:
        roads_geom, bridge_lines = None, []
    parks_geom = _buffer_layer(map_data.parks, lambda r: None, 0.0) if include_parks else None
    water_geom = shp_translate(water_geom, -cx, -cy) if water_geom is not None else None
    roads_geom = shp_translate(roads_geom, -cx, -cy) if roads_geom is not None else None
    parks_geom = shp_translate(parks_geom, -cx, -cy) if parks_geom is not None else None
    bridge_lines = [(shp_translate(line, -cx, -cy), w) for line, w in bridge_lines]

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

    # With terrain, buildings are draped onto the surface individually, so we must
    # not union footprints into shared height bins (each needs its own ground).
    use_terrain = terrain is not None
    elev_ref = terrain.elev_min if use_terrain else 0.0

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
            tile.bridges = _bridge_lines_to_tile(bridge_lines, tile_box, tcx, tcy, scale)

            if use_terrain:
                tile.terrain = terrain_mod.build_tile_terrain(
                    terrain, tcx_m=tcx, tcy_m=tcy, scale=scale,
                    tile_w_mm=tile_w_mm, tile_h_mm=tile_h_mm, base_mm=preset["base_mm"],
                    elev_ref=elev_ref, exaggeration=terrain_exaggeration,
                    cap_mm=terrain_cap_mm, res=TERRAIN_TILE_RES,
                )

            if bulk is not None and not bulk.empty:
                in_tile = gpd.clip(bulk, tile_box)
                in_tile = in_tile[in_tile.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
                tile.building_bins = _bin_buildings(
                    in_tile, tcx, tcy, scale, z_mult, max_height_mm, union=not use_terrain)

            if detail is not None and not detail.empty:
                in_tile = gpd.clip(detail, tile_box)
                in_tile = in_tile[in_tile.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
                generic, records = _build_tile_detail(in_tile, tcx, tcy, scale)
                tile.detail_buildings = generic
                tile.landmarks = records

            results.append(tile)

    return results


def _bin_buildings(in_tile: gpd.GeoDataFrame, tcx, tcy, scale, z_mult, max_height_mm, union=True):
    """Bulk buildings as (height_mm, geometry).

    ``union=True`` merges overlapping footprints per height bin (flat base, kills
    internal faces). ``union=False`` keeps each building separate — required for
    terrain, where every building is draped onto its own ground elevation.
    """
    if in_tile.empty:
        return []
    entries = []
    bins: dict[float, list] = {}
    for _, row in in_tile.iterrows():
        geom = _to_tile_mm(row.geometry, tcx, tcy, scale)
        if geom is None:
            continue
        height_mm = _soft_cap(_as_float(row.get("height_m")) * z_mult * scale, max_height_mm)
        height_mm = max(MIN_BUILDING_MM, height_mm)
        key = max(MIN_BUILDING_MM, round(round(height_mm / HEIGHT_BIN_MM) * HEIGHT_BIN_MM, 3))
        if union:
            bins.setdefault(key, []).append(geom)
        else:
            entries.append((key, geom))
    if not union:
        return entries
    out = []
    for height_mm, geoms in sorted(bins.items()):
        merged = _only_polygons(unary_union(geoms))
        if merged is not None and not merged.is_empty:
            out.append((height_mm, merged))
    return out


def _assign_detail_heights(detail: gpd.GeoDataFrame, mult: float, scale: float, cap: float):
    """Add printed ``pz_b`` / ``pz_t`` (mm) columns to landmark/part features.

    A landmark's overall top is soft-capped to the printable ceiling, but each of
    its 3D parts is then scaled *proportionally to that landmark's own real height*
    (``cluster_h_m``). That keeps the tapered silhouette (setbacks, spire) intact
    instead of letting ``tanh`` saturation flatten every upper part to one level.
    """
    detail = detail.copy()
    pz_b = pd.Series(0.0, index=detail.index)
    pz_t = pd.Series(0.0, index=detail.index)
    for idx, row in detail.iterrows():
        h_real = _as_float(row.get("cluster_h_m")) or _as_float(row.get("zt_m")) or 1.0
        target = _soft_cap(h_real * mult * scale, cap)
        pz_t[idx] = target * (_as_float(row.get("zt_m")) / h_real)
        pz_b[idx] = target * (max(0.0, _as_float(row.get("zb_m"))) / h_real)
    detail["pz_b"] = pz_b
    detail["pz_t"] = pz_t
    return detail


def _build_tile_detail(in_tile: gpd.GeoDataFrame, tcx, tcy, scale):
    """Split cropped detail rows into generic part-pieces and landmark records.

    Returns (generic_pieces, landmark_records) where generic_pieces is a list of
    ``(z_bottom_mm, z_top_mm, geom)`` and landmark_records is a list of
    ``LandmarkRecord`` (walls + optional procedural roof, ready for model injection).
    """
    if in_tile.empty:
        return [], []

    generic: list[tuple[float, float, object]] = []
    groups: dict[str, dict] = {}
    for _, row in in_tile.iterrows():
        geom = _to_tile_mm(row.geometry, tcx, tcy, scale)
        if geom is None:
            continue
        zb = max(0.0, _as_float(row.get("pz_b")))
        zt = _as_float(row.get("pz_t"))
        if zt - zb < MIN_PART_MM:
            zt = zb + MIN_PART_MM

        key = row.get("lm_key")
        if key is None or (isinstance(key, float) and pd.isna(key)):
            generic.append((zb, zt, geom))
            continue

        g = groups.get(key)
        if g is None:
            qid = row.get("lm_qid")
            name = row.get("lm_name")
            shape = row.get("roof_shape")
            g = {"geoms": [], "pieces": [], "zt_max": 0.0, "n": 0,
                 "qid": str(qid) if _truthy(qid) else None,
                 "name": str(name) if _truthy(name) else None,
                 "roof_shape": str(shape) if _truthy(shape) else None,
                 "roof_frac": _as_float(row.get("roof_frac")),
                 "roof_dir": row.get("roof_dir")}
            groups[key] = g
        g["geoms"].append(geom)
        g["pieces"].append((zb, zt, geom))
        g["zt_max"] = max(g["zt_max"], zt)
        g["n"] += 1

    records: list[LandmarkRecord] = []
    for key, g in groups.items():
        footprint = _only_polygons(unary_union(g["geoms"]))
        if footprint is None or footprint.is_empty:
            continue
        minx, miny, maxx, maxy = footprint.bounds
        centroid = footprint.centroid
        height = g["zt_max"]

        wall_pieces = g["pieces"]
        roof_shape = None
        roof_base = roof_h = 0.0
        roof_dir = g["roof_dir"]
        # Procedural roof only for single-footprint landmarks (e.g. a domed
        # cathedral) — towers built from stacked parts model their own top.
        frac = g["roof_frac"]
        if g["n"] == 1 and g["roof_shape"] and frac > 0:
            zb0, zt0, geom0 = g["pieces"][0]
            wall_top = zb0 + (zt0 - zb0) * (1.0 - frac)
            wall_pieces = [(zb0, wall_top, geom0)]
            roof_shape = g["roof_shape"]
            roof_base = wall_top
            roof_h = zt0 - wall_top

        records.append(LandmarkRecord(
            key=str(key), name=g["name"], qid=g["qid"],
            centroid_mm=(float(centroid.x), float(centroid.y)),
            target_height_mm=float(height),
            max_footprint_mm=float(max(maxx - minx, maxy - miny)),
            footprint_mm=footprint, wall_pieces=wall_pieces,
            roof_shape=roof_shape, roof_base_mm=float(roof_base),
            roof_height_mm=float(roof_h), roof_direction=roof_dir,
        ))
    return generic, records
