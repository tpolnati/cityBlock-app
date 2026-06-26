"""2D geometry processing for LuminaMaps (Step 2).

Pure ``shapely`` / ``geopandas`` operations. This module turns the raw, projected
OSM layers (in UTM metres, produced by ``osm_fetcher``) into clean, millimetre-space
geometry ready for extrusion by ``mesh_generator``.

Pipeline implemented here
-------------------------
1. Building-height resolution (``building:levels`` / ``height`` tags; sensible
   random fallback) — *no footprint culling*.
2. Dynamic crop-box sizing matched to the preset's physical aspect ratio.
3. Strict bounding-box cropping (boolean intersection) so nothing overhangs.
4. LineString -> Polygon buffering for roads / waterways (width by class).
5. ``unary_union`` of overlapping footprints (grouped by height bin) to kill
   internal faces while preserving height variation.
6. Microscopic ``simplify`` to drop redundant colinear nodes only.
7. Re-centre + scale into millimetres, sliced into tiles when the preset asks
   for it (Tier 3 mega map -> 2x2 grid).

Everything is returned as ``TileGeom`` objects (one per printed STL tile).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import geopandas as gpd
from pyproj import Transformer
from shapely.affinity import scale as shp_scale
from shapely.affinity import translate as shp_translate
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.ops import unary_union

from src.osm_fetcher import MapData

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
LEVEL_HEIGHT_M = 3.0            # 1 building level == 3 metres (per brief)
DEFAULT_MIN_M = 3.0            # random fallback height range (metres)
DEFAULT_MAX_M = 9.0
SIMPLIFY_M = 0.20             # microscopic node-removal tolerance (metres)
HEIGHT_BIN_MM = 0.5            # height quantisation for footprint unioning
MIN_BUILDING_MM = 0.6            # floor on printed building height (mm)

# Road carve widths by OSM highway class (full width, metres).
ROAD_WIDTH_M = {
    "motorway": 16.0, "motorway_link": 10.0,
    "trunk": 14.0, "primary": 12.0, "primary_link": 8.0,
    "secondary": 10.0, "secondary_link": 7.0,
    "tertiary": 8.0, "residential": 6.0, "unclassified": 6.0,
    "living_street": 5.0, "service": 4.0, "pedestrian": 4.0,
}
ROAD_WIDTH_DEFAULT_M = 6.0

# Waterway widths by class (full width, metres). Polygon water bodies keep their
# real footprint; only line features get buffered.
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

    name: str                                   # e.g. "tile" or "tile_r0_c1"
    size_w_mm: float                            # tile width (mm)
    size_h_mm: float                            # tile height (mm)
    base_mm: float                              # base-plate thickness (mm)
    # (height_mm, geometry) — footprints already unioned within each height bin.
    building_bins: list[tuple[float, object]] = field(default_factory=list)
    water: object | None = None                 # shapely geometry (mm) or None
    roads: object | None = None
    parks: object | None = None

    @property
    def building_count_bins(self) -> int:
        return len(self.building_bins)


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
# Height resolution
# ---------------------------------------------------------------------------
def _parse_float(value) -> float | None:
    """Best-effort parse of an OSM numeric tag like '12', '12 m', '12.5'."""
    if value is None:
        return None
    try:
        token = str(value).strip().split()[0].replace(",", ".")
        return float(token)
    except (ValueError, IndexError):
        return None


def resolve_heights(buildings: gpd.GeoDataFrame, seed: int | None = None) -> list[float]:
    """Return a real-world height in metres (pre Z-multiplier) for each building.

    Priority: ``height`` tag -> ``building:levels`` x 3 m -> random 3–9 m.
    """
    rng = random.Random(seed)
    has_height = "height" in buildings.columns
    has_levels = "building:levels" in buildings.columns

    heights: list[float] = []
    for _, row in buildings.iterrows():
        h = _parse_float(row["height"]) if has_height else None
        if (h is None or h <= 0) and has_levels:
            lv = _parse_float(row["building:levels"])
            if lv and lv > 0:
                h = lv * LEVEL_HEIGHT_M
        if h is None or h <= 0:
            h = rng.uniform(DEFAULT_MIN_M, DEFAULT_MAX_M)
        heights.append(float(h))
    return heights


# ---------------------------------------------------------------------------
# Line -> polygon buffering
# ---------------------------------------------------------------------------
def _buffer_layer(gdf: gpd.GeoDataFrame, width_for_row, default_width: float):
    """Convert a mixed line/polygon layer to a single unioned polygon geometry.

    LineStrings are buffered by half their class width; existing polygons are
    kept (cleaned with ``buffer(0)``). Returns a shapely geometry or ``None``.
    """
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
        # Points / other geometries are not printable land features -> skip.

    if not pieces:
        return None
    merged = unary_union(pieces)
    return merged if not merged.is_empty else None


def _road_width(row) -> float:
    hw = row.get("highway")
    if isinstance(hw, list):
        hw = hw[0] if hw else None
    return ROAD_WIDTH_M.get(hw, ROAD_WIDTH_DEFAULT_M)


def _water_width(row) -> float:
    ww = row.get("waterway")
    if isinstance(ww, list):
        ww = ww[0] if ww else None
    return WATERWAY_WIDTH_M.get(ww, WATERWAY_WIDTH_DEFAULT_M)


# ---------------------------------------------------------------------------
# Small geometry utilities
# ---------------------------------------------------------------------------
def _only_polygons(geom):
    """Keep only Polygon/MultiPolygon parts of a possibly mixed geometry."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return geom
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            return None
        return unary_union(polys)
    return None


def _to_tile_mm(geom, tcx_m: float, tcy_m: float, scale: float):
    """Recentre a metres-space geometry on a tile centre and scale to millimetres."""
    if geom is None or geom.is_empty:
        return None
    g = shp_translate(geom, xoff=-tcx_m, yoff=-tcy_m)
    g = shp_scale(g, xfact=scale, yfact=scale, origin=(0, 0))
    return g if not g.is_empty else None


def _crop(geom, tile_box: Polygon):
    """Strict bounding-box crop (boolean intersection)."""
    if geom is None or geom.is_empty:
        return None
    clipped = geom.intersection(tile_box)
    return _only_polygons(clipped)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def prepare_tiles(
    map_data: MapData,
    preset: dict,
    z_mult: float,
    *,
    include_water: bool = True,
    include_roads: bool = True,
    include_parks: bool = True,
    seed: int | None = 42,
) -> list[TileGeom]:
    """Turn fetched OSM layers into a list of millimetre-space ``TileGeom`` tiles.

    Parameters mirror the preset (size, base, tiling) and the live Z-multiplier
    slider. Re-running this with a new ``z_mult`` recomputes geometry only — it
    never touches the network (the fetch is cached upstream).
    """
    cx, cy = project_center(map_data.center_latlon, map_data.utm_crs)

    # --- Buildings: resolve heights, recentre on full-crop centre, simplify ---
    buildings = map_data.buildings
    buildings = buildings[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if not buildings.empty:
        buildings["height_m"] = resolve_heights(buildings, seed=seed)
        geom = buildings.geometry.buffer(0)                       # fix invalid rings
        geom = geom.translate(xoff=-cx, yoff=-cy)                  # full-crop centred
        geom = geom.simplify(SIMPLIFY_M, preserve_topology=True)   # node cleanup only
        buildings = buildings.set_geometry(geom)
        buildings = buildings[~buildings.geometry.is_empty]

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

    results: list[TileGeom] = []
    for j in range(tiles_y):                       # rows (top -> bottom)
        for i in range(tiles_x):                   # cols (left -> right)
            tcx = -half_w_m + tile_w_m * (i + 0.5)
            tcy = half_h_m - tile_h_m * (j + 0.5)   # row 0 = top (max northing)
            tile_box = box(
                tcx - tile_w_m / 2.0, tcy - tile_h_m / 2.0,
                tcx + tile_w_m / 2.0, tcy + tile_h_m / 2.0,
            )

            name = "tile" if (tiles_x * tiles_y) == 1 else f"tile_r{j}_c{i}"
            tile = TileGeom(
                name=name,
                size_w_mm=tile_w_mm,
                size_h_mm=tile_h_mm,
                base_mm=preset["base_mm"],
                water=_to_tile_mm(_crop(water_geom, tile_box), tcx, tcy, scale),
                roads=_to_tile_mm(_crop(roads_geom, tile_box), tcx, tcy, scale),
                parks=_to_tile_mm(_crop(parks_geom, tile_box), tcx, tcy, scale),
            )

            # Buildings: crop in metres, transform to mm, bin by height, union.
            if not buildings.empty:
                in_tile = gpd.clip(buildings, tile_box)
                in_tile = in_tile[in_tile.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
                tile.building_bins = _bin_buildings(in_tile, tcx, tcy, scale, z_mult)

            results.append(tile)

    return results


def _bin_buildings(in_tile: gpd.GeoDataFrame, tcx: float, tcy: float, scale: float, z_mult: float):
    """Group cropped buildings by quantised printed height and union each group.

    Unioning overlapping footprints *within a height bin* removes internal faces
    (per the brief) while preserving genuine height variation between bins. XY
    footprint detail is untouched.
    """
    if in_tile.empty:
        return []

    bins: dict[float, list] = {}
    for _, row in in_tile.iterrows():
        geom = _to_tile_mm(row.geometry, tcx, tcy, scale)
        if geom is None:
            continue
        height_mm = max(MIN_BUILDING_MM, row["height_m"] * z_mult * scale)
        key = round(round(height_mm / HEIGHT_BIN_MM) * HEIGHT_BIN_MM, 3)
        key = max(MIN_BUILDING_MM, key)
        bins.setdefault(key, []).append(geom)

    out: list[tuple[float, object]] = []
    for height_mm, geoms in sorted(bins.items()):
        merged = _only_polygons(unary_union(geoms))
        if merged is not None and not merged.is_empty:
            out.append((height_mm, merged))
    return out
