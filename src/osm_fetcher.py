"""OSM data retrieval, validation, and projection for LuminaMaps.

This module is the *only* place that talks to the internet (via osmnx /
Overpass). Everything here is wrapped in Streamlit caching so that touching a
UI slider in app.py never re-triggers a network download — the heavy fetch is
keyed purely on (latitude, longitude, radius).

Design notes
------------
* osmnx 2.x API is assumed (``ox.features_from_point`` / ``ox.geocode`` /
  ``ox.projection``).
* All geometry is projected into a *single* dynamically-chosen local UTM CRS so
  that buildings, water, roads and parks stay perfectly aligned and are in
  metres (the unit we later scale down to millimetres for printing).
* Failures for any one feature layer are non-fatal: an empty GeoDataFrame is
  returned for that layer so the rest of the map can still be built. Only the
  *buildings* layer being empty is treated as a hard error (handled in app.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import geopandas as gpd
import osmnx as ox
import streamlit as st

# ---------------------------------------------------------------------------
# osmnx global configuration
# ---------------------------------------------------------------------------
# Use osmnx's on-disk HTTP cache as a second line of defence on top of
# Streamlit's in-memory cache. Keep the console quiet inside Streamlit.
ox.settings.use_cache = True
ox.settings.log_console = False
# Be a little patient: dense urban queries can take a while on Overpass.
ox.settings.requests_timeout = 180


# OSM tag filters for each feature layer. Kept deliberately *broad* — per the
# project brief we must NOT cull small/unique footprints, so we pull every
# building and a wide net of land features.
#
# We always pull `building:part` too (the "Simple 3D Buildings" scheme). These
# are the per-section polygons — each with its own `height` / `min_height` —
# that encode the setbacks, tapers and spires of landmarks like the Burj
# Khalifa. They are cheap to fetch and let the Detail Level control decide
# whether to *use* them, so toggling detail never triggers a re-download.
BUILDING_TAGS = {"building": True, "building:part": True}

WATER_TAGS = {
    "natural": ["water", "bay", "strait"],
    "waterway": ["river", "stream", "canal", "riverbank", "dock"],
    "landuse": ["reservoir", "basin"],
}

ROAD_TAGS = {
    "highway": [
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "residential", "unclassified", "service", "living_street",
        "pedestrian", "motorway_link", "primary_link", "secondary_link",
    ],
}

# Standalone attractions/monuments that are NOT buildings — statues, memorials,
# obelisks, towers, lighthouses… (e.g. the Statue of Liberty). These come as
# points or small polygons and are routed into the landmark pipeline so they can
# receive a detailed downloaded model.
ATTRACTION_TAGS = {
    "tourism": ["attraction", "monument", "artwork", "viewpoint"],
    "historic": ["monument", "memorial", "castle", "fort", "citadel",
                 "ruins", "archaeological_site", "tower", "city_gate"],
    "man_made": ["monument", "tower", "obelisk", "statue", "lighthouse",
                 "communications_tower", "campanile"],
}

PARK_TAGS = {
    "leisure": ["park", "garden", "recreation_ground", "pitch", "playground"],
    "landuse": ["grass", "recreation_ground", "village_green", "meadow", "forest"],
    "natural": ["wood", "scrub", "grassland"],
}


# ---------------------------------------------------------------------------
# Container for a fully-fetched, projected map
# ---------------------------------------------------------------------------
@dataclass
class MapData:
    """Everything app.py needs from a single fetch, already in local UTM metres."""

    center_latlon: tuple[float, float]          # (lat, lon) used for the query
    radius_m: float                             # requested radius in metres
    utm_crs: str                                # e.g. "EPSG:32633"
    buildings: gpd.GeoDataFrame
    water: gpd.GeoDataFrame
    roads: gpd.GeoDataFrame
    parks: gpd.GeoDataFrame
    attractions: gpd.GeoDataFrame = None      # non-building monuments/statues
    notes: list[str] = field(default_factory=list)  # non-fatal fetch warnings

    @property
    def attraction_count(self) -> int:
        return 0 if self.attractions is None else len(self.attractions)

    # --- convenience accessors -------------------------------------------------
    @property
    def building_count(self) -> int:
        return 0 if self.buildings is None else len(self.buildings)

    @property
    def water_count(self) -> int:
        return 0 if self.water is None else len(self.water)

    @property
    def road_count(self) -> int:
        return 0 if self.roads is None else len(self.roads)

    @property
    def park_count(self) -> int:
        return 0 if self.parks is None else len(self.parks)

    @property
    def has_buildings(self) -> bool:
        return self.building_count > 0

    @property
    def bounds_m(self) -> Optional[tuple[float, float, float, float]]:
        """(minx, miny, maxx, maxy) in UTM metres across all non-empty layers."""
        frames = [
            g for g in (self.buildings, self.water, self.roads, self.parks)
            if g is not None and not g.empty
        ]
        if not frames:
            return None
        bounds = [f.total_bounds for f in frames]
        minx = min(b[0] for b in bounds)
        miny = min(b[1] for b in bounds)
        maxx = max(b[2] for b in bounds)
        maxy = max(b[3] for b in bounds)
        return (minx, miny, maxx, maxy)


# ---------------------------------------------------------------------------
# UTM helpers
# ---------------------------------------------------------------------------
def utm_crs_from_latlon(lat: float, lon: float) -> str:
    """Return the EPSG code for the WGS84 / UTM zone containing (lat, lon).

    Northern hemisphere -> 326xx, southern -> 327xx, where xx is the zone.
    Computing this ourselves (rather than letting each layer pick its own zone
    via ``project_gdf``) guarantees every layer shares one identical CRS.
    """
    zone = int((lon + 180.0) / 6.0) + 1
    zone = max(1, min(60, zone))
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


# ---------------------------------------------------------------------------
# Geocoding (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def geocode_location(query: str) -> tuple[float, float]:
    """Geocode a free-text location (city / neighbourhood / address) to (lat, lon).

    Raises
    ------
    ValueError
        If the location cannot be geocoded.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("Please enter a location.")
    try:
        lat, lon = ox.geocode(query)
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the UI
        raise ValueError(
            f"Could not find '{query}'. Try a more specific address or city name."
        ) from exc
    return float(lat), float(lon)


# ---------------------------------------------------------------------------
# Low-level single-layer fetch (not cached on its own; called by fetch_map_data)
# ---------------------------------------------------------------------------
def _fetch_features(lat: float, lon: float, radius_m: float, tags: dict) -> gpd.GeoDataFrame:
    """Fetch one feature layer as a GeoDataFrame in EPSG:4326.

    Returns an empty GeoDataFrame (never raises) so a single missing layer can't
    sink the whole fetch.
    """
    try:
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=radius_m)
    except Exception:  # noqa: BLE001 - e.g. osmnx InsufficientResponseError in rural areas
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Drop rows with no geometry; keep the index for traceability.
    gdf = gdf[~gdf.geometry.isna()].copy()
    return gdf


def _project(gdf: gpd.GeoDataFrame, utm_crs: str) -> gpd.GeoDataFrame:
    """Project a (possibly empty) GeoDataFrame into the shared UTM CRS."""
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=utm_crs)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(utm_crs)


# ---------------------------------------------------------------------------
# Top-level cached fetch
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_map_data(lat: float, lon: float, radius_m: float) -> MapData:
    """Fetch + project all map layers for a point. Cached on (lat, lon, radius).

    This is the expensive, network-bound call. Because it is wrapped in
    ``st.cache_data`` and keyed only on the geographic inputs, adjusting any
    downstream UI control (Z-multiplier, preset, etc.) will reuse this result
    instead of hitting Overpass again.
    """
    utm_crs = utm_crs_from_latlon(lat, lon)
    notes: list[str] = []

    buildings_ll = _fetch_features(lat, lon, radius_m, BUILDING_TAGS)
    water_ll = _fetch_features(lat, lon, radius_m, WATER_TAGS)
    roads_ll = _fetch_features(lat, lon, radius_m, ROAD_TAGS)
    parks_ll = _fetch_features(lat, lon, radius_m, PARK_TAGS)
    attractions_ll = _fetch_features(lat, lon, radius_m, ATTRACTION_TAGS)

    if water_ll.empty:
        notes.append("No waterways/water bodies found in this area.")
    if roads_ll.empty:
        notes.append("No major roads found in this area.")
    if parks_ll.empty:
        notes.append("No parks/green spaces found in this area.")

    return MapData(
        center_latlon=(lat, lon),
        radius_m=radius_m,
        utm_crs=utm_crs,
        buildings=_project(buildings_ll, utm_crs),
        water=_project(water_ll, utm_crs),
        roads=_project(roads_ll, utm_crs),
        parks=_project(parks_ll, utm_crs),
        attractions=_project(attractions_ll, utm_crs),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# "Notable attractions nearby" hint (looks in a wider ring than the crop)
# ---------------------------------------------------------------------------
def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _compass(lat1, lon1, lat2, lon2) -> str:
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
         - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon))
    brng = (math.degrees(math.atan2(y, x)) + 360) % 360
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((brng + 22.5) % 360 // 45)]


@st.cache_data(show_spinner=False)
def nearby_attractions(lat: float, lon: float, outer_radius_m: float) -> list[dict]:
    """Notable (named + Wikidata-linked) attractions within ``outer_radius_m``.

    Used to hint when a famous landmark sits just outside the chosen crop radius.
    Returns [{name, qid, dist_m, compass}], de-duplicated by Wikidata id, nearest
    first. Never raises — returns [] on any failure.
    """
    try:
        gdf = _fetch_features(lat, lon, outer_radius_m, ATTRACTION_TAGS)
    except Exception:  # noqa: BLE001
        return []
    if gdf is None or gdf.empty:
        return []

    seen: set[str] = set()
    out: list[dict] = []
    for _, row in gdf.iterrows():
        name = row["name"] if "name" in row.index else None
        qid = row["wikidata"] if "wikidata" in row.index else None
        if not name or not qid or not isinstance(name, str) or not isinstance(qid, str):
            continue
        if qid in seen:
            continue
        try:
            c = row.geometry.centroid
            dist = _haversine_m(lat, lon, c.y, c.x)
        except Exception:  # noqa: BLE001
            continue
        seen.add(qid)
        out.append({"name": name.strip(), "qid": qid.strip(),
                    "dist_m": dist, "compass": _compass(lat, lon, c.y, c.x)})
    out.sort(key=lambda a: a["dist_m"])
    return out
